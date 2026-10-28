# Architecture

## End-to-end flow

```text
PDF upload or CLI path
        │
        ▼
PDF text extraction (pypdf)
        │ pages with source/page metadata
        ▼
Overlapping text chunks
        │
        ▼
Embedding adapter ───────► SQLite vector store
        │                         │
Question ─► query embedding ─────┘
        │
        ▼
Hybrid semantic + keyword retrieval
        │ top relevant chunks
        ▼
Grounded Responses API prompt
        │
        ▼
Answer + source-page citations
```

## Module responsibilities

- `documents.py`: extracts PDF text and creates citation-preserving chunks.
- `providers.py`: adapts the OpenAI embeddings and Responses APIs.
- `retrieval.py`: provides deterministic in-memory vector search.
- `persistence.py`: stores chunks and vectors in SQLite between runs.
- `rag.py`: coordinates indexing, retrieval, unsupported-question handling, citation validation, and citations.
- `application.py`: assembles production adapters and the indexing workflow.
- `observability.py`: records structured request timing and outcome telemetry through a small adapter interface.
- `web.py`: exposes health, document upload, and question endpoints.
- `ui.py`: provides the optional Streamlit upload-and-chat interface, side-by-side cited-page viewer, and the same RAG service.
- `evaluation.py`: measures retrieval hit rate, answer presence, and citation presence on a fixed dataset.

The important seam is `RAGAssistant`. Its callers do not need to know whether vectors are held in memory or SQLite, or whether embeddings come from OpenAI or a fake test adapter.

## Technical trade-offs

- SQLite plus Python cosine search keeps the first production version easy to inspect and run locally. It will not scale to millions of chunks; a dedicated vector database can replace the store later.
- Retrieval combines embedding similarity with lightweight keyword overlap. Semantic search handles paraphrases; keyword overlap protects exact names, terms, and list items. A dedicated reranker can replace this implementation later.
- Character windows with overlap are simple and preserve page citations. A later version can add token-aware or structure-aware chunking.
- A similarity threshold lets the assistant decline unsupported questions, but it is only a heuristic. Evaluation data is needed to tune it honestly.
- The model receives retrieved excerpts rather than the entire PDF. This reduces context size and makes citations possible, but answer quality depends on extraction, chunking, embeddings, and retrieval.
- Exhaustive list questions are detected and retrieve up to 20 chunks with a relaxed score threshold. This improves recall without sending the entire document to the model, but very large or widely scattered lists can still be incomplete.
- A model answer without valid numbered citations is discarded and replaced with the unsupported-answer response. This is a safety check, not proof that a cited answer is factually correct.
- The answer object keeps retrieved chunk IDs separate from displayed citations, so evaluation can distinguish a retrieval miss from a model-citation miss.
- The Streamlit interface stores the current assistant and chat messages in per-browser session state. It uses an in-memory index for simple, isolated browser sessions; the CLI and FastAPI paths support SQLite persistence.
- The API isolates indexes by validated session ID and can require a bearer token. Session IDs are isolation keys, not user identity, so public deployments still need a real authentication provider.
- Structured telemetry records latency, counts, and outcomes without coupling the core module to one observability vendor.
- Provider requests use a 30-second timeout and limited SDK retries so a web request does not wait indefinitely.

## Current limitations

- Authentication is optional and in-process; production identity and distributed session storage are still needed.
- SQLite is local and the vector search is linear, even with hybrid ranking.
- Scanned/image-only PDFs need OCR, which is not implemented.
- The API has upload, question, session, and request limits but no malware scanning.
- Evaluation now reports retrieval recall, citation coverage, required-fact coverage, and failure categories, but still requires human or reference-answer review to measure factual correctness.
