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
Cosine-similarity retrieval
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
- `rag.py`: coordinates indexing, retrieval, unsupported-question handling, and citations.
- `application.py`: assembles production adapters and the indexing workflow.
- `web.py`: exposes health, document upload, and question endpoints.
- `evaluation.py`: measures retrieval hit rate and answer presence on a fixed dataset.

The important seam is `RAGAssistant`. Its callers do not need to know whether vectors are held in memory or SQLite, or whether embeddings come from OpenAI or a fake test adapter.

## Technical trade-offs

- SQLite plus Python cosine search keeps the first production version easy to inspect and run locally. It will not scale to millions of chunks; a dedicated vector database can replace the store later.
- Character windows with overlap are simple and preserve page citations. A later version can add token-aware or structure-aware chunking.
- A similarity threshold lets the assistant decline unsupported questions, but it is only a heuristic. Evaluation data is needed to tune it honestly.
- The model receives retrieved excerpts rather than the entire PDF. This reduces context size and makes citations possible, but answer quality depends on extraction, chunking, embeddings, and retrieval.

## Current limitations

- No user authentication or per-user document isolation.
- SQLite is local and the vector search is linear.
- Scanned/image-only PDFs need OCR, which is not implemented.
- The API has an upload-size limit but no malware scanning or rate limiting.
- The evaluation currently measures retrieval coverage and answer presence, not factual correctness.
