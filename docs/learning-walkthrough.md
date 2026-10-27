# Learning walkthrough

This guide explains what the project does and how the files connect. Read it alongside the code, not instead of running the code.

## Recommended study loop

For each section:

1. Read the explanation.
2. Open the named file.
3. Run the related tests.
4. Change one small thing and run the tests again.
5. Explain the data flow in your own words.

Start with fake tests. They are fast and do not require an API key.

```powershell
python -m pytest -p no:anyio
```

## 1. Direct question answering

Start with:

- `src/ai_document_assistant/__main__.py`
- `src/ai_document_assistant/assistant.py`
- `tests/test_assistant.py`

The flow is:

```text
command-line text
    → main()
    → ask_question()
    → client.responses.create()
    → response.output_text
    → terminal output
```

`ask_question()` is the first important module interface. It accepts a question and returns a string. Internally it validates input, loads configuration, creates the OpenAI client, sends the request, and rejects an empty response.

The `client` parameter is injectable. Production code uses the real client; tests use `FakeClient`. This is dependency injection: the function receives a dependency instead of constructing it unconditionally. That is why the test can check our code without contacting the network.

## 2. Interactive mode

Read `src/ai_document_assistant/cli.py`.

`run_interactive()` uses a `while` loop because it needs to accept more than one question. Each iteration:

1. Reads input.
2. Checks for `quit` or `exit`.
3. Rejects blank input.
4. Calls the question-answering function.
5. Displays either an answer or an error.

The `ask`, `input_fn`, and `output_fn` parameters are seams for testing. Tests replace them with small fake functions, so they can simulate a conversation without a person typing or a model responding.

## 3. PDF extraction and chunks

Read:

- `src/ai_document_assistant/documents.py`
- `tests/test_documents.py`

`extract_pdf_pages()` uses `pypdf` to turn a PDF into `DocumentPage` objects:

```text
PDF
    → page 1 text
    → page 2 text
    → page 3 text
```

Each page keeps its source filename and page number. That metadata is essential later for citations.

`chunk_pages()` divides long pages into overlapping `DocumentChunk` objects. Overlap means the end of one chunk appears at the beginning of the next. This reduces the chance that an important sentence is split across two unrelated retrieval units.

The chunk is deliberately more than a string. It contains:

- `chunk_id`
- `source`
- `page_number`
- `text`

The text is used for retrieval; the other fields are used to explain where an answer came from.

## 4. Embeddings and retrieval

Read:

- `src/ai_document_assistant/providers.py`
- `src/ai_document_assistant/retrieval.py`
- `tests/test_retrieval.py`

An embedding is a list of numbers representing the meaning of text. Similar texts tend to have vectors that point in similar directions.

The embedding flow is:

```text
chunk text → embedding model → vector such as [0.12, -0.04, ...]
question  → embedding model → query vector
```

`OpenAIEmbedder` is an adapter around the provider SDK. The rest of the project only needs its small `embed()` interface.

`InMemoryVectorStore` compares the query vector with every stored chunk vector using cosine similarity. A higher score means the vectors are more similar. The store returns the highest-scoring chunks.

This first version searches every vector in Python. That is easy to understand and test, but it is not ideal for millions of chunks. `SQLiteVectorStore` persists the same information between runs, but still performs a linear Python search. A dedicated vector database could replace it later without changing the RAG interface.

## 5. Retrieval-augmented generation

Read:

- `src/ai_document_assistant/rag.py`
- `src/ai_document_assistant/providers.py`
- `tests/test_rag.py`

RAG means retrieval-augmented generation. The application does not send the entire PDF to the model. It first retrieves likely relevant chunks, then sends only those chunks with the question.

The flow is:

```text
question
    → question embedding
    → similarity search
    → relevant chunks
    → grounded prompt
    → model answer
    → citations
```

`RAGAssistant` coordinates this flow. If no chunk reaches the similarity threshold, it returns:

```text
I could not find an answer in the uploaded documents.
```

That is an important reliability behavior. The assistant should not pretend that every question is answerable from the uploaded documents.

`OpenAIAnswerer` tells the model to use only the retrieved excerpts and cite numbered sources. The application also returns structured citation metadata containing the source filename and page number.

## 6. Persistence and evaluation

Read:

- `src/ai_document_assistant/persistence.py`
- `src/ai_document_assistant/evaluation.py`
- `tests/test_persistence.py`
- `tests/test_evaluation.py`

SQLite stores each chunk, its metadata, and its embedding as JSON. This means the indexed document can be reused after the process stops.

The evaluation module runs a fixed set of questions and reports:

- Retrieval hit rate: did the answer cite an expected chunk?
- Answer rate: did the system return non-empty text?

These are useful baseline metrics, but they do not prove that the answer is factually correct. A later evaluation system should also measure groundedness and answer quality.

## 7. HTTP interface

Read:

- `src/ai_document_assistant/web.py`
- `tests/test_web.py`

FastAPI exposes the application through HTTP:

- `GET /health` checks that the process is alive.
- `POST /documents` accepts a PDF, extracts it, chunks it, and indexes it.
- `POST /ask` accepts a question and returns an answer plus citations.

The web app accepts an assistant during creation. Tests inject a fake assistant. Production lazily creates the configured assistant on the first document or question request.

## 8. What to understand for interviews

You should be able to explain:

- Why chunks need overlap.
- What embeddings represent.
- Why cosine similarity is used.
- How RAG differs from asking a model a general question.
- Why citations depend on preserving page metadata.
- Why a similarity threshold is only a heuristic.
- Why fake adapters are useful in tests.
- Why SQLite is sufficient for a small demo but not a high-scale vector system.
- Why API keys belong in environment variables.
- Why tests should not require a live paid API.

## Self-check quiz

1. What does `response.output_text` contain?
2. Why does `DocumentChunk` store `page_number`?
3. What happens when retrieval finds no chunk above `min_score`?
4. What is the difference between an embedding vector and the final answer text?
5. Which module would you replace if you moved from SQLite to a hosted vector database?
6. Why can a RAG system still produce an incorrect answer even when it has citations?

## Book connection

Géron’s sections on attention and transformers explain how modern language models process relationships between tokens. That helps you understand the model inside the provider adapter, but the project’s main engineering work is around the model: extraction, chunking, retrieval, prompting, evaluation, and reliability.

Use the *Developer’s Guide to AI* sections on AI application architecture, prompting, embeddings, vector databases, RAG, and tool calling as the practical companion when you add those features.
