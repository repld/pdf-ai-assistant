# AI Document Assistant

An educational project that will grow into a document question-answering application. Users will eventually upload PDFs, ask questions, and receive answers grounded in the documents with source citations.

This repository is separate from the existing breast-cancer data project. It has its own Python package, tests, environment file, and Git history.

## Current status

The project now supports direct AI questions, PDF extraction, overlapping chunks, embeddings, cosine-similarity retrieval, SQLite persistence, grounded RAG answers, source-page citations, evaluation, a FastAPI interface, Docker configuration, and CI checks.

The current reliability safeguards require model answers to include valid numbered citations. Answers without valid citations are replaced with the unsupported-document response.

## Milestone 1: one AI question

The first milestone proves the smallest useful application loop:

```text
terminal question → ask_question() → OpenAI Responses API → terminal answer
```

The test uses a fake client, so it does not contact the API or incur usage costs.

## Setup

Requirements: Python 3.12+. The project installs `pypdf` with its font-parsing extra so PDFs with embedded fonts have better text extraction.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

For the recommended OpenRouter setup, follow [docs/openrouter-setup.md](docs/openrouter-setup.md). Keep `.env` private; it is ignored by Git.

## Run

Run the network-free test:

```powershell
python -m pytest
```

After configuring an API key, make one real request:

```powershell
python -m ai_document_assistant "Explain what an API is in one paragraph."
```

Or omit the question to start interactive mode:

```powershell
python -m ai_document_assistant
```

Type `quit` or `exit` to stop.

To index one or more PDFs and ask grounded questions:

```powershell
python -m ai_document_assistant --pdf .\path\to\document.pdf "What is the main idea?"
```

Use `--pdf` more than once to index multiple documents. Indexing calls the embeddings API, and questions call the model API, so both activities can incur API usage.

Persist the index in SQLite so it can be reused:

```powershell
python -m ai_document_assistant --database data\assistant.db --pdf .\path\to\document.pdf
python -m ai_document_assistant --database data\assistant.db "What is the main idea?"
```

For a small first experiment, limit indexing to the first few pages:

```powershell
python -m ai_document_assistant --database data\assistant.db --max-pages 3 --pdf "book.pdf" "What is the main idea?"
```

To index a specific page range, such as pages 30–32:

```powershell
python -m ai_document_assistant --database data\assistant.db --page-start 30 --page-end 32 --pdf "book.pdf" "What do these pages discuss?"
```

The example evaluation file is [data/evaluation.example.json](data/evaluation.example.json). Replace its placeholder chunk ID with a retrieved chunk ID, then run:

```powershell
python -m ai_document_assistant --database data\assistant.db --evaluate data\evaluation.json
```

Evaluation requests use embeddings and model calls, so they can incur API usage.

The included Chapter 1 evaluation set covers pages 33–55 of the local Géron reference book:

```powershell
python -m ai_document_assistant `
  --database ":memory:" `
  --page-start 33 `
  --page-end 55 `
  --pdf "book.pdf" `
  --evaluate data\chapter1-evaluation.json
```

The report includes retrieval hit rate, answer rate, citation rate, and one line per case. Use a small page range while learning because indexing and evaluation use API requests.

## HTTP interface

Install the web dependencies:

```powershell
python -m pip install -e ".[dev,web]"
```

Start the local API:

```powershell
uvicorn ai_document_assistant.web:app --reload
```

Endpoints:

- `GET /health` — health check.
- `POST /documents` — upload a PDF as multipart form field `file`.
- `POST /ask` — JSON body such as `{"question": "What is the main idea?"}`.

The same API can be started with Docker Compose after configuring `.env`:

```powershell
docker compose up --build
```

## Milestone 2: robust interaction

The second milestone adds input validation, missing-configuration errors, empty-response checks, API error handling, and a reusable interactive question loop. The tests still use fake clients and do not contact the API.

The document milestone now adds PDF text extraction, overlapping chunks, in-memory vector similarity search, RAG context construction, unsupported-question handling, and source-page citations.

Real requests use API credits. This project does not make a live request automatically during tests.

## Learning plan

See [docs/learning-plan.md](docs/learning-plan.md) for the high-level architecture, first-week plan, and book topics to study before the next feature.

See [docs/architecture.md](docs/architecture.md) for the complete data flow, module responsibilities, trade-offs, and current limitations.

For a detailed beginner-friendly explanation of the code and self-check quiz, read [docs/learning-walkthrough.md](docs/learning-walkthrough.md).

## Continuous integration

GitHub Actions runs the test suite and Ruff on pushes and pull requests. The repository intentionally excludes `.env`, local databases, virtual environments, and reference PDFs.
