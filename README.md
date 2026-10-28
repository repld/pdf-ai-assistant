# AI Document Assistant

An AI-powered document question-answering application. Upload a PDF, ask questions about its contents, and receive grounded answers with source-page citations.

Live demo: [pdfaiassist.streamlit.app](https://pdfaiassist.streamlit.app/)

## Features

- Streamlit browser interface for PDF upload and chat
- Reader-focused PDF workspace with a secondary question panel
- Citation-driven page navigation and highlighted supporting excerpts
- Search with previous/next match navigation, table-of-contents navigation, layout controls, and reading mode
- Streaming answers when supported, token diagnostics, indexing progress, document metadata, and answer feedback
- PDF text extraction with page metadata preserved for citations
- Overlapping text chunking
- Embedding-based similarity search
- Hybrid semantic and keyword retrieval
- Retrieval-augmented generation (RAG)
- Answers grounded in retrieved document excerpts
- Numbered source citations with source filename and page number
- Honest fallback when the answer cannot be supported by the document
- FastAPI endpoints for document upload, questions, and health checks
- Optional SQLite persistence for CLI and API usage
- Automated tests, linting, and GitHub Actions CI

## How it works

```text
PDF upload
    -> text extraction
    -> overlapping chunks with page metadata
    -> embeddings and vector index

Question
    -> question embedding
    -> similarity search
    -> grounded prompt with retrieved excerpts
    -> answer with source citations
```

See [docs/architecture.md](docs/architecture.md) for module responsibilities, design decisions, and limitations.

## Quick start

Requirements: Python 3.12 or newer.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,ui]"
Copy-Item .env.example .env
```

Add your provider settings to `.env`. The recommended configuration is documented in [docs/openrouter-setup.md](docs/openrouter-setup.md). Never commit `.env` or share its API key.

## Run the browser application

```powershell
streamlit run src\ai_document_assistant\ui.py
```

On Windows, double-click `start-assistant.vbs` to start the application without leaving a command window open. Upload a PDF, wait for indexing to finish, and ask questions in the chat interface.

The browser interface prioritizes the book while keeping the question panel available beside it. Citation buttons navigate to the cited page and show the exact retrieved excerpt beside the PDF. The reader includes page navigation, document search, optional outline navigation, a full-screen reading mode, and native viewer controls for zoom and fit-to-width. The conversation uses a fixed-height scrollable panel so long answers and source cards do not push the question input indefinitely down the page. The current document and messages stay in memory for that browser session; restarting the app clears the session.

## Run the command-line application

Ask a direct question:

```powershell
python -m ai_document_assistant "Explain what an API is in one paragraph."
```

Index a PDF and ask a grounded question:

```powershell
python -m ai_document_assistant `
  --database data\assistant.db `
  --max-pages 3 `
  --pdf "path\to\document.pdf" `
  "What is the main idea?"
```

Use `--page-start` and `--page-end` to index a specific page range. Use `--pdf` more than once to index multiple documents.

Indexing uses the embedding model and asking questions uses the chat model, so both operations can incur provider usage.

## Run the API

Install the API dependencies:

```powershell
python -m pip install -e ".[dev,web]"
```

Start FastAPI:

```powershell
uvicorn ai_document_assistant.web:app --reload
```

Endpoints:

- `GET /health` - process health check
- `POST /documents` - upload a PDF as the multipart field `file`
- `POST /ask` - ask a question, for example `{"question": "What is the main idea?"}`

API requests may include an `X-Session-ID` header to isolate document indexes. If
`API_AUTH_TOKEN` is configured, send `Authorization: Bearer <token>` with every
request. Session IDs provide isolation, not identity; use the bearer token and a
real identity provider for public deployments.

Docker support is available through `Dockerfile` and `docker-compose.yml`.

## Evaluation

Create a JSON evaluation file using [data/evaluation.example.json](data/evaluation.example.json), then run:

```powershell
python -m ai_document_assistant `
  --database data\assistant.db `
  --evaluate data\evaluation.json `
  --evaluation-report data\evaluation-report.md
```

The report measures retrieval hits and recall, supported-answer rate, citation coverage,
required-fact coverage, and categorized failures. These metrics are useful signals but
do not replace human review of factual accuracy.

## Observability and security

RAG requests emit structured telemetry for duration, retrieval time, generation time,
retrieved chunk count, citation count, provider-reported token usage when available,
and outcome status. Provider adapters stream answer text when supported while preserving
final citation validation. The telemetry seam can be connected to a production collector
without changing `RAGAssistant`.

The API supports optional bearer authentication, validated session IDs, per-session
SQLite databases, session limits, request limits, upload limits, question limits, and
prompts that treat document text as untrusted data.

## Testing and code quality

The test suite uses fake provider clients for network-free, predictable tests:

```powershell
python -m pytest -p no:anyio
python -m ruff check .
```

GitHub Actions runs these checks for pushes and pull requests.

## Configuration

The application reads configuration from environment variables or `.env`:

```text
AI_PROVIDER=openrouter
OPENROUTER_API_KEY=your-key
OPENROUTER_MODEL=deepseek/deepseek-v3.2
OPENROUTER_EMBEDDING_MODEL=openai/text-embedding-3-small
API_AUTH_TOKEN=change-me-before-public-deployment
```

See [docs/openrouter-setup.md](docs/openrouter-setup.md) for provider setup and [docs/deployment.md](docs/deployment.md) for Streamlit Community Cloud deployment.

## Limitations

- Scanned or image-only PDFs require OCR, which is not included.
- Streamlit sessions use in-memory indexes and are not permanent.
- SQLite and linear vector search are intended for small deployments, not large-scale production traffic.
- Authentication is opt-in through `API_AUTH_TOKEN`; a production deployment should connect it to a real identity provider.
- Session isolation and rate limiting are in-process controls; distributed deployments need shared session and rate-limit storage.
- There is no malware scanning for uploaded files.
- Broad requests such as “list every item in the document” use a wider retrieval window, but they can still be incomplete when items are spread across many pages.

## Repository layout

```text
src/ai_document_assistant/   application code
tests/                       automated tests
data/evaluation.example.json evaluation file template
docs/                        architecture, setup, and deployment documentation
```
