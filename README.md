# AI Document Assistant

An AI-powered document question-answering application. Upload a PDF, ask questions about its contents, and receive grounded answers with source-page citations.

Live demo: [pdfaiassist.streamlit.app](https://pdfaiassist.streamlit.app/)

## Features

- Streamlit browser interface for PDF upload and chat
- PDF text extraction with page metadata preserved for citations
- Overlapping text chunking
- Embedding-based similarity search
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

The browser interface keeps the current document and messages in memory for that browser session. Restarting the app clears the session.

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

Docker support is available through `Dockerfile` and `docker-compose.yml`.

## Evaluation

Create a JSON evaluation file using [data/evaluation.example.json](data/evaluation.example.json), then run:

```powershell
python -m ai_document_assistant `
  --database data\assistant.db `
  --evaluate data\evaluation.json
```

The report measures retrieval hits, answer presence, and citation presence. These metrics are useful signals but do not replace human review of factual accuracy.

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
```

See [docs/openrouter-setup.md](docs/openrouter-setup.md) for provider setup and [docs/deployment.md](docs/deployment.md) for Streamlit Community Cloud deployment.

## Limitations

- Scanned or image-only PDFs require OCR, which is not included.
- Streamlit sessions use in-memory indexes and are not permanent.
- SQLite and linear vector search are intended for small deployments, not large-scale production traffic.
- There is no authentication, per-user document isolation, malware scanning, or rate limiting.
- Broad requests such as “list every item in the document” may require a document-wide retrieval strategy rather than a small set of top-matching chunks.

## Repository layout

```text
src/ai_document_assistant/   application code
tests/                       automated tests
data/evaluation.example.json evaluation file template
docs/                        architecture, setup, and deployment documentation
```
