# AI Document Assistant

An educational project that will grow into a document question-answering application. Users will eventually upload PDFs, ask questions, and receive answers grounded in the documents with source citations.

This repository is separate from the existing breast-cancer data project. It has its own Python package, tests, environment file, and Git history.

## Milestone 1: one AI question

The first milestone proves the smallest useful application loop:

```text
terminal question → ask_question() → OpenAI Responses API → terminal answer
```

The test uses a fake client, so it does not contact the API or incur usage costs.

## Setup

Requirements: Python 3.12+.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

Edit `.env` and set `OPENAI_API_KEY` to your API key. Keep `.env` private; it is ignored by Git. You can change the model with `OPENAI_MODEL`.

## Run

Run the network-free test:

```powershell
python -m pytest
```

After configuring an API key, make one real request:

```powershell
python -m ai_document_assistant "Explain what an API is in one paragraph."
```

Real requests use API credits. This project does not make a live request automatically during tests.

## Learning plan

See [docs/learning-plan.md](docs/learning-plan.md) for the high-level architecture, first-week plan, and book topics to study before the next feature.
