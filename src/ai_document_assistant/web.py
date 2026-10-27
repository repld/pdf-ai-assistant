"""FastAPI interface for uploading PDFs and asking grounded questions."""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from openai import OpenAIError
from pydantic import BaseModel, Field

from .application import build_rag_assistant
from .documents import DocumentError, chunk_pages, extract_pdf_pages
from .rag import RAGAssistant

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
logger = logging.getLogger(__name__)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)


def create_app(assistant: Any | None = None) -> FastAPI:
    """Create the web application, optionally with an injected assistant."""

    app = FastAPI(title="AI Document Assistant", version="0.1.0")
    state: dict[str, Any | None] = {"assistant": assistant}

    def get_assistant() -> RAGAssistant:
        if state["assistant"] is None:
            database_path = Path(os.getenv("DATABASE_PATH", "data/assistant.db"))
            state["assistant"] = build_rag_assistant(database_path=database_path)
        return state["assistant"]

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/ask")
    def ask(request: AskRequest) -> dict[str, object]:
        question = request.question.strip()
        if not question:
            raise HTTPException(status_code=400, detail="question must not be empty")

        try:
            result = get_assistant().answer(question)
        except (OpenAIError, RuntimeError) as error:
            logger.exception("The assistant failed to answer a question")
            raise HTTPException(status_code=502, detail="AI provider request failed") from error
        return {
            "answer": result.answer,
            "citations": [
                {
                    "chunk_id": citation.chunk_id,
                    "source": citation.source,
                    "page_number": citation.page_number,
                }
                for citation in result.citations
            ],
        }

    @app.post("/documents")
    async def upload_document(
        file: Annotated[UploadFile, File(description="A text-based PDF document")],
    ) -> dict[str, int | str]:
        if file.content_type not in {"application/pdf", None}:
            raise HTTPException(status_code=415, detail="only PDF files are supported")

        content = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="PDF exceeds the 20 MB limit")

        suffix = Path(file.filename or "document.pdf").suffix.lower()
        if suffix != ".pdf":
            raise HTTPException(status_code=415, detail="only PDF files are supported")

        temporary_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temporary_file:
                temporary_file.write(content)
                temporary_path = temporary_file.name

            pages = extract_pdf_pages(Path(temporary_path))
            chunks = chunk_pages(pages)
            get_assistant().index(chunks)
            logger.info("Indexed uploaded document %s", file.filename or "document.pdf")
        except (DocumentError, RuntimeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        finally:
            if temporary_path is not None:
                Path(temporary_path).unlink(missing_ok=True)

        return {
            "source": file.filename or "document.pdf",
            "page_count": len(pages),
            "chunk_count": len(chunks),
        }

    return app


app = create_app()
