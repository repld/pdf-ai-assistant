"""FastAPI interface for uploading PDFs and asking grounded questions."""

from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from openai import OpenAIError
from pydantic import BaseModel, Field

from .application import build_rag_assistant, index_uploaded_pdf
from .documents import DocumentError
from .limits import MAX_UPLOAD_BYTES, UPLOAD_LIMIT_MB
from .rag import RAGAssistant

DEFAULT_SESSION_ID = "default"
MAX_SESSIONS = 100
MAX_REQUESTS_PER_MINUTE = 60
_SESSION_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
logger = logging.getLogger(__name__)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)


class _RateLimiter:
    def __init__(self, limit: int = MAX_REQUESTS_PER_MINUTE) -> None:
        self._limit = limit
        self._requests: dict[str, list[float]] = {}

    def allow(self, session_id: str) -> bool:
        now = time.monotonic()
        recent_requests = [
            started_at
            for started_at in self._requests.get(session_id, [])
            if now - started_at < 60
        ]
        if len(recent_requests) >= self._limit:
            self._requests[session_id] = recent_requests
            return False
        recent_requests.append(now)
        self._requests[session_id] = recent_requests
        return True


def create_app(assistant: Any | None = None, *, auth_token: str | None = None) -> FastAPI:
    """Create the web application, optionally with an injected assistant."""

    app = FastAPI(title="AI Document Assistant", version="0.1.0")
    configured_token = auth_token if auth_token is not None else os.getenv("API_AUTH_TOKEN")
    assistants: dict[str, Any] = {}
    if assistant is not None:
        assistants[DEFAULT_SESSION_ID] = assistant
    rate_limiter = _RateLimiter()

    def validate_session_id(session_id: str) -> str:
        if not _SESSION_ID_PATTERN.fullmatch(session_id):
            raise HTTPException(
                status_code=400,
                detail="X-Session-ID must contain 1-64 letters, numbers, dots, underscores, or hyphens",
            )
        return session_id

    def authorize(authorization: str | None) -> None:
        if configured_token is None:
            return
        if authorization != f"Bearer {configured_token}":
            raise HTTPException(
                status_code=401,
                detail="authentication required",
                headers={"WWW-Authenticate": "Bearer"},
            )

    def check_rate_limit(session_id: str) -> None:
        if not rate_limiter.allow(session_id):
            raise HTTPException(status_code=429, detail="request rate limit exceeded")

    def get_assistant(session_id: str) -> RAGAssistant:
        session_id = validate_session_id(session_id)
        if session_id not in assistants:
            if len(assistants) >= MAX_SESSIONS:
                raise HTTPException(status_code=429, detail="too many active sessions")
            database_path = Path(os.getenv("DATABASE_PATH", "data/assistant.db"))
            if session_id != DEFAULT_SESSION_ID:
                database_path = database_path.with_name(
                    f"{database_path.stem}-{session_id}{database_path.suffix}"
                )
            assistants[session_id] = build_rag_assistant(database_path=database_path)
        return assistants[session_id]

    @app.get("/health")
    def health(authorization: str | None = Header(default=None)) -> dict[str, str]:
        authorize(authorization)
        return {"status": "ok"}

    @app.post("/ask")
    def ask(
        request: AskRequest,
        session_id: str = Header(default=DEFAULT_SESSION_ID, alias="X-Session-ID"),
        authorization: str | None = Header(default=None),
    ) -> dict[str, object]:
        authorize(authorization)
        session_id = validate_session_id(session_id)
        check_rate_limit(session_id)
        question = request.question.strip()
        if not question:
            raise HTTPException(status_code=400, detail="question must not be empty")

        try:
            result = get_assistant(session_id).answer(question)
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
        session_id: str = Header(default=DEFAULT_SESSION_ID, alias="X-Session-ID"),
        authorization: str | None = Header(default=None),
    ) -> dict[str, int | str]:
        authorize(authorization)
        session_id = validate_session_id(session_id)
        check_rate_limit(session_id)
        if file.content_type not in {"application/pdf", None}:
            raise HTTPException(status_code=415, detail="only PDF files are supported")

        content = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"PDF exceeds the {UPLOAD_LIMIT_MB} MB limit",
            )

        source_name = Path(file.filename or "document.pdf").name
        suffix = Path(source_name).suffix.lower()
        if suffix != ".pdf":
            raise HTTPException(status_code=415, detail="only PDF files are supported")

        try:
            summary = index_uploaded_pdf(
                get_assistant(session_id),
                content,
                source_name=source_name,
            )
        except (DocumentError, RuntimeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

        return {
            "session_id": session_id,
            "source": source_name,
            "page_count": summary.page_count,
            "chunk_count": summary.chunk_count,
        }

    return app


app = create_app()
