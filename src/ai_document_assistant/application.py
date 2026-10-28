"""Application assembly for document indexing and grounded answers."""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI

from .assistant import Settings, load_settings
from .documents import DocumentChunk, DocumentPage, chunk_pages, extract_pdf_pages
from .persistence import SQLiteVectorStore
from .providers import ChatCompletionAnswerer, OpenAIAnswerer, OpenAIEmbedder
from .rag import RAGAnswer, RAGAssistant

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IndexingSummary:
    """Counts produced while indexing a collection of PDFs."""

    page_count: int
    chunk_count: int


def build_rag_assistant(*, database_path: Path | None = None) -> RAGAssistant:
    """Create the configured adapters and vector store."""

    settings = load_settings()
    client = _build_client(settings)
    return RAGAssistant(
        embedder=OpenAIEmbedder(client=client, model=settings.embedding_model),
        answerer=_build_answerer(settings, client),
        store=_build_store(database_path),
        model_name=settings.model,
    )


def _build_client(settings: Settings) -> OpenAI:
    client_kwargs = {"api_key": settings.api_key}
    if settings.base_url:
        client_kwargs["base_url"] = settings.base_url
    return OpenAI(**client_kwargs, timeout=30.0, max_retries=2)


def _build_answerer(settings: Settings, client: OpenAI) -> OpenAIAnswerer | ChatCompletionAnswerer:
    if settings.provider == "openrouter":
        return ChatCompletionAnswerer(client=client, model=settings.model)
    return OpenAIAnswerer(client=client, model=settings.model)


def _build_store(database_path: Path | None) -> SQLiteVectorStore | None:
    if database_path is None:
        return None
    database_path.parent.mkdir(parents=True, exist_ok=True)
    return SQLiteVectorStore(database_path)


def index_uploaded_pdf(
    assistant: RAGAssistant,
    content: bytes,
    *,
    source_name: str,
    max_chars: int = 1600,
    overlap: int = 200,
    progress: Callable[[str], None] | None = None,
) -> IndexingSummary:
    """Extract, chunk, and index one uploaded PDF from memory."""

    temporary_path: Path | None = None
    try:
        if progress:
            progress("Saving the upload and extracting PDF text...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temporary_file:
            temporary_file.write(content)
            temporary_path = Path(temporary_file.name)

        if progress:
            progress("Splitting pages into citation-preserving chunks...")
        pages = extract_pdf_pages(temporary_path, source_name=source_name)
        chunks = chunk_pages(pages, max_chars=max_chars, overlap=overlap)
        if progress:
            progress(f"Embedding and indexing {len(chunks)} chunks...")
        assistant.index(chunks, progress=progress)
        logger.info("Indexed uploaded document %s", source_name)
        return IndexingSummary(page_count=len(pages), chunk_count=len(chunks))
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def index_pdfs(
    assistant: RAGAssistant,
    paths: Sequence[Path],
    *,
    max_chars: int = 1200,
    overlap: int = 200,
    max_pages: int | None = None,
    page_start: int = 1,
    page_end: int | None = None,
) -> IndexingSummary:
    """Extract, chunk, and index all supplied PDF files."""

    pages: list[DocumentPage] = []
    for path in paths:
        logger.info("Extracting PDF pages from %s", path.name)
        pages.extend(
            extract_pdf_pages(
                path,
                max_pages=max_pages,
                page_start=page_start,
                page_end=page_end,
            )
        )

    chunks: list[DocumentChunk] = chunk_pages(
        pages,
        max_chars=max_chars,
        overlap=overlap,
    )
    assistant.index(chunks)
    logger.info("Indexed %s pages into %s chunks", len(pages), len(chunks))
    return IndexingSummary(page_count=len(pages), chunk_count=len(chunks))


def format_grounded_answer(result: RAGAnswer) -> str:
    """Format an answer and its citation metadata for terminal output."""

    if not result.citations:
        return result.answer

    sources = "\n".join(
        f"- {citation.source}, page {citation.page_number}"
        for citation in result.citations
    )
    return f"{result.answer}\n\nSources:\n{sources}"
