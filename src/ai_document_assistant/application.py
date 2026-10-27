"""Application assembly for document indexing and grounded answers."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI

from .assistant import load_settings
from .documents import DocumentChunk, DocumentPage, chunk_pages, extract_pdf_pages
from .persistence import SQLiteVectorStore
from .providers import OpenAIAnswerer, OpenAIEmbedder
from .rag import RAGAnswer, RAGAssistant

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IndexingSummary:
    """Counts produced while indexing a collection of PDFs."""

    page_count: int
    chunk_count: int


def build_rag_assistant(*, database_path: Path | None = None) -> RAGAssistant:
    """Create the production adapters and an in-memory RAG assistant."""

    settings = load_settings()
    client = OpenAI(api_key=settings.api_key)
    store = None
    if database_path is not None:
        database_path.parent.mkdir(parents=True, exist_ok=True)
        store = SQLiteVectorStore(database_path)
    return RAGAssistant(
        embedder=OpenAIEmbedder(client=client, model=settings.embedding_model),
        answerer=OpenAIAnswerer(client=client, model=settings.model),
        store=store,
    )


def index_pdfs(
    assistant: RAGAssistant,
    paths: Sequence[Path],
    *,
    max_chars: int = 1200,
    overlap: int = 200,
) -> IndexingSummary:
    """Extract, chunk, and index all supplied PDF files."""

    pages: list[DocumentPage] = []
    for path in paths:
        logger.info("Extracting PDF pages from %s", path.name)
        pages.extend(extract_pdf_pages(path))

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
