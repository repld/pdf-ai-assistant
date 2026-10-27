"""PDF extraction and citation-preserving text chunking."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pypdf import PdfReader


class DocumentError(RuntimeError):
    """Raised when a document cannot provide usable text."""


@dataclass(frozen=True)
class DocumentPage:
    """Extracted text from one page of a source document."""

    source: str
    page_number: int
    text: str


@dataclass(frozen=True)
class DocumentChunk:
    """A retrieval unit that retains enough metadata to cite its source."""

    chunk_id: str
    source: str
    page_number: int
    text: str


def extract_pdf_pages(
    path: Path,
    *,
    max_pages: int | None = None,
    page_start: int = 1,
    page_end: int | None = None,
    source_name: str | None = None,
    reader_factory: Callable[[str], Any] = PdfReader,
) -> list[DocumentPage]:
    """Extract non-empty pages from a PDF while preserving page numbers."""

    if max_pages is not None and max_pages <= 0:
        raise ValueError("max_pages must be positive")
    if page_start <= 0:
        raise ValueError("page_start must be positive")
    if page_end is not None and page_end < page_start:
        raise ValueError("page_end must not be smaller than page_start")

    reader = reader_factory(str(path))
    display_source = Path(source_name).name if source_name else path.name
    pages: list[DocumentPage] = []

    for page_number, page in enumerate(reader.pages, start=1):
        if page_number < page_start:
            continue
        if page_end is not None and page_number > page_end:
            break
        if max_pages is not None and page_number > page_start + max_pages - 1:
            break
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(
                DocumentPage(
                    source=display_source,
                    page_number=page_number,
                    text=text,
                )
            )

    if not pages:
        raise DocumentError(f"{path.name} contains no extractable text")
    return pages


def chunk_pages(
    pages: Sequence[DocumentPage],
    *,
    max_chars: int = 1200,
    overlap: int = 200,
) -> list[DocumentChunk]:
    """Split pages into overlapping windows without losing source metadata."""

    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if overlap < 0 or overlap >= max_chars:
        raise ValueError("overlap must be smaller than max_chars and non-negative")

    chunks: list[DocumentChunk] = []
    for page in pages:
        text = page.text.strip()
        start = 0
        chunk_number = 1

        while start < len(text):
            proposed_end = min(start + max_chars, len(text))
            end = proposed_end

            if proposed_end < len(text):
                whitespace = text.rfind(" ", start, proposed_end)
                if whitespace > start:
                    end = whitespace

            chunk_text = text[start:end].strip()
            if not chunk_text:
                end = proposed_end
                chunk_text = text[start:end].strip()

            chunks.append(
                DocumentChunk(
                    chunk_id=f"{page.source}:p{page.page_number}:c{chunk_number}",
                    source=page.source,
                    page_number=page.page_number,
                    text=chunk_text,
                )
            )

            if end >= len(text):
                break

            start = max(start + 1, end - overlap)
            chunk_number += 1

    return chunks
