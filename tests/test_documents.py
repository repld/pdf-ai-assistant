from pathlib import Path

import pytest

from ai_document_assistant.documents import (
    DocumentError,
    DocumentPage,
    chunk_pages,
    extract_pdf_pages,
)


class FakePage:
    def __init__(self, text: str | None) -> None:
        self.text = text

    def extract_text(self) -> str | None:
        return self.text


class FakeReader:
    def __init__(self, _path: str) -> None:
        self.pages = [FakePage(" First page "), FakePage(None), FakePage("Second page")]


def test_extract_pdf_pages_preserves_source_and_page_numbers() -> None:
    pages = extract_pdf_pages(
        Path("guide.pdf"),
        reader_factory=FakeReader,
    )

    assert pages == [
        DocumentPage(source="guide.pdf", page_number=1, text="First page"),
        DocumentPage(source="guide.pdf", page_number=3, text="Second page"),
    ]


def test_extract_pdf_pages_rejects_pdf_without_extractable_text() -> None:
    with pytest.raises(DocumentError, match="no extractable text"):
        extract_pdf_pages(
            Path("scanned.pdf"),
            reader_factory=lambda _path: type("Reader", (), {"pages": [FakePage(None)]})(),
        )


def test_extract_pdf_pages_can_limit_pages_for_a_small_smoke_test() -> None:
    pages = extract_pdf_pages(
        Path("guide.pdf"),
        max_pages=1,
        reader_factory=FakeReader,
    )

    assert pages == [
        DocumentPage(source="guide.pdf", page_number=1, text="First page"),
    ]


def test_extract_pdf_pages_can_select_a_page_range() -> None:
    pages = extract_pdf_pages(
        Path("guide.pdf"),
        page_start=3,
        page_end=3,
        reader_factory=FakeReader,
    )

    assert pages == [
        DocumentPage(source="guide.pdf", page_number=3, text="Second page"),
    ]


def test_chunk_pages_keeps_page_metadata_and_overlapping_windows() -> None:
    pages = [DocumentPage(source="guide.pdf", page_number=4, text="ABCDEFGHIJKL")]

    chunks = chunk_pages(pages, max_chars=6, overlap=2)

    assert [chunk.text for chunk in chunks] == ["ABCDEF", "EFGHIJ", "IJKL"]
    assert all(chunk.source == "guide.pdf" for chunk in chunks)
    assert all(chunk.page_number == 4 for chunk in chunks)
    assert [chunk.chunk_id for chunk in chunks] == ["guide.pdf:p4:c1", "guide.pdf:p4:c2", "guide.pdf:p4:c3"]


def test_chunk_pages_rejects_invalid_window_settings() -> None:
    pages = [DocumentPage(source="guide.pdf", page_number=1, text="text")]

    with pytest.raises(ValueError, match="max_chars must be positive"):
        chunk_pages(pages, max_chars=0, overlap=0)

    with pytest.raises(ValueError, match="overlap must be smaller"):
        chunk_pages(pages, max_chars=10, overlap=10)
