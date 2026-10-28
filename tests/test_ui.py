from unittest.mock import patch

import pytest

from ai_document_assistant import ui
from ai_document_assistant.application import IndexingSummary
from ai_document_assistant.documents import DocumentError


def test_citation_navigation_updates_page_and_excerpt_state() -> None:
    state: dict[str, object] = {}
    citation = {
        "source": "book.pdf",
        "page_number": 24,
        "text": "Supporting passage.",
        "score": 0.9,
    }

    with patch.object(ui.st, "session_state", state):
        ui._select_citation(citation)

    assert state["selected_page"] == 24
    assert state["selected_citation"] == citation


def test_manual_page_navigation_clears_previous_excerpt() -> None:
    state: dict[str, object] = {
        "selected_page": 24,
        "selected_citation": {"page_number": 24, "text": "Old excerpt"},
    }

    with patch.object(ui.st, "session_state", state):
        ui._select_page(31)

    assert state["selected_page"] == 31
    assert "selected_citation" not in state


def test_upload_size_limit_accepts_a_68_mb_file_without_indexing_copy() -> None:
    class UploadedFile:
        name = "large-book.pdf"
        size = 68 * 1024 * 1024

        def getvalue(self) -> bytes:
            return b"pdf-content"

    state: dict[str, object] = {}
    assistant = object()

    with (
        patch.object(ui.st, "session_state", state),
        patch.object(ui, "build_rag_assistant", return_value=assistant),
        patch.object(
            ui,
            "index_uploaded_pdf",
            return_value=IndexingSummary(page_count=1, chunk_count=1),
        ) as index_pdf,
    ):
        source = ui._index_upload(UploadedFile())

    assert source == "large-book.pdf"
    assert index_pdf.call_args.args[1] == b"pdf-content"
    assert ui.MAX_UPLOAD_BYTES == 100 * 1024 * 1024


def test_upload_size_limit_rejects_before_reading_an_oversized_file() -> None:
    class OversizedFile:
        name = "too-large.pdf"
        size = 101 * 1024 * 1024

        def getvalue(self) -> bytes:
            raise AssertionError("the oversized upload should be rejected first")

    with pytest.raises(DocumentError, match="100 MB limit"):
        ui._index_upload(OversizedFile())
