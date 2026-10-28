from unittest.mock import patch

from ai_document_assistant import ui


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
