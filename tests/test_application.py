from ai_document_assistant.application import format_grounded_answer
from ai_document_assistant.rag import Citation, RAGAnswer


def test_format_grounded_answer_includes_source_pages() -> None:
    result = RAGAnswer(
        answer="Python is a language. [1]",
        citations=(
            Citation(chunk_id="python", source="guide.pdf", page_number=3),
        ),
    )

    assert format_grounded_answer(result) == (
        "Python is a language. [1]\n\nSources:\n- guide.pdf, page 3"
    )
