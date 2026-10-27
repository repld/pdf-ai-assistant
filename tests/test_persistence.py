from ai_document_assistant.documents import DocumentChunk
from ai_document_assistant.persistence import SQLiteVectorStore


def test_sqlite_vector_store_survives_reopening(tmp_path) -> None:
    database_path = tmp_path / "assistant.db"
    chunk = DocumentChunk(
        chunk_id="guide.pdf:p1:c1",
        source="guide.pdf",
        page_number=1,
        text="Python is a programming language.",
    )

    first_store = SQLiteVectorStore(database_path)
    first_store.add([chunk], [[1.0, 0.0]])
    first_store.close()

    reopened_store = SQLiteVectorStore(database_path)
    results = reopened_store.search([1.0, 0.0])
    reopened_store.close()

    assert results[0].chunk == chunk
    assert results[0].score == 1.0
