from fastapi.testclient import TestClient

from ai_document_assistant import web
from ai_document_assistant.application import IndexingSummary
from ai_document_assistant.rag import Citation, RAGAnswer
from ai_document_assistant.web import create_app


class FakeAssistant:
    def answer(self, question: str) -> RAGAnswer:
        return RAGAnswer(
            answer=f"Answer for: {question}",
            citations=(Citation("guide.pdf:p1:c1", "guide.pdf", 1),),
        )


class FailingAssistant:
    def answer(self, question: str) -> RAGAnswer:
        raise RuntimeError("provider detail should not be exposed")


def test_health_endpoint() -> None:
    client = TestClient(create_app(FakeAssistant()))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_endpoint_returns_answer_and_citations() -> None:
    client = TestClient(create_app(FakeAssistant()))

    response = client.post("/ask", json={"question": "What is Python?"})

    assert response.status_code == 200
    assert response.json() == {
        "answer": "Answer for: What is Python?",
        "citations": [
            {"chunk_id": "guide.pdf:p1:c1", "source": "guide.pdf", "page_number": 1}
        ],
    }


def test_ask_endpoint_rejects_blank_question() -> None:
    client = TestClient(create_app(FakeAssistant()))

    response = client.post("/ask", json={"question": "   "})

    assert response.status_code == 400
    assert response.json() == {"detail": "question must not be empty"}


def test_ask_endpoint_rejects_an_oversized_question() -> None:
    client = TestClient(create_app(FakeAssistant()))

    response = client.post("/ask", json={"question": "x" * 4001})

    assert response.status_code == 422


def test_ask_endpoint_hides_provider_errors() -> None:
    client = TestClient(create_app(FailingAssistant()))

    response = client.post("/ask", json={"question": "What is Python?"})

    assert response.status_code == 502
    assert response.json() == {"detail": "AI provider request failed"}


def test_upload_endpoint_preserves_original_filename(monkeypatch) -> None:
    source_names: list[str] = []

    def fake_index_uploaded_pdf(
        assistant: object,
        content: bytes,
        *,
        source_name: str,
    ) -> IndexingSummary:
        source_names.append(source_name)
        return IndexingSummary(page_count=1, chunk_count=1)

    monkeypatch.setattr(web, "index_uploaded_pdf", fake_index_uploaded_pdf)
    client = TestClient(create_app(FakeAssistant()))

    response = client.post(
        "/documents",
        files={"file": ("original.pdf", b"not a real PDF", "application/pdf")},
    )

    assert response.status_code == 200
    assert response.json() == {
        "session_id": "default",
        "source": "original.pdf",
        "page_count": 1,
        "chunk_count": 1,
    }
    assert source_names == ["original.pdf"]


def test_api_authentication_is_enforced_when_configured() -> None:
    client = TestClient(create_app(FakeAssistant(), auth_token="secret"))

    assert client.get("/health").status_code == 401
    assert client.get(
        "/health",
        headers={"Authorization": "Bearer secret"},
    ).status_code == 200


def test_session_id_is_validated() -> None:
    client = TestClient(create_app(FakeAssistant()))

    response = client.post(
        "/ask",
        headers={"X-Session-ID": "not a valid session"},
        json={"question": "What is Python?"},
    )

    assert response.status_code == 400
