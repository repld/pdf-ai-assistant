from fastapi.testclient import TestClient

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
