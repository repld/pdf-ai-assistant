from fastapi.testclient import TestClient

from ai_document_assistant.rag import Citation, RAGAnswer
from ai_document_assistant.web import create_app


class FakeAssistant:
    def answer(self, question: str) -> RAGAnswer:
        return RAGAnswer(
            answer=f"Answer for: {question}",
            citations=(Citation("guide.pdf:p1:c1", "guide.pdf", 1),),
        )


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
