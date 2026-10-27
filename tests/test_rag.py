from dataclasses import dataclass

from ai_document_assistant.documents import DocumentChunk
from ai_document_assistant.rag import UNSUPPORTED_ANSWER, RAGAssistant


def chunk(chunk_id: str, text: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        source="guide.pdf",
        page_number=2,
        text=text,
    )


class FakeEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            lowered = text.lower()
            if "python" in lowered:
                vectors.append([1.0, 0.0])
            elif "unknown" in lowered:
                vectors.append([0.0, 0.0])
            else:
                vectors.append([0.0, 1.0])
        return vectors


@dataclass
class FakeAnswerer:
    questions: list[str]
    contexts: list[list[DocumentChunk]]

    def answer(self, question: str, context: list[DocumentChunk]) -> str:
        self.questions.append(question)
        self.contexts.append(context)
        return "Python is a programming language. [1]"


def test_rag_assistant_retrieves_context_and_returns_citations() -> None:
    answerer = FakeAnswerer([], [])
    assistant = RAGAssistant(
        embedder=FakeEmbedder(),
        answerer=answerer,
        min_score=0.5,
    )
    assistant.index([chunk("python", "Python is a programming language.")])

    result = assistant.answer("What is Python?")

    assert result.answer == "Python is a programming language. [1]"
    assert result.citations[0].chunk_id == "python"
    assert answerer.questions == ["What is Python?"]
    assert [item.chunk_id for item in answerer.contexts[0]] == ["python"]


def test_rag_assistant_declines_when_no_relevant_context_exists() -> None:
    answerer = FakeAnswerer([], [])
    assistant = RAGAssistant(
        embedder=FakeEmbedder(),
        answerer=answerer,
        min_score=0.5,
    )
    assistant.index([chunk("python", "Python is a programming language.")])

    result = assistant.answer("Tell me something unknown.")

    assert result.answer == "I could not find an answer in the uploaded documents."
    assert result.citations == ()
    assert answerer.questions == []


@dataclass
class CitingAnswerer:
    answer_text: str

    def answer(self, question: str, context: list[DocumentChunk]) -> str:
        return self.answer_text


def test_rag_assistant_returns_only_model_cited_sources() -> None:
    assistant = RAGAssistant(
        embedder=FakeEmbedder(),
        answerer=CitingAnswerer("The answer is in the second excerpt. [2]"),
        min_score=0.5,
    )
    assistant.index(
        [
            chunk("first", "Python is a programming language."),
            chunk("second", "Python was created by Guido van Rossum."),
        ]
    )

    result = assistant.answer("Who created Python?")

    assert [citation.chunk_id for citation in result.citations] == ["second"]


def test_rag_assistant_rejects_an_answer_without_valid_citations() -> None:
    assistant = RAGAssistant(
        embedder=FakeEmbedder(),
        answerer=CitingAnswerer("This answer is not supported by a citation."),
        min_score=0.5,
    )
    assistant.index([chunk("python", "Python is a programming language.")])

    result = assistant.answer("What is Python?")

    assert result == type(result)(answer=UNSUPPORTED_ANSWER, citations=())
