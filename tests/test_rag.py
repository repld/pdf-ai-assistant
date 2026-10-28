from dataclasses import dataclass

from ai_document_assistant.documents import DocumentChunk
from ai_document_assistant.observability import InMemoryTelemetry
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
    assert result.citations[0].text == "Python is a programming language."
    assert result.citations[0].score is not None
    assert result.citations[0].score >= 0.5
    assert result.retrieved_chunk_ids == ("python",)
    assert result.diagnostics is not None
    assert result.diagnostics.total_ms >= 0
    assert answerer.questions == ["What is Python?"]
    assert [item.chunk_id for item in answerer.contexts[0]] == ["python"]


def test_rag_assistant_retrieves_more_context_for_exhaustive_list_questions() -> None:
    answerer = FakeAnswerer([], [])
    assistant = RAGAssistant(
        embedder=FakeEmbedder(),
        answerer=answerer,
        min_score=0.5,
    )
    assistant.index(
        [chunk(f"item-{index}", f"Python feature {index}") for index in range(25)]
    )

    result = assistant.answer("List every Python feature.")

    assert len(answerer.contexts[0]) == 20
    assert len(result.retrieved_chunk_ids) == 20


def test_rag_assistant_keeps_normal_context_limit_for_focused_questions() -> None:
    answerer = FakeAnswerer([], [])
    assistant = RAGAssistant(
        embedder=FakeEmbedder(),
        answerer=answerer,
        min_score=0.5,
    )
    assistant.index(
        [chunk(f"item-{index}", f"Python feature {index}") for index in range(25)]
    )

    assistant.answer("What is Python?")

    assert len(answerer.contexts[0]) == 5


def test_rag_assistant_records_request_metrics() -> None:
    telemetry = InMemoryTelemetry()
    assistant = RAGAssistant(
        embedder=FakeEmbedder(),
        answerer=FakeAnswerer([], []),
        telemetry=telemetry,
        min_score=0.5,
    )
    assistant.index([chunk("python", "Python is a programming language.")])

    assistant.answer("What is Python?")

    observation = telemetry.observations[-1]
    assert observation.event == "rag.answer"
    assert observation.attributes["status"] == "success"
    assert observation.attributes["retrieved_count"] == 1
    assert observation.attributes["citation_count"] == 1


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
    assert result.retrieved_chunk_ids == ()
    assert answerer.questions == []


@dataclass
class CitingAnswerer:
    answer_text: str

    def answer(self, question: str, context: list[DocumentChunk]) -> str:
        return self.answer_text


def test_rag_assistant_returns_only_model_cited_sources() -> None:
    assistant = RAGAssistant(
        embedder=FakeEmbedder(),
        answerer=CitingAnswerer("The answer is in the cited excerpt. [1]"),
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
    assert result.retrieved_chunk_ids == ("second", "first")


def test_rag_assistant_rejects_an_answer_without_valid_citations() -> None:
    assistant = RAGAssistant(
        embedder=FakeEmbedder(),
        answerer=CitingAnswerer("This answer is not supported by a citation."),
        min_score=0.5,
    )
    assistant.index([chunk("python", "Python is a programming language.")])

    result = assistant.answer("What is Python?")

    assert result.answer == UNSUPPORTED_ANSWER
    assert result.citations == ()
    assert result.retrieved_chunk_ids == ("python",)
    assert result.diagnostics is not None
