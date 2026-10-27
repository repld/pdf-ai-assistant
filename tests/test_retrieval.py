from types import SimpleNamespace

import pytest

from ai_document_assistant.documents import DocumentChunk
from ai_document_assistant.providers import ChatCompletionAnswerer, OpenAIAnswerer, OpenAIEmbedder
from ai_document_assistant.retrieval import InMemoryVectorStore


def chunk(chunk_id: str, text: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        source="guide.pdf",
        page_number=1,
        text=text,
    )


class FakeEmbeddings:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(
            data=[
                SimpleNamespace(index=1, embedding=[0.0, 1.0]),
                SimpleNamespace(index=0, embedding=[1.0, 0.0]),
            ]
        )


class FakeEmbeddingClient:
    def __init__(self) -> None:
        self.embeddings = FakeEmbeddings()


class FakeResponses:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(output_text="The answer is grounded. [1]")


class FakeAnswerClient:
    def __init__(self) -> None:
        self.responses = FakeResponses()


class FakeChatCompletions:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Chat answer [1]"))]
        )


class FakeChatClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=FakeChatCompletions())


def test_openai_embedder_batches_text_and_restores_input_order() -> None:
    client = FakeEmbeddingClient()
    embedder = OpenAIEmbedder(client=client, model="test-embedding-model")

    vectors = embedder.embed(["first", "second"])

    assert vectors == [[1.0, 0.0], [0.0, 1.0]]
    assert client.embeddings.calls == [
        {
            "model": "test-embedding-model",
            "input": ["first", "second"],
            "encoding_format": "float",
        }
    ]


def test_openai_answerer_sends_numbered_context_and_returns_text() -> None:
    client = FakeAnswerClient()
    answerer = OpenAIAnswerer(client=client, model="test-model")

    answer = answerer.answer("What is Python?", [chunk("python", "Python is a language.")])

    assert answer == "The answer is grounded. [1]"
    call = client.responses.calls[0]
    assert call["model"] == "test-model"
    assert "[1] guide.pdf, page 1" in str(call["input"])


def test_chat_completion_answerer_returns_openrouter_style_answer() -> None:
    client = FakeChatClient()
    answerer = ChatCompletionAnswerer(client=client, model="openrouter/free")

    answer = answerer.answer("What is Python?", [chunk("python", "Python is a language.")])

    assert answer == "Chat answer [1]"
    call = client.chat.completions.calls[0]
    assert call["model"] == "openrouter/free"
    assert call["messages"][0]["role"] == "system"
    assert "[1] guide.pdf, page 1" in call["messages"][1]["content"]


def test_in_memory_store_returns_most_similar_chunks_first() -> None:
    store = InMemoryVectorStore()
    chunks = [chunk("python", "Python"), chunk("cats", "Cats"), chunk("mixed", "Both")]
    store.add(chunks, [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

    results = store.search([1.0, 0.0], limit=2)

    assert [result.chunk.chunk_id for result in results] == ["python", "mixed"]
    assert results[0].score == pytest.approx(1.0)
    assert results[1].score == pytest.approx(2**-0.5)


def test_in_memory_store_rejects_mismatched_vectors() -> None:
    store = InMemoryVectorStore()

    with pytest.raises(ValueError, match="same number"):
        store.add([chunk("one", "One")], [[1.0, 0.0], [0.0, 1.0]])


def test_in_memory_store_returns_no_results_when_empty() -> None:
    assert InMemoryVectorStore().search([1.0, 0.0]) == []
