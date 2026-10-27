"""Adapters for model-provider operations used by retrieval."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .documents import DocumentChunk

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


class OpenAIEmbedder:
    """Adapt the OpenAI embeddings endpoint to the application's embedder interface."""

    def __init__(self, *, client: Any, model: str = DEFAULT_EMBEDDING_MODEL) -> None:
        self._client = client
        self._model = model

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one vector per input text, in the same order as the inputs."""

        if not texts:
            return []
        if any(not text.strip() for text in texts):
            raise ValueError("embedding input must not contain empty text")

        response = self._client.embeddings.create(
            model=self._model,
            input=list(texts),
            encoding_format="float",
        )
        data = sorted(response.data, key=lambda item: item.index)
        if len(data) != len(texts):
            raise RuntimeError("embedding provider returned the wrong number of vectors")
        return [list(item.embedding) for item in data]


class OpenAIAnswerer:
    """Adapt the Responses API to grounded answering over retrieved chunks."""

    def __init__(self, *, client: Any, model: str) -> None:
        self._client = client
        self._model = model

    def answer(self, question: str, context: Sequence[DocumentChunk]) -> str:
        if not context:
            raise ValueError("answer context must not be empty")

        source_text = "\n\n".join(
            f"[{index}] {chunk.source}, page {chunk.page_number}\n{chunk.text}"
            for index, chunk in enumerate(context, start=1)
        )
        response = self._client.responses.create(
            model=self._model,
            instructions=(
                "Answer the question using only the supplied document excerpts. "
                "If the excerpts do not support an answer, say that the answer was "
                "not found in the uploaded documents. Cite supporting excerpts with "
                "their bracketed source number, such as [1]."
            ),
            input=f"Question: {question}\n\nDocument excerpts:\n{source_text}",
        )
        answer = response.output_text.strip()
        if not answer:
            raise RuntimeError("the model returned an empty grounded answer")
        return answer
