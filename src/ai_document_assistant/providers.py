"""Adapters for model-provider operations used by retrieval."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .documents import DocumentChunk

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_EMBEDDING_BATCH_SIZE = 64
GROUNDING_INSTRUCTIONS = (
    "Answer the question using only the supplied document excerpts. "
    "If the excerpts do not support an answer, say exactly: "
    "'I could not find an answer in the uploaded documents.' "
    "Every factual answer must cite one or more supporting excerpts with their "
    "bracketed source number, such as [1]. Do not invent citations or discuss "
    "information that is not in the excerpts."
)


def _source_text(context: Sequence[DocumentChunk]) -> str:
    return "\n\n".join(
        f"[{index}] {chunk.source}, page {chunk.page_number}\n{chunk.text}"
        for index, chunk in enumerate(context, start=1)
    )


class OpenAIEmbedder:
    """Adapt the OpenAI embeddings endpoint to the application's embedder interface."""

    def __init__(
        self,
        *,
        client: Any,
        model: str = DEFAULT_EMBEDDING_MODEL,
        batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self._client = client
        self._model = model
        self._batch_size = batch_size

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one vector per input text, in the same order as the inputs."""

        if not texts:
            return []
        if any(not text.strip() for text in texts):
            raise ValueError("embedding input must not contain empty text")

        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start : start + self._batch_size])
            response = self._client.embeddings.create(
                model=self._model,
                input=batch,
                encoding_format="float",
            )
            data = sorted(response.data, key=lambda item: item.index)
            if len(data) != len(batch):
                raise RuntimeError("embedding provider returned the wrong number of vectors")
            vectors.extend(list(item.embedding) for item in data)
        return vectors


class OpenAIAnswerer:
    """Adapt the Responses API to grounded answering over retrieved chunks."""

    def __init__(self, *, client: Any, model: str) -> None:
        self._client = client
        self._model = model

    def answer(self, question: str, context: Sequence[DocumentChunk]) -> str:
        if not context:
            raise ValueError("answer context must not be empty")

        response = self._client.responses.create(
            model=self._model,
            instructions=GROUNDING_INSTRUCTIONS,
            input=f"Question: {question}\n\nDocument excerpts:\n{_source_text(context)}",
        )
        answer = response.output_text.strip()
        if not answer:
            raise RuntimeError("the model returned an empty grounded answer")
        return answer


class ChatCompletionAnswerer:
    """Adapt OpenAI-compatible chat-completions providers such as OpenRouter."""

    def __init__(self, *, client: Any, model: str) -> None:
        self._client = client
        self._model = model

    def answer(self, question: str, context: Sequence[DocumentChunk]) -> str:
        if not context:
            raise ValueError("answer context must not be empty")

        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": GROUNDING_INSTRUCTIONS},
                {
                    "role": "user",
                    "content": f"Question: {question}\n\nDocument excerpts:\n{_source_text(context)}",
                },
            ],
        )
        answer = (response.choices[0].message.content or "").strip()
        if not answer:
            raise RuntimeError("the model returned an empty grounded answer")
        return answer
