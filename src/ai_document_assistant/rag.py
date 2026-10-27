"""Retrieval-augmented question answering over indexed document chunks."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from .documents import DocumentChunk
from .retrieval import InMemoryVectorStore

UNSUPPORTED_ANSWER = "I could not find an answer in the uploaded documents."


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class Answerer(Protocol):
    def answer(self, question: str, context: Sequence[DocumentChunk]) -> str: ...


@dataclass(frozen=True)
class Citation:
    """A source reference attached to a grounded answer."""

    chunk_id: str
    source: str
    page_number: int


@dataclass(frozen=True)
class RAGAnswer:
    """Answer text plus the chunks used to produce it."""

    answer: str
    citations: tuple[Citation, ...]


class RAGAssistant:
    """Index chunks and answer questions using only retrieved context."""

    def __init__(
        self,
        *,
        embedder: Embedder,
        answerer: Answerer,
        store: InMemoryVectorStore | None = None,
        min_score: float = 0.2,
    ) -> None:
        self._embedder = embedder
        self._answerer = answerer
        self._store = store or InMemoryVectorStore()
        self._min_score = min_score

    def index(self, chunks: Sequence[DocumentChunk]) -> None:
        if not chunks:
            raise ValueError("at least one document chunk is required")
        vectors = self._embedder.embed([chunk.text for chunk in chunks])
        self._store.add(chunks, vectors)

    def answer(self, question: str, *, top_k: int = 3) -> RAGAnswer:
        question = question.strip()
        if not question:
            raise ValueError("question must not be empty")

        query_vector = self._embedder.embed([question])[0]
        matches = self._store.search(
            query_vector,
            limit=top_k,
            min_score=self._min_score,
        )
        if not matches:
            return RAGAnswer(answer=UNSUPPORTED_ANSWER, citations=())

        context = [match.chunk for match in matches]
        answer = self._answerer.answer(question, context)
        citations = tuple(
            Citation(
                chunk_id=chunk.chunk_id,
                source=chunk.source,
                page_number=chunk.page_number,
            )
            for chunk in context
        )
        return RAGAnswer(answer=answer, citations=citations)
