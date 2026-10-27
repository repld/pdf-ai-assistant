"""Retrieval-augmented question answering over indexed document chunks."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from .documents import DocumentChunk
from .retrieval import InMemoryVectorStore, VectorStore

UNSUPPORTED_ANSWER = "I could not find an answer in the uploaded documents."
_CITATION_PATTERN = re.compile(r"\[(\d+)\]")


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
        store: VectorStore | None = None,
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
        if answer.strip() == UNSUPPORTED_ANSWER:
            return RAGAnswer(answer=UNSUPPORTED_ANSWER, citations=())
        citation_indexes = _citation_indexes(answer, len(context))
        if not citation_indexes:
            return RAGAnswer(answer=UNSUPPORTED_ANSWER, citations=())

        cited_context = [context[index - 1] for index in citation_indexes]
        citations = tuple(
            Citation(
                chunk_id=chunk.chunk_id,
                source=chunk.source,
                page_number=chunk.page_number,
            )
            for chunk in cited_context
        )
        return RAGAnswer(answer=answer, citations=citations)


def _citation_indexes(answer: str, context_count: int) -> list[int]:
    """Return valid, unique 1-based source indexes from a model answer."""

    indexes: list[int] = []
    for match in _CITATION_PATTERN.finditer(answer):
        index = int(match.group(1))
        if index < 1 or index > context_count:
            return []
        if index not in indexes:
            indexes.append(index)
    return indexes
