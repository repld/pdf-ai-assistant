"""Vector similarity retrieval over in-memory document chunks."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from .documents import DocumentChunk


@dataclass(frozen=True)
class SearchResult:
    """A chunk and its cosine similarity to a query vector."""

    chunk: DocumentChunk
    score: float


class InMemoryVectorStore:
    """Store chunk/vector pairs and return the most similar chunks."""

    def __init__(self) -> None:
        self._items: list[tuple[DocumentChunk, list[float]]] = []
        self._dimension: int | None = None

    def add(
        self,
        chunks: Sequence[DocumentChunk],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must contain the same number of items")

        for chunk, vector in zip(chunks, vectors, strict=True):
            values = list(vector)
            if not values:
                raise ValueError("vectors must not be empty")
            if self._dimension is None:
                self._dimension = len(values)
            elif len(values) != self._dimension:
                raise ValueError("all vectors must have the same dimension")
            self._items.append((chunk, values))

    def search(
        self,
        query_vector: Sequence[float],
        *,
        limit: int = 3,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        if limit <= 0:
            raise ValueError("limit must be positive")
        if not self._items:
            return []
        if len(query_vector) != self._dimension:
            raise ValueError("query vector has the wrong dimension")

        query_values = list(query_vector)
        query_norm = math.sqrt(sum(value * value for value in query_values))
        if query_norm == 0:
            return []

        results = []
        for chunk, vector in self._items:
            vector_norm = math.sqrt(sum(value * value for value in vector))
            score = 0.0 if vector_norm == 0 else _dot(query_values, vector) / (query_norm * vector_norm)
            if score >= min_score:
                results.append(SearchResult(chunk=chunk, score=score))

        results.sort(key=lambda result: (-result.score, result.chunk.chunk_id))
        return results[:limit]


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(left_value * right_value for left_value, right_value in zip(left, right, strict=True))
