"""Vector similarity retrieval over in-memory document chunks."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from .documents import DocumentChunk


@dataclass(frozen=True)
class SearchResult:
    """A chunk and its cosine similarity to a query vector."""

    chunk: DocumentChunk
    score: float


class VectorStore(Protocol):
    def add(
        self,
        chunks: Sequence[DocumentChunk],
        vectors: Sequence[Sequence[float]],
    ) -> None: ...

    def search(
        self,
        query_vector: Sequence[float],
        *,
        limit: int = 3,
        min_score: float = 0.0,
    ) -> list[SearchResult]: ...


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

        results = []
        for chunk, vector in self._items:
            score = cosine_similarity(query_vector, vector)
            if score >= min_score:
                results.append(SearchResult(chunk=chunk, score=score))

        results.sort(key=lambda result: (-result.score, result.chunk.chunk_id))
        return results[:limit]


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Return cosine similarity, treating a zero vector as unrelated."""

    if len(left) != len(right):
        raise ValueError("vectors must have the same dimension")

    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0

    dot = sum(left_value * right_value for left_value, right_value in zip(left, right, strict=True))
    return dot / (left_norm * right_norm)
