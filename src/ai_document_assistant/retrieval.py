"""Vector similarity retrieval over in-memory document chunks."""

from __future__ import annotations

import math
import re
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
        query_text: str | None = None,
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
        query_text: str | None = None,
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
            score = hybrid_score(query_vector, vector, query_text, chunk.text)
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


_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def hybrid_score(
    query_vector: Sequence[float],
    document_vector: Sequence[float],
    query_text: str | None,
    document_text: str,
) -> float:
    """Combine semantic similarity with lightweight keyword overlap."""

    semantic_score = max(cosine_similarity(query_vector, document_vector), 0.0)
    if not query_text:
        return semantic_score

    query_terms = set(_TOKEN_PATTERN.findall(query_text.lower()))
    document_terms = set(_TOKEN_PATTERN.findall(document_text.lower()))
    if not query_terms:
        return semantic_score

    lexical_score = len(query_terms & document_terms) / len(query_terms)
    return 0.75 * semantic_score + 0.25 * lexical_score
