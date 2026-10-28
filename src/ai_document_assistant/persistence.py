"""SQLite persistence for indexed document chunks and their vectors."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from pathlib import Path

from .documents import DocumentChunk
from .retrieval import SearchResult, hybrid_score


class SQLiteVectorStore:
    """Persist vector-search inputs in SQLite and search them in Python."""

    def __init__(self, path: Path) -> None:
        self._connection = sqlite3.connect(path)
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS document_chunks (
                chunk_id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                page_number INTEGER NOT NULL,
                text TEXT NOT NULL,
                vector_json TEXT NOT NULL
            )
            """
        )
        self._connection.commit()

    def add(
        self,
        chunks: Sequence[DocumentChunk],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must contain the same number of items")

        self._connection.executemany(
            """
            INSERT OR REPLACE INTO document_chunks
                (chunk_id, source, page_number, text, vector_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (
                    chunk.chunk_id,
                    chunk.source,
                    chunk.page_number,
                    chunk.text,
                    json.dumps(list(vector)),
                )
                for chunk, vector in zip(chunks, vectors, strict=True)
            ],
        )
        self._connection.commit()

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

        rows = self._connection.execute(
            "SELECT chunk_id, source, page_number, text, vector_json FROM document_chunks"
        ).fetchall()
        results: list[SearchResult] = []
        for chunk_id, source, page_number, text, vector_json in rows:
            vector = json.loads(vector_json)
            score = hybrid_score(query_vector, vector, query_text, text)
            if score >= min_score:
                results.append(
                    SearchResult(
                        chunk=DocumentChunk(
                            chunk_id=chunk_id,
                            source=source,
                            page_number=page_number,
                            text=text,
                        ),
                        score=score,
                    )
                )

        results.sort(key=lambda result: (-result.score, result.chunk.chunk_id))
        return results[:limit]

    def close(self) -> None:
        self._connection.close()
