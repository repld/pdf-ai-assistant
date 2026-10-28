"""Retrieval-augmented question answering over indexed document chunks."""

from __future__ import annotations

import re
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from .documents import DocumentChunk
from .observability import LoggingTelemetry, Telemetry
from .retrieval import InMemoryVectorStore, VectorStore

UNSUPPORTED_ANSWER = "I could not find an answer in the uploaded documents."
BROAD_QUESTION_TOP_K = 20
_CITATION_PATTERN = re.compile(r"\[(\d+)\]")
_BROAD_SCOPE_PATTERN = re.compile(r"\b(?:all|every|each|complete|entire|full|whole)\b")
_BROAD_LIST_PATTERN = re.compile(
    r"\b(?:list|enumerate|name|items?|types?|categories?|kinds?|features?)\b"
)


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
    text: str = ""
    score: float | None = None


@dataclass(frozen=True)
class AnswerDiagnostics:
    """Operational details safe to display to a local user."""

    request_id: str
    total_ms: float
    retrieval_ms: float
    generation_ms: float
    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True)
class RAGAnswer:
    """Answer text plus the chunks used to produce it."""

    answer: str
    citations: tuple[Citation, ...]
    retrieved_chunk_ids: tuple[str, ...] = ()
    diagnostics: AnswerDiagnostics | None = None


class RAGAssistant:
    """Index chunks and answer questions using only retrieved context."""

    def __init__(
        self,
        *,
        embedder: Embedder,
        answerer: Answerer,
        store: VectorStore | None = None,
        min_score: float = 0.2,
        telemetry: Telemetry | None = None,
        model_name: str | None = None,
    ) -> None:
        self._embedder = embedder
        self._answerer = answerer
        self._store = store or InMemoryVectorStore()
        self._min_score = min_score
        self._telemetry = telemetry or LoggingTelemetry()
        self._model_name = model_name

    def index(self, chunks: Sequence[DocumentChunk]) -> None:
        if not chunks:
            raise ValueError("at least one document chunk is required")
        vectors = self._embedder.embed([chunk.text for chunk in chunks])
        self._store.add(chunks, vectors)

    def answer(
        self,
        question: str,
        *,
        top_k: int = 5,
        on_token: Callable[[str], None] | None = None,
    ) -> RAGAnswer:
        request_id = uuid.uuid4().hex[:12]
        started_at = time.perf_counter()
        retrieval_ms = 0.0
        generation_ms = 0.0
        retrieved_count = 0
        citation_count = 0
        status = "error"
        usage: dict[str, int] | None = None

        def record_metrics() -> AnswerDiagnostics:
            diagnostics = AnswerDiagnostics(
                request_id=request_id,
                total_ms=round(_elapsed_ms(started_at), 2),
                retrieval_ms=round(retrieval_ms, 2),
                generation_ms=round(generation_ms, 2),
                model=self._model_name,
                input_tokens=usage.get("input_tokens") if usage else None,
                output_tokens=usage.get("output_tokens") if usage else None,
                total_tokens=usage.get("total_tokens") if usage else None,
            )
            self._record_answer_metrics(
                request_id=request_id,
                started_at=started_at,
                retrieval_ms=retrieval_ms,
                generation_ms=generation_ms,
                retrieved_count=retrieved_count,
                citation_count=citation_count,
                status=status,
                input_tokens=usage.get("input_tokens") if usage else None,
                output_tokens=usage.get("output_tokens") if usage else None,
                total_tokens=usage.get("total_tokens") if usage else None,
            )
            return diagnostics

        question = question.strip()
        if not question:
            record_metrics()
            raise ValueError("question must not be empty")
        if top_k <= 0:
            record_metrics()
            raise ValueError("top_k must be positive")

        try:
            query_vector = self._embedder.embed([question])[0]
        except Exception:
            record_metrics()
            raise
        search_limit, search_min_score = _search_parameters(question, top_k, self._min_score)
        retrieval_started_at = time.perf_counter()
        try:
            matches = self._store.search(
                query_vector,
                query_text=question,
                limit=search_limit,
                min_score=search_min_score,
            )
        except Exception:
            retrieval_ms = _elapsed_ms(retrieval_started_at)
            record_metrics()
            raise
        retrieval_ms = _elapsed_ms(retrieval_started_at)
        retrieved_count = len(matches)
        if not matches:
            status = "unsupported"
            return _unsupported_result(diagnostics=record_metrics())

        context = [match.chunk for match in matches]
        scores = tuple(match.score for match in matches)
        retrieved_chunk_ids = tuple(chunk.chunk_id for chunk in context)
        generation_started_at = time.perf_counter()
        try:
            answer_stream = getattr(self._answerer, "answer_stream", None)
            if on_token is not None and callable(answer_stream):
                answer = answer_stream(question, context, on_token)
            else:
                answer = self._answerer.answer(question, context)
            usage = getattr(self._answerer, "last_usage", None)
        except Exception:
            generation_ms = _elapsed_ms(generation_started_at)
            record_metrics()
            raise
        generation_ms = _elapsed_ms(generation_started_at)
        if answer.strip() == UNSUPPORTED_ANSWER:
            status = "unsupported"
            return _unsupported_result(
                retrieved_chunk_ids,
                diagnostics=record_metrics(),
            )

        citations = _citations_for_answer(answer, context, scores)
        citation_count = len(citations)
        if not citations:
            status = "uncited"
            return _unsupported_result(
                retrieved_chunk_ids,
                diagnostics=record_metrics(),
            )

        status = "success"
        diagnostics = record_metrics()
        return RAGAnswer(
            answer=answer,
            citations=citations,
            retrieved_chunk_ids=retrieved_chunk_ids,
            diagnostics=diagnostics,
        )

    def _record_answer_metrics(
        self,
        *,
        request_id: str,
        started_at: float,
        retrieval_ms: float,
        generation_ms: float,
        retrieved_count: int,
        citation_count: int,
        status: str,
        input_tokens: int | None,
        output_tokens: int | None,
        total_tokens: int | None,
    ) -> None:
        self._telemetry.record(
            "rag.answer",
            request_id=request_id,
            duration_ms=round(_elapsed_ms(started_at), 2),
            retrieval_ms=round(retrieval_ms, 2),
            generation_ms=round(generation_ms, 2),
            retrieved_count=retrieved_count,
            citation_count=citation_count,
            status=status,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )


def _unsupported_result(
    retrieved_chunk_ids: tuple[str, ...] = (),
    *,
    diagnostics: AnswerDiagnostics | None = None,
) -> RAGAnswer:
    return RAGAnswer(
        answer=UNSUPPORTED_ANSWER,
        citations=(),
        retrieved_chunk_ids=retrieved_chunk_ids,
        diagnostics=diagnostics,
    )


def _citations_for_answer(
    answer: str,
    context: Sequence[DocumentChunk],
    scores: Sequence[float],
) -> tuple[Citation, ...]:
    citation_indexes = _citation_indexes(answer, len(context))
    return tuple(
        Citation(
            chunk_id=context[index - 1].chunk_id,
            source=context[index - 1].source,
            page_number=context[index - 1].page_number,
            text=context[index - 1].text,
            score=scores[index - 1],
        )
        for index in citation_indexes
    )


def _search_parameters(
    question: str,
    top_k: int,
    min_score: float,
) -> tuple[int, float]:
    """Use a wider, higher-recall search for exhaustive list questions."""

    if _is_broad_question(question):
        return max(top_k, BROAD_QUESTION_TOP_K), 0.0
    return top_k, min_score


def _is_broad_question(question: str) -> bool:
    return bool(
        _BROAD_SCOPE_PATTERN.search(question)
        and _BROAD_LIST_PATTERN.search(question)
    )


def _elapsed_ms(started_at: float) -> float:
    return (time.perf_counter() - started_at) * 1000


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
