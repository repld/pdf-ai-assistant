"""Small, reproducible retrieval and answerability evaluation."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .rag import RAGAnswer


class EvaluatedAssistant(Protocol):
    def answer(self, question: str) -> RAGAnswer: ...


@dataclass(frozen=True)
class EvaluationCase:
    """A question and the chunk IDs expected to support it."""

    case_id: str
    question: str
    expected_chunk_ids: frozenset[str]


def load_cases(path: Path) -> list[EvaluationCase]:
    """Load evaluation cases from a JSON array."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError("evaluation file must contain a JSON array")

    cases = []
    for item in payload:
        if not isinstance(item, dict):
            raise TypeError("each evaluation case must be a JSON object")
        cases.append(
            EvaluationCase(
                case_id=str(item["case_id"]),
                question=str(item["question"]),
                expected_chunk_ids=frozenset(item["expected_chunk_ids"]),
            )
        )
    return cases


@dataclass(frozen=True)
class EvaluationResult:
    case_id: str
    retrieval_hit: bool
    answer_present: bool


@dataclass(frozen=True)
class EvaluationReport:
    results: tuple[EvaluationResult, ...]

    @property
    def case_count(self) -> int:
        return len(self.results)

    @property
    def retrieval_hit_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(result.retrieval_hit for result in self.results) / len(self.results)

    @property
    def answer_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(result.answer_present for result in self.results) / len(self.results)


def evaluate(
    assistant: EvaluatedAssistant,
    cases: Sequence[EvaluationCase],
) -> EvaluationReport:
    results = []
    for case in cases:
        answer = assistant.answer(case.question)
        actual_chunk_ids = {citation.chunk_id for citation in answer.citations}
        results.append(
            EvaluationResult(
                case_id=case.case_id,
                retrieval_hit=bool(actual_chunk_ids & case.expected_chunk_ids),
                answer_present=bool(answer.answer.strip()),
            )
        )
    return EvaluationReport(results=tuple(results))
