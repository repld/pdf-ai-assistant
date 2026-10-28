"""Small, reproducible retrieval and answerability evaluation."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .rag import UNSUPPORTED_ANSWER, RAGAnswer


class EvaluatedAssistant(Protocol):
    def answer(self, question: str) -> RAGAnswer: ...


@dataclass(frozen=True)
class EvaluationCase:
    """A question and the chunk IDs expected to support it."""

    case_id: str
    question: str
    expected_chunk_ids: frozenset[str]
    reference_answer: str | None = None
    required_facts: frozenset[str] = frozenset()


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
                expected_chunk_ids=frozenset(
                    str(chunk_id) for chunk_id in item["expected_chunk_ids"]
                ),
                reference_answer=(
                    str(item["reference_answer"])
                    if item.get("reference_answer") is not None
                    else None
                ),
                required_facts=frozenset(
                    str(fact) for fact in item.get("required_facts", [])
                ),
            )
        )
    return cases


@dataclass(frozen=True)
class EvaluationResult:
    case_id: str
    retrieval_hit: bool
    retrieval_recall: float
    answer_present: bool
    citations_present: bool
    citation_coverage: float
    required_fact_coverage: float
    failure_reasons: tuple[str, ...] = ()


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

    @property
    def citation_rate(self) -> float:
        """Return the fraction of cases that produced at least one citation."""

        if not self.results:
            return 0.0
        return sum(result.citations_present for result in self.results) / len(self.results)

    @property
    def retrieval_recall(self) -> float:
        return _average(result.retrieval_recall for result in self.results)

    @property
    def citation_coverage(self) -> float:
        return _average(result.citation_coverage for result in self.results)

    @property
    def required_fact_rate(self) -> float:
        return _average(result.required_fact_coverage for result in self.results)

    @property
    def failure_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for result in self.results:
            for reason in result.failure_reasons:
                counts[reason] = counts.get(reason, 0) + 1
        return dict(sorted(counts.items()))

    def to_markdown(self) -> str:
        lines = [
            "# Evaluation report",
            "",
            f"Cases: {self.case_count}",
            f"Retrieval hit rate: {self.retrieval_hit_rate:.1%}",
            f"Retrieval recall: {self.retrieval_recall:.1%}",
            f"Supported answer rate: {self.answer_rate:.1%}",
            f"Citation presence rate: {self.citation_rate:.1%}",
            f"Citation coverage: {self.citation_coverage:.1%}",
            f"Required-fact coverage: {self.required_fact_rate:.1%}",
            "",
            "## Failure counts",
            "",
        ]
        if self.failure_counts:
            lines.extend(f"- `{reason}`: {count}" for reason, count in self.failure_counts.items())
        else:
            lines.append("No failures recorded.")

        lines.extend(["", "## Cases", "", "| Case | Recall | Answer | Citations | Failures |", "|---|---:|---:|---:|---|"])
        for result in self.results:
            failures = ", ".join(result.failure_reasons) or "-"
            lines.append(
                f"| {result.case_id} | {result.retrieval_recall:.1%} | "
                f"{'yes' if result.answer_present else 'no'} | "
                f"{result.citation_coverage:.1%} | {failures} |"
            )
        return "\n".join(lines) + "\n"


def evaluate(
    assistant: EvaluatedAssistant,
    cases: Sequence[EvaluationCase],
) -> EvaluationReport:
    results = []
    for case in cases:
        answer = assistant.answer(case.question)
        actual_chunk_ids = set(answer.retrieved_chunk_ids)
        if not actual_chunk_ids:
            actual_chunk_ids = {citation.chunk_id for citation in answer.citations}
        cited_chunk_ids = {citation.chunk_id for citation in answer.citations}
        answer_present = bool(
            answer.answer.strip() and answer.answer.strip() != UNSUPPORTED_ANSWER
        )
        retrieval_recall = _coverage(actual_chunk_ids, case.expected_chunk_ids)
        citation_coverage = _coverage(cited_chunk_ids, case.expected_chunk_ids)
        required_fact_coverage = _text_coverage(answer.answer, case.required_facts)
        failure_reasons: list[str] = []
        if case.expected_chunk_ids and not actual_chunk_ids & case.expected_chunk_ids:
            failure_reasons.append("retrieval_miss")
        if not answer_present:
            failure_reasons.append("unsupported_answer")
        if case.expected_chunk_ids and not cited_chunk_ids & case.expected_chunk_ids:
            failure_reasons.append("citation_miss")
        if case.required_facts and required_fact_coverage < 1.0:
            failure_reasons.append("missing_required_facts")
        results.append(
            EvaluationResult(
                case_id=case.case_id,
                retrieval_hit=bool(actual_chunk_ids & case.expected_chunk_ids),
                retrieval_recall=retrieval_recall,
                answer_present=answer_present,
                citations_present=bool(answer.citations),
                citation_coverage=citation_coverage,
                required_fact_coverage=required_fact_coverage,
                failure_reasons=tuple(failure_reasons),
            )
        )
    return EvaluationReport(results=tuple(results))


def _average(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _coverage(actual: set[str], expected: frozenset[str]) -> float:
    if not expected:
        return 1.0
    return len(actual & expected) / len(expected)


def _text_coverage(answer: str, required_facts: frozenset[str]) -> float:
    if not required_facts:
        return 1.0
    normalized_answer = answer.casefold()
    matched = sum(fact.casefold() in normalized_answer for fact in required_facts)
    return matched / len(required_facts)
