from dataclasses import dataclass

from ai_document_assistant.evaluation import EvaluationCase, evaluate, load_cases
from ai_document_assistant.rag import Citation, RAGAnswer


@dataclass
class FakeAssistant:
    answers: dict[str, RAGAnswer]

    def answer(self, question: str) -> RAGAnswer:
        return self.answers[question]


def test_evaluation_reports_retrieval_hit_rate_and_answer_rate() -> None:
    assistant = FakeAssistant(
        {
            "What is Python?": RAGAnswer(
                answer="A programming language.",
                citations=(Citation("python", "guide.pdf", 1),),
                retrieved_chunk_ids=("python", "other"),
            ),
            "What is missing?": RAGAnswer(
                answer="I could not find an answer in the uploaded documents.",
                citations=(),
            ),
        }
    )
    cases = [
        EvaluationCase("python", "What is Python?", frozenset({"python"})),
        EvaluationCase("missing", "What is missing?", frozenset({"missing"})),
    ]

    report = evaluate(assistant, cases)

    assert report.case_count == 2
    assert report.retrieval_hit_rate == 0.5
    assert report.answer_rate == 1.0
    assert report.citation_rate == 0.5
    assert report.results[0].retrieval_hit is True
    assert report.results[0].citations_present is True
    assert report.results[1].retrieval_hit is False
    assert report.results[1].citations_present is False


def test_load_cases_reads_json_file(tmp_path) -> None:
    path = tmp_path / "evaluation.json"
    path.write_text(
        '[{"case_id": "one", "question": "What?", "expected_chunk_ids": ["c1"]}]',
        encoding="utf-8",
    )

    assert load_cases(path) == [EvaluationCase("one", "What?", frozenset({"c1"}))]
