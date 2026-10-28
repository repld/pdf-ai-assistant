"""Command-line entry point for the AI Document Assistant."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from openai import OpenAIError

from .application import build_rag_assistant, format_grounded_answer, index_pdfs
from .assistant import ConfigurationError, ask_question
from .cli import run_interactive
from .documents import DocumentError
from .evaluation import evaluate, load_cases


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ask an AI model a question.")
    parser.add_argument(
        "question",
        nargs="?",
        help="the question to send to the model; omit it for interactive mode",
    )
    parser.add_argument(
        "--pdf",
        action="append",
        type=Path,
        default=[],
        help="PDF to index; repeat this option for multiple PDFs",
    )
    parser.add_argument(
        "--database",
        type=Path,
        help="SQLite database for persistent chunks and vectors",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        help="only read this many pages from each PDF for a smoke test",
    )
    parser.add_argument(
        "--page-start",
        type=int,
        default=1,
        help="first page to read from each PDF",
    )
    parser.add_argument(
        "--page-end",
        type=int,
        help="last page to read from each PDF, inclusive",
    )
    parser.add_argument(
        "--evaluate",
        type=Path,
        help="JSON evaluation file; requires --database",
    )
    parser.add_argument(
        "--evaluation-report",
        type=Path,
        help="write a Markdown evaluation report; requires --evaluate",
    )
    args = parser.parse_args(argv)

    if args.evaluate and args.database is None:
        parser.error("--evaluate requires --database")
    if args.evaluation_report and args.evaluate is None:
        parser.error("--evaluation-report requires --evaluate")

    if args.pdf or args.database or args.evaluate:
        try:
            assistant = build_rag_assistant(database_path=args.database)
            if args.pdf:
                summary = index_pdfs(
                    assistant,
                    args.pdf,
                    max_pages=args.max_pages,
                    page_start=args.page_start,
                    page_end=args.page_end,
                )
            else:
                summary = None
        except (ConfigurationError, DocumentError, OpenAIError, RuntimeError, ValueError) as error:
            print(f"Error: {error}", file=sys.stderr)
            return 1

        if summary is not None:
            print(f"Indexed {summary.page_count} pages into {summary.chunk_count} chunks.")

        if args.evaluate:
            try:
                report = evaluate(assistant, load_cases(args.evaluate))
            except (OpenAIError, RuntimeError, TypeError, ValueError, KeyError) as error:
                print(f"Error: {error}", file=sys.stderr)
                return 1
            print(f"Cases: {report.case_count}")
            print(f"Retrieval hit rate: {report.retrieval_hit_rate:.1%}")
            print(f"Retrieval recall: {report.retrieval_recall:.1%}")
            print(f"Supported answer rate: {report.answer_rate:.1%}")
            print(f"Citation rate: {report.citation_rate:.1%}")
            print(f"Citation coverage: {report.citation_coverage:.1%}")
            print(f"Required-fact coverage: {report.required_fact_rate:.1%}")
            for reason, count in report.failure_counts.items():
                print(f"Failure {reason}: {count}")
            for result in report.results:
                print(
                    f"- {result.case_id}: retrieval={'yes' if result.retrieval_hit else 'no'}, "
                    f"recall={result.retrieval_recall:.1%}, "
                    f"citations={result.citation_coverage:.1%}"
                )
            if args.evaluation_report:
                args.evaluation_report.parent.mkdir(parents=True, exist_ok=True)
                args.evaluation_report.write_text(report.to_markdown(), encoding="utf-8")
            return 0

        def ask_documents(question: str) -> str:
            return format_grounded_answer(assistant.answer(question))

        if args.question is None:
            run_interactive(ask=ask_documents)
        else:
            try:
                print(ask_documents(args.question))
            except (OpenAIError, RuntimeError, ValueError) as error:
                print(f"Error: {error}", file=sys.stderr)
                return 1
        return 0

    if args.question is None:
        run_interactive()
        return 0

    try:
        print(ask_question(args.question))
    except (ConfigurationError, OpenAIError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
