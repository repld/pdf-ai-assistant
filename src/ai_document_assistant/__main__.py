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
    args = parser.parse_args(argv)

    if args.pdf:
        try:
            assistant = build_rag_assistant()
            summary = index_pdfs(assistant, args.pdf)
        except (ConfigurationError, DocumentError, OpenAIError, RuntimeError, ValueError) as error:
            print(f"Error: {error}", file=sys.stderr)
            return 1

        print(f"Indexed {summary.page_count} pages into {summary.chunk_count} chunks.")

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
