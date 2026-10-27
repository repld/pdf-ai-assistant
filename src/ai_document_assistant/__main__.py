"""Command-line entry point for the AI Document Assistant."""

from __future__ import annotations

import argparse
import sys

from openai import OpenAIError

from .assistant import ConfigurationError, ask_question
from .cli import run_interactive


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ask an AI model a question.")
    parser.add_argument(
        "question",
        nargs="?",
        help="the question to send to the model; omit it for interactive mode",
    )
    args = parser.parse_args(argv)

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
