"""Command-line entry point for the first AI milestone."""

from __future__ import annotations

import argparse
import sys

from .assistant import ConfigurationError, ask_question


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ask an AI model one question.")
    parser.add_argument("question", help="the question to send to the model")
    args = parser.parse_args(argv)

    try:
        print(ask_question(args.question))
    except (ConfigurationError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
