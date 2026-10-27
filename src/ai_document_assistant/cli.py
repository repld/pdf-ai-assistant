"""Command-line interaction for the AI Document Assistant."""

from __future__ import annotations

from collections.abc import Callable

from openai import OpenAIError

from .assistant import ConfigurationError, ask_question

QuestionAsker = Callable[[str], str]
InputFunction = Callable[[str], str]
OutputFunction = Callable[[str], None]


def run_interactive(
    *,
    ask: QuestionAsker = ask_question,
    input_fn: InputFunction = input,
    output_fn: OutputFunction = print,
) -> None:
    """Run a question loop until the user enters ``quit`` or ``exit``."""

    output_fn("Ask questions about AI. Type 'quit' or 'exit' to stop.")

    while True:
        try:
            question = input_fn("You: ")
        except (EOFError, KeyboardInterrupt):
            output_fn("\nGoodbye!")
            return

        if question.strip().lower() in {"quit", "exit"}:
            output_fn("Goodbye!")
            return

        if not question.strip():
            output_fn("Please enter a question.")
            continue

        try:
            answer = ask(question)
        except (ConfigurationError, OpenAIError, RuntimeError, ValueError) as error:
            output_fn(f"Error: {error}")
        else:
            output_fn(f"Assistant: {answer}")
