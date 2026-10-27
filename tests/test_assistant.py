from types import SimpleNamespace

import pytest

import ai_document_assistant.assistant as assistant_module
from ai_document_assistant import ConfigurationError, ask_question
from ai_document_assistant.cli import run_interactive


class FakeResponses:
    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []

    def create(self, **kwargs: str) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(output_text="  A test answer.  ")


class FakeClient:
    def __init__(self) -> None:
        self.responses = FakeResponses()


class EmptyAnswerResponses:
    def create(self, **kwargs: str) -> SimpleNamespace:
        return SimpleNamespace(output_text="   ")


class EmptyAnswerClient:
    def __init__(self) -> None:
        self.responses = EmptyAnswerResponses()


def test_ask_question_sends_text_and_returns_clean_answer() -> None:
    client = FakeClient()

    answer = ask_question("  What is Python?  ", client=client, model="test-model")

    assert answer == "A test answer."
    assert client.responses.calls == [
        {"model": "test-model", "input": "What is Python?"}
    ]


def test_empty_question_is_rejected() -> None:
    with pytest.raises(ValueError, match="question must not be empty"):
        ask_question("   ", client=FakeClient())


def test_missing_api_key_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(assistant_module, "load_dotenv", lambda: None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        assistant_module.load_settings()


def test_empty_model_answer_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="empty answer"):
        ask_question("What is Python?", client=EmptyAnswerClient())


def test_interactive_mode_answers_questions_and_stops() -> None:
    inputs = iter(["What is Python?", "", "quit"])
    prompts: list[str] = []
    outputs: list[str] = []

    def input_fn(prompt: str) -> str:
        prompts.append(prompt)
        return next(inputs)

    run_interactive(
        ask=lambda question: "Python is a programming language.",
        input_fn=input_fn,
        output_fn=outputs.append,
    )

    assert prompts == ["You: ", "You: ", "You: "]
    assert outputs == [
        "Ask questions about AI. Type 'quit' or 'exit' to stop.",
        "Assistant: Python is a programming language.",
        "Please enter a question.",
        "Goodbye!",
    ]
