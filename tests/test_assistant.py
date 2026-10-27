from types import SimpleNamespace

from ai_document_assistant import ask_question


class FakeResponses:
    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []

    def create(self, **kwargs: str) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(output_text="  A test answer.  ")


class FakeClient:
    def __init__(self) -> None:
        self.responses = FakeResponses()


def test_ask_question_sends_text_and_returns_clean_answer() -> None:
    client = FakeClient()

    answer = ask_question("  What is Python?  ", client=client, model="test-model")

    assert answer == "A test answer."
    assert client.responses.calls == [
        {"model": "test-model", "input": "What is Python?"}
    ]
