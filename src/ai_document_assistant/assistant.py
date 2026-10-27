"""Core question-answering function for the first AI milestone."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

DEFAULT_MODEL = "gpt-5-mini"


class ConfigurationError(RuntimeError):
    """Raised when the application is not configured to call the AI API."""


@dataclass(frozen=True)
class Settings:
    """Configuration needed to make one model request."""

    api_key: str
    model: str


def load_settings() -> Settings:
    """Load local environment configuration without exposing the API key."""

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ConfigurationError(
            "OPENAI_API_KEY is not configured. Add it to .env or set it in your shell."
        )

    return Settings(
        api_key=api_key,
        model=os.getenv("OPENAI_MODEL", DEFAULT_MODEL),
    )


def ask_question(
    question: str,
    *,
    client: Any | None = None,
    model: str | None = None,
) -> str:
    """Ask an AI model a question and return its text answer.

    ``client`` is injectable so this function can be tested without a network
    request or API usage.
    """

    question = question.strip()
    if not question:
        raise ValueError("question must not be empty")

    if client is None:
        settings = load_settings()
        client = OpenAI(api_key=settings.api_key)
        model = model or settings.model
    else:
        model = model or DEFAULT_MODEL

    response = client.responses.create(model=model, input=question)
    answer = response.output_text.strip()
    if not answer:
        raise RuntimeError("the model returned an empty answer")
    return answer
