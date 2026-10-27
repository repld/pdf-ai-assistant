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

    provider: str
    api_key: str
    base_url: str | None
    model: str
    embedding_model: str


def load_settings() -> Settings:
    """Load local environment configuration without exposing the API key."""

    load_dotenv()
    provider = os.getenv("AI_PROVIDER", "openai").strip().lower()
    if provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        model = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-v3.2")
        embedding_model = os.getenv(
            "OPENROUTER_EMBEDDING_MODEL",
            "openai/text-embedding-3-small",
        )
    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        model = os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
        embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    else:
        raise ConfigurationError(
            f"unsupported AI_PROVIDER '{provider}'. Use 'openai' or 'openrouter'."
        )

    if not api_key:
        raise ConfigurationError(
            f"API key for provider '{provider}' is not configured. "
            "Add it to .env or set it in your shell."
        )

    return Settings(
        provider=provider,
        api_key=api_key,
        base_url=base_url,
        model=model,
        embedding_model=embedding_model,
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
        client_kwargs = {"api_key": settings.api_key}
        if settings.base_url:
            client_kwargs["base_url"] = settings.base_url
        client = OpenAI(**client_kwargs, timeout=30.0, max_retries=2)
        model = model or settings.model
        provider = settings.provider
    else:
        model = model or DEFAULT_MODEL
        provider = "openai"

    if provider == "openrouter":
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": question}],
        )
        answer = (response.choices[0].message.content or "").strip()
    else:
        response = client.responses.create(model=model, input=question)
        answer = response.output_text.strip()
    if not answer:
        raise RuntimeError("the model returned an empty answer")
    return answer
