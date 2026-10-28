"""Small telemetry seam for request timing and operational diagnostics."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Protocol


class Telemetry(Protocol):
    """Record an operational event without exposing application internals."""

    def record(self, event: str, **attributes: object) -> None: ...


@dataclass(frozen=True)
class Observation:
    event: str
    attributes: dict[str, object]


class InMemoryTelemetry:
    """Collect observations for tests and local diagnostics."""

    def __init__(self) -> None:
        self.observations: list[Observation] = []

    def record(self, event: str, **attributes: object) -> None:
        self.observations.append(Observation(event=event, attributes=attributes))


class LoggingTelemetry:
    """Write structured JSON events to the application's logger."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._logger = logger or logging.getLogger("ai_document_assistant.telemetry")

    def record(self, event: str, **attributes: object) -> None:
        payload = {"event": event, **attributes}
        self._logger.info("%s", json.dumps(payload, sort_keys=True, default=str))
