"""A small in-memory log of what happened to tasks, for debugging and metrics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterator


@dataclass(frozen=True)
class Event:
    kind: str
    task_id: str
    at: datetime
    detail: str = ""


class EventLog:
    def __init__(self, limit: int = 1000) -> None:
        self.limit = limit
        self._events: list[Event] = []

    def record(self, kind: str, task_id: str, at: datetime, detail: str = "") -> Event:
        event = Event(kind, task_id, at, detail)
        self._events.append(event)
        if len(self._events) > self.limit:
            del self._events[: len(self._events) - self.limit]
        return event

    def of_kind(self, kind: str) -> list[Event]:
        return [e for e in self._events if e.kind == kind]

    def for_task(self, task_id: str) -> list[Event]:
        return [e for e in self._events if e.task_id == task_id]

    def since(self, moment: datetime) -> list[Event]:
        return [e for e in self._events if e.at >= moment]

    def __len__(self) -> int:
        return len(self._events)

    def __iter__(self) -> Iterator[Event]:
        return iter(list(self._events))
