"""The :class:`Task` record and its (de)serialisation."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Mapping


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_final(self) -> bool:
        return self in (TaskStatus.DONE, TaskStatus.FAILED, TaskStatus.CANCELLED)


def new_task_id() -> str:
    return uuid.uuid4().hex[:12]


def _format_dt(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _parse_dt(value: str | datetime | None) -> datetime | None:
    if value is None or isinstance(value, datetime):
        return value
    return datetime.fromisoformat(value)


@dataclass
class Task:
    """A unit of work: a handler name plus a JSON-serialisable payload."""

    name: str
    payload: dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    id: str = field(default_factory=new_task_id)
    status: TaskStatus = TaskStatus.PENDING
    attempts: int = 0
    tags: list[str] = field(default_factory=list)
    created_at: datetime | None = None
    next_run: datetime | None = None
    last_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "payload": dict(self.payload),
            "priority": self.priority,
            "status": self.status.value,
            "attempts": self.attempts,
            "tags": list(self.tags),
            "created_at": _format_dt(self.created_at),
            "next_run": _format_dt(self.next_run),
            "last_error": self.last_error,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Task":
        return cls(
            id=str(data["id"]),
            name=str(data["name"]),
            payload=dict(data.get("payload") or {}),
            priority=int(data.get("priority", 0)),
            status=TaskStatus(data.get("status", TaskStatus.PENDING.value)),
            attempts=int(data.get("attempts", 0)),
            tags=list(data.get("tags") or []),
            created_at=_parse_dt(data.get("created_at")),
            next_run=_parse_dt(data.get("next_run")),
            last_error=data.get("last_error"),
        )

    def has_tag(self, tag: str) -> bool:
        return tag in self.tags
