"""JSON file persistence for tasks."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

from taskq.task import Task

FORMAT_VERSION = 1


class StorageError(Exception):
    """The task file exists but cannot be read."""


class JsonStore:
    """Stores tasks in a single JSON document, replaced atomically on save."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)

    def save(self, tasks: Iterable[Task]) -> None:
        document = {"version": FORMAT_VERSION, "tasks": [task.to_dict() for task in tasks]}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self.path)

    def load(self) -> list[Task]:
        if not self.path.exists():
            return []
        try:
            document = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise StorageError(f"{self.path}: not valid JSON ({exc})") from exc
        # Files written before 0.4 are a bare list of task records.
        records = document["tasks"] if isinstance(document, dict) else document
        return [Task.from_dict(record) for record in records]

    def exists(self) -> bool:
        return self.path.exists()

    def clear(self) -> None:
        if self.path.exists():
            self.path.unlink()
