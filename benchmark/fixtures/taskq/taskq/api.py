"""The application-facing API: submit tasks, check on them, persist them."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, Iterable, Mapping

from taskq.clock import utcnow
from taskq.queue import TaskQueue
from taskq.storage import JsonStore
from taskq.task import Task, TaskStatus


class UnknownTask(KeyError):
    """No task with the given id has been submitted."""


class TaskService:
    def __init__(
        self,
        queue: TaskQueue | None = None,
        store: JsonStore | None = None,
        *,
        clock: Callable[[], datetime] = utcnow,
    ) -> None:
        self.queue = queue if queue is not None else TaskQueue()
        self.store = store
        self.clock = clock
        self._tasks: dict[str, Task] = {}

    @classmethod
    def from_config(cls, config: Mapping[str, Any], *, clock: Callable[[], datetime] = utcnow) -> "TaskService":
        path = config["storage"]["path"]
        return cls(
            TaskQueue(max_size=config["queue"]["max_size"]),
            JsonStore(path) if path else None,
            clock=clock,
        )

    def submit(
        self,
        name: str,
        payload: Mapping[str, Any] | None = None,
        *,
        priority: int = 0,
        tags: Iterable[str] = (),
        run_at: datetime | None = None,
        task_id: str | None = None,
    ) -> str:
        """Queue a new task and return its id."""
        task = Task(
            name=name,
            payload=dict(payload or {}),
            priority=priority,
            tags=list(tags),
            created_at=self.clock(),
            next_run=run_at,
        )
        if task_id is not None:
            if task_id in self._tasks:
                raise ValueError(f"duplicate task id {task_id!r}")
            task.id = task_id
        self.queue.push(task)
        self._tasks[task.id] = task
        return task.id

    def get(self, task_id: str) -> Task:
        try:
            return self._tasks[task_id]
        except KeyError:
            raise UnknownTask(task_id) from None

    def status(self, task_id: str) -> TaskStatus:
        return self.get(task_id).status

    def cancel(self, task_id: str) -> bool:
        """Cancel a pending task.  Returns False if it already ran or finished."""
        task = self.get(task_id)
        if task.status is not TaskStatus.PENDING:
            return False
        task.status = TaskStatus.CANCELLED
        self.queue.remove(task_id)
        return True

    def tasks(self, *, status: TaskStatus | None = None, tag: str | None = None) -> list[Task]:
        result = list(self._tasks.values())
        if status is not None:
            result = [t for t in result if t.status is status]
        if tag is not None:
            result = [t for t in result if t.has_tag(tag)]
        return result

    def save(self) -> None:
        if self.store is None:
            raise RuntimeError("no storage configured")
        self.store.save(self._tasks.values())

    def restore(self) -> int:
        """Load tasks from storage and queue the pending ones.  Returns tasks loaded."""
        if self.store is None:
            raise RuntimeError("no storage configured")
        loaded = self.store.load()
        for task in loaded:
            self._tasks[task.id] = task
            if task.status is TaskStatus.PENDING and task.id not in self.queue:
                self.queue.push(task)
        return len(loaded)
