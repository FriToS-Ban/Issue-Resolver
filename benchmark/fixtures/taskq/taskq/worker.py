"""Executes queued tasks."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Callable, Mapping

from taskq.clock import utcnow
from taskq.events import EventLog
from taskq.queue import TaskQueue
from taskq.retry import RetryPolicy, backoff_delay, should_retry
from taskq.task import Task, TaskStatus

Handler = Callable[[dict[str, Any]], Any]


class Worker:
    """Pulls due tasks off a queue and runs their handlers."""

    def __init__(
        self,
        queue: TaskQueue,
        handlers: Mapping[str, Handler],
        *,
        retry: RetryPolicy | None = None,
        clock: Callable[[], datetime] = utcnow,
        batch_size: int = 10,
        name: str = "worker",
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self.queue = queue
        self.handlers = dict(handlers)
        self.retry = retry or RetryPolicy()
        self.clock = clock
        self.batch_size = batch_size
        self.name = name
        self.events = EventLog()

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        handlers: Mapping[str, Handler],
        *,
        queue: TaskQueue | None = None,
        clock: Callable[[], datetime] = utcnow,
    ) -> "Worker":
        retry = RetryPolicy.from_config(config["retry"])
        worker_cfg = config["worker"]
        if queue is None:
            queue = TaskQueue(max_size=config["queue"]["max_size"])
        return cls(
            queue,
            handlers,
            retry=retry,
            clock=clock,
            batch_size=int(worker_cfg["batch_size"]),
            name=str(worker_cfg["name"]),
        )

    def register(self, name: str, handler: Handler) -> None:
        self.handlers[name] = handler

    def _is_due(self, task: Task, now: datetime) -> bool:
        return task.next_run is None or task.next_run <= now

    def run_pending(self) -> list[Task]:
        """Run up to ``batch_size`` due tasks; returns the tasks that were run."""
        now = self.clock()
        due: list[Task] = []
        later: list[Task] = []
        while self.queue and len(due) < self.batch_size:
            task = self.queue.pop()
            if task.status is not TaskStatus.PENDING:
                continue
            if self._is_due(task, now):
                due.append(task)
            else:
                later.append(task)
        for task in later:
            self.queue.push(task)
        for task in due:
            self._execute(task, now)
        return due

    def run_until_empty(self, max_rounds: int = 100) -> int:
        """Keep running batches until nothing is due.  Returns tasks run."""
        total = 0
        for _ in range(max_rounds):
            ran = self.run_pending()
            if not ran:
                break
            total += len(ran)
        return total

    def _execute(self, task: Task, now: datetime) -> None:
        task.attempts += 1
        task.status = TaskStatus.RUNNING
        self.events.record("started", task.id, now)
        handler = self.handlers.get(task.name)
        if handler is None:
            task.status = TaskStatus.FAILED
            task.last_error = f"no handler registered for {task.name!r}"
            self.events.record("failed", task.id, now, task.last_error)
            return
        try:
            handler(task.payload)
        except Exception as exc:
            task.last_error = f"{type(exc).__name__}: {exc}"
            if should_retry(task, exc, self.retry):
                delay = backoff_delay(task.attempts, self.retry)
                task.status = TaskStatus.PENDING
                task.next_run = now + timedelta(seconds=delay)
                self.queue.push(task)
                self.events.record("retry_scheduled", task.id, now, f"in {delay:g}s")
            else:
                task.status = TaskStatus.FAILED
                self.events.record("failed", task.id, now, task.last_error)
        else:
            task.status = TaskStatus.DONE
            task.last_error = None
            self.events.record("done", task.id, now)
