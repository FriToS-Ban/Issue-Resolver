"""In-memory priority queue of tasks.

Higher ``priority`` values are served first; tasks with equal priority are served
in the order they were pushed.
"""

from __future__ import annotations

import heapq
import itertools
from typing import Iterator

from taskq.task import Task


class QueueFull(Exception):
    """Raised when pushing onto a queue that has reached ``max_size``."""


class QueueEmpty(IndexError):
    """Raised when popping or peeking an empty queue."""


class TaskQueue:
    def __init__(self, max_size: int | None = None) -> None:
        self.max_size = max_size
        self._heap: list[tuple[int, int, Task]] = []
        self._counter = itertools.count()

    def push(self, task: Task) -> None:
        if self.max_size is not None and len(self._heap) >= self.max_size:
            raise QueueFull(f"queue is full ({self.max_size} tasks)")
        seq = next(self._counter)
        heapq.heappush(self._heap, (-task.priority, seq, task))

    def pop(self) -> Task:
        if not self._heap:
            raise QueueEmpty("pop from an empty task queue")
        return heapq.heappop(self._heap)[2]

    def peek(self) -> Task:
        if not self._heap:
            raise QueueEmpty("peek at an empty task queue")
        return self._heap[0][2]

    def remove(self, task_id: str) -> Task | None:
        """Take the task with ``task_id`` out of the queue, if present."""
        for index, entry in enumerate(self._heap):
            if entry[2].id == task_id:
                self._heap.pop(index)
                heapq.heapify(self._heap)
                return entry[2]
        return None

    def snapshot(self) -> list[Task]:
        """Queued tasks in the order they would be popped."""
        return [entry[2] for entry in sorted(self._heap)]

    def __len__(self) -> int:
        return len(self._heap)

    def __bool__(self) -> bool:
        return bool(self._heap)

    def __iter__(self) -> Iterator[Task]:
        return iter(self.snapshot())

    def __contains__(self, task_id: object) -> bool:
        return any(entry[2].id == task_id for entry in self._heap)
