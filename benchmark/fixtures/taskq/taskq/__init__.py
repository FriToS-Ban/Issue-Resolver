"""taskq: a small in-process task queue."""

from taskq.queue import TaskQueue
from taskq.task import Task, TaskStatus

__all__ = ["Task", "TaskQueue", "TaskStatus"]
__version__ = "0.6.0"
