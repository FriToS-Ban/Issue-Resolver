"""Retry policy and exponential backoff."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from taskq.task import Task


class PermanentError(Exception):
    """Raise from a handler to fail the task without further retries."""


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 5
    base_delay: float = 1.0
    factor: float = 2.0
    max_delay: float = 300.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.base_delay < 0 or self.max_delay < 0:
            raise ValueError("delays must not be negative")
        if self.factor < 1:
            raise ValueError("factor must be at least 1")

    @classmethod
    def from_config(cls, section: Mapping[str, Any]) -> "RetryPolicy":
        """Build a policy from the ``[retry]`` section of a loaded config."""
        return cls(
            max_attempts=int(section["max_attempts"]),
            base_delay=float(section["base_delay"]),
            factor=float(section["factor"]),
            max_delay=float(section["max_delay"]),
        )


def backoff_delay(attempt: int, policy: RetryPolicy) -> float:
    """Seconds to wait after failed attempt number ``attempt`` (1-based)."""
    if attempt < 1:
        raise ValueError("attempt numbers start at 1")
    return min(policy.base_delay * policy.factor ** (attempt - 1), policy.max_delay)


def backoff_delays(policy: RetryPolicy) -> list[float]:
    """The whole delay schedule: one entry per retry the policy allows."""
    return [backoff_delay(attempt, policy) for attempt in range(1, policy.max_attempts)]


def should_retry(task: Task, error: BaseException, policy: RetryPolicy) -> bool:
    """Whether ``task`` should run again after failing with ``error``."""
    if isinstance(error, PermanentError):
        return False
    return task.attempts < policy.max_attempts
