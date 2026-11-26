from datetime import datetime, timedelta, timezone

import pytest

from taskq.clock import FixedClock
from taskq.queue import TaskQueue
from taskq.retry import RetryPolicy, backoff_delay, backoff_delays
from taskq.task import Task
from taskq.worker import Worker


def test_schedule_from_issue():
    policy = RetryPolicy(max_attempts=7, base_delay=5, factor=2, max_delay=30)
    assert backoff_delays(policy) == [5, 10, 20, 30, 30, 30]


@pytest.mark.parametrize(
    "policy",
    [
        RetryPolicy(max_attempts=10, base_delay=15, factor=2, max_delay=120),
        RetryPolicy(max_attempts=6, base_delay=0.5, factor=3, max_delay=4),
        RetryPolicy(max_attempts=12, base_delay=2, factor=1.5, max_delay=60),
    ],
)
def test_no_delay_exceeds_max_delay(policy):
    delays = backoff_delays(policy)
    assert max(delays) <= policy.max_delay
    assert delays[-1] == policy.max_delay


def test_fractional_base_reaches_cap():
    policy = RetryPolicy(max_attempts=6, base_delay=0.5, factor=3, max_delay=4)
    assert backoff_delays(policy) == [0.5, 1.5, 4, 4, 4]


def test_base_delay_larger_than_cap():
    policy = RetryPolicy(max_attempts=4, base_delay=45, factor=2, max_delay=30)
    assert backoff_delays(policy) == [30, 30, 30]
    assert backoff_delay(1, policy) == 30


def test_worker_reschedules_within_cap():
    start = datetime(2024, 3, 1, 12, 0, tzinfo=timezone.utc)
    clock = FixedClock(start)
    queue = TaskQueue()
    task = Task("webhook", attempts=5)
    queue.push(task)

    def failing(payload):
        raise RuntimeError("503 from receiver")

    policy = RetryPolicy(max_attempts=10, base_delay=15, factor=2, max_delay=120)
    Worker(queue, {"webhook": failing}, retry=policy, clock=clock).run_pending()
    assert task.next_run - start == timedelta(seconds=120)
