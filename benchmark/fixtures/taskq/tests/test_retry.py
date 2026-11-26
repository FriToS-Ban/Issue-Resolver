import pytest

from taskq.retry import PermanentError, RetryPolicy, backoff_delay, backoff_delays, should_retry
from taskq.task import Task


def test_default_schedule():
    assert backoff_delays(RetryPolicy()) == [1.0, 2.0, 4.0, 8.0]


def test_delay_is_capped():
    policy = RetryPolicy(max_attempts=8, base_delay=1, factor=2, max_delay=10)
    assert backoff_delays(policy) == [1, 2, 4, 8, 10, 10, 10]


def test_attempt_numbers_start_at_one():
    with pytest.raises(ValueError):
        backoff_delay(0, RetryPolicy())


def test_single_attempt_policy_has_no_retries():
    assert backoff_delays(RetryPolicy(max_attempts=1)) == []


def test_should_retry_until_max_attempts():
    policy = RetryPolicy(max_attempts=3)
    task = Task("t", attempts=2)
    assert should_retry(task, RuntimeError(), policy)
    task.attempts = 3
    assert not should_retry(task, RuntimeError(), policy)


def test_permanent_errors_are_not_retried():
    assert not should_retry(Task("t", attempts=1), PermanentError("bad input"), RetryPolicy())


def test_from_config():
    policy = RetryPolicy.from_config({"max_attempts": 2, "base_delay": 0.5, "factor": 3, "max_delay": 9})
    assert policy == RetryPolicy(max_attempts=2, base_delay=0.5, factor=3.0, max_delay=9.0)


def test_invalid_policy():
    with pytest.raises(ValueError):
        RetryPolicy(factor=0.5)
