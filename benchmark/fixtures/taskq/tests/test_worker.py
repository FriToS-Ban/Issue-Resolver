from datetime import datetime, timedelta, timezone

from taskq.clock import FixedClock
from taskq.config import load_config
from taskq.queue import TaskQueue
from taskq.retry import PermanentError, RetryPolicy
from taskq.task import Task, TaskStatus
from taskq.worker import Worker

START = datetime(2024, 3, 1, 12, 0, tzinfo=timezone.utc)


def test_runs_successful_task():
    seen = []
    q = TaskQueue()
    task = Task("echo", {"msg": "hi"})
    q.push(task)
    worker = Worker(q, {"echo": lambda p: seen.append(p["msg"])}, clock=FixedClock(START))
    assert worker.run_pending() == [task]
    assert seen == ["hi"]
    assert task.status is TaskStatus.DONE
    assert task.attempts == 1


def test_failed_task_is_rescheduled_with_backoff():
    clock = FixedClock(START)
    q = TaskQueue()
    task = Task("flaky")
    q.push(task)

    def flaky(payload):
        raise RuntimeError("boom")

    worker = Worker(q, {"flaky": flaky}, retry=RetryPolicy(max_attempts=3), clock=clock)
    worker.run_pending()
    assert task.status is TaskStatus.PENDING
    assert task.next_run == START + timedelta(seconds=1)
    assert task.last_error == "RuntimeError: boom"
    assert worker.run_pending() == []
    clock.advance(seconds=1)
    worker.run_pending()
    assert task.attempts == 2
    assert task.next_run == START + timedelta(seconds=1) + timedelta(seconds=2)


def test_task_fails_after_max_attempts():
    clock = FixedClock(START)
    q = TaskQueue()
    task = Task("bad")
    q.push(task)
    worker = Worker(q, {"bad": lambda p: 1 / 0}, retry=RetryPolicy(max_attempts=2), clock=clock)
    worker.run_pending()
    clock.advance(minutes=5)
    worker.run_pending()
    assert task.status is TaskStatus.FAILED
    assert len(worker.events.of_kind("failed")) == 1


def test_permanent_error_fails_immediately():
    q = TaskQueue()
    task = Task("strict")
    q.push(task)

    def strict(payload):
        raise PermanentError("invalid payload")

    Worker(q, {"strict": strict}, clock=FixedClock(START)).run_pending()
    assert task.status is TaskStatus.FAILED


def test_unknown_handler_fails_task():
    q = TaskQueue()
    task = Task("mystery")
    q.push(task)
    Worker(q, {}, clock=FixedClock(START)).run_pending()
    assert task.status is TaskStatus.FAILED
    assert "no handler" in task.last_error


def test_batch_size_limits_work():
    q = TaskQueue()
    for i in range(5):
        q.push(Task("noop", priority=i))
    worker = Worker(q, {"noop": lambda p: None}, batch_size=2, clock=FixedClock(START))
    assert len(worker.run_pending()) == 2
    assert len(q) == 3


def test_from_default_config():
    worker = Worker.from_config(load_config(), {"noop": lambda p: None})
    assert worker.retry == RetryPolicy()
    assert worker.batch_size == 10
    assert worker.queue.max_size == 10_000


def test_from_config_with_overrides():
    config = load_config({"queue": {"max_size": 3}})
    worker = Worker.from_config(config, {})
    assert worker.queue.max_size == 3


def test_run_until_empty():
    q = TaskQueue()
    for _ in range(3):
        q.push(Task("noop"))
    worker = Worker(q, {"noop": lambda p: None}, batch_size=1, clock=FixedClock(START))
    assert worker.run_until_empty() == 3
