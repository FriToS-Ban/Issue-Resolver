from datetime import datetime, timedelta, timezone

from taskq.api import TaskService
from taskq.clock import FixedClock
from taskq.retry import RetryPolicy
from taskq.storage import JsonStore
from taskq.task import Task, TaskStatus
from taskq.worker import Worker

START = datetime(2024, 3, 1, 9, 0, tzinfo=timezone.utc)


def test_task_round_trip_keeps_next_run_datetime():
    task = Task("send_report", {"report": 7}, next_run=START, created_at=START - timedelta(hours=1))
    restored = Task.from_dict(task.to_dict())
    assert isinstance(restored.next_run, datetime)
    assert restored.next_run == START
    assert restored.next_run.tzinfo is not None
    assert restored == task


def test_round_trip_without_next_run():
    assert Task.from_dict(Task("x").to_dict()).next_run is None


def test_store_round_trip(tmp_path):
    store = JsonStore(tmp_path / "tasks.json")
    tasks = [Task("a", next_run=START), Task("b", next_run=START + timedelta(minutes=5), attempts=2)]
    store.save(tasks)
    assert store.load() == tasks


def test_restored_task_runs_when_due(tmp_path):
    path = tmp_path / "tasks.json"
    service = TaskService(store=JsonStore(path), clock=FixedClock(START - timedelta(days=1)))
    due = service.submit("send_report", {"report": 7}, run_at=START)
    later = service.submit("send_report", {"report": 8}, run_at=START + timedelta(hours=2))
    service.save()

    fresh = TaskService(store=JsonStore(path))
    fresh.restore()
    seen = []
    clock = FixedClock(START + timedelta(minutes=1))
    worker = Worker(fresh.queue, {"send_report": lambda p: seen.append(p["report"])}, clock=clock)
    worker.run_pending()
    assert seen == [7]
    assert fresh.status(due) is TaskStatus.DONE
    assert fresh.status(later) is TaskStatus.PENDING

    clock.advance(hours=2)
    worker.run_pending()
    assert seen == [7, 8]


def test_retry_survives_restart(tmp_path):
    path = tmp_path / "tasks.json"
    clock = FixedClock(START)
    service = TaskService(store=JsonStore(path), clock=clock)
    task_id = service.submit("flaky")
    calls = []

    def flaky(payload):
        calls.append(clock())
        if len(calls) == 1:
            raise RuntimeError("temporary")

    Worker(service.queue, {"flaky": flaky}, retry=RetryPolicy(base_delay=30), clock=clock).run_pending()
    assert service.get(task_id).next_run == START + timedelta(seconds=30)
    service.save()

    fresh = TaskService(store=JsonStore(path))
    fresh.restore()
    worker = Worker(fresh.queue, {"flaky": flaky}, retry=RetryPolicy(base_delay=30), clock=clock)
    clock.advance(seconds=10)
    assert worker.run_pending() == []
    clock.advance(seconds=20)
    worker.run_pending()
    assert fresh.status(task_id) is TaskStatus.DONE
    assert len(calls) == 2
