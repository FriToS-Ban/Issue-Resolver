from datetime import datetime, timezone

import pytest

from taskq.api import TaskService, UnknownTask
from taskq.clock import FixedClock
from taskq.config import load_config
from taskq.storage import JsonStore
from taskq.task import TaskStatus
from taskq.worker import Worker

START = datetime(2024, 3, 1, 12, 0, tzinfo=timezone.utc)


def test_submit_and_status():
    service = TaskService(clock=FixedClock(START))
    task_id = service.submit("email", {"to": "a@example.com"}, priority=2, tags=["mail"])
    assert service.status(task_id) is TaskStatus.PENDING
    assert service.get(task_id).created_at == START
    assert len(service.queue) == 1


def test_unknown_task():
    with pytest.raises(UnknownTask):
        TaskService().status("nope")


def test_custom_task_id():
    service = TaskService()
    assert service.submit("x", task_id="job-1") == "job-1"
    with pytest.raises(ValueError):
        service.submit("x", task_id="job-1")


def test_cancel():
    service = TaskService()
    task_id = service.submit("x")
    assert service.cancel(task_id)
    assert service.status(task_id) is TaskStatus.CANCELLED
    assert len(service.queue) == 0
    assert not service.cancel(task_id)


def test_filter_tasks():
    service = TaskService()
    a = service.submit("x", tags=["red"])
    service.submit("y", tags=["blue"])
    assert [t.id for t in service.tasks(tag="red")] == [a]
    assert len(service.tasks(status=TaskStatus.PENDING)) == 2


def test_save_and_restore(tmp_path):
    store = JsonStore(tmp_path / "tasks.json")
    service = TaskService(store=store, clock=FixedClock(START))
    pending = service.submit("x", priority=1)
    done = service.submit("y")
    service.get(done).status = TaskStatus.DONE
    service.save()

    restored = TaskService(store=JsonStore(tmp_path / "tasks.json"))
    assert restored.restore() == 2
    assert restored.status(done) is TaskStatus.DONE
    assert [t.id for t in restored.queue] == [pending]


def test_service_and_worker_share_queue():
    clock = FixedClock(START)
    config = load_config()
    service = TaskService.from_config(config, clock=clock)
    calls = []
    worker = Worker.from_config(config, {"x": calls.append}, queue=service.queue, clock=clock)
    task_id = service.submit("x", {"n": 1})
    worker.run_pending()
    assert calls == [{"n": 1}]
    assert service.status(task_id) is TaskStatus.DONE


def test_save_without_store():
    with pytest.raises(RuntimeError):
        TaskService().save()
