from datetime import datetime, timezone

from taskq.task import Task, TaskStatus


def test_defaults():
    task = Task("send_email")
    assert task.status is TaskStatus.PENDING
    assert task.attempts == 0
    assert len(task.id) == 12


def test_round_trip():
    created = datetime(2024, 4, 2, 8, 0, tzinfo=timezone.utc)
    task = Task("resize", {"w": 100}, priority=3, tags=["images"], created_at=created, attempts=2)
    data = task.to_dict()
    assert data["created_at"] == "2024-04-02T08:00:00+00:00"
    assert Task.from_dict(data) == task


def test_from_dict_minimal_record():
    task = Task.from_dict({"id": "abc", "name": "noop", "tags": []})
    assert task.priority == 0
    assert task.payload == {}
    assert task.status is TaskStatus.PENDING


def test_status_is_final():
    assert TaskStatus.DONE.is_final
    assert not TaskStatus.PENDING.is_final


def test_has_tag():
    assert Task("x", tags=["a", "b"]).has_tag("b")
