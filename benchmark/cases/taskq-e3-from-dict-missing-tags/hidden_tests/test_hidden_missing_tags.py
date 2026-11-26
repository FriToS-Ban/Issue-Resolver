import json

from taskq.api import TaskService
from taskq.storage import JsonStore
from taskq.task import Task, TaskStatus

LEGACY = {"id": "a1f3c9e2b7d4", "name": "send_digest", "payload": {"user": 42}, "priority": 1, "status": "pending", "attempts": 0}


def test_from_dict_without_tags():
    task = Task.from_dict(LEGACY)
    assert task.tags == []
    assert task.name == "send_digest"
    assert task.priority == 1


def test_from_dict_with_null_tags():
    assert Task.from_dict({**LEGACY, "tags": None}).tags == []


def test_from_dict_keeps_existing_tags():
    assert Task.from_dict({**LEGACY, "tags": ["mail", "daily"]}).tags == ["mail", "daily"]


def test_tags_are_not_shared_between_loaded_tasks():
    first = Task.from_dict(LEGACY)
    second = Task.from_dict({**LEGACY, "id": "other"})
    first.tags.append("x")
    assert second.tags == []


def test_restore_legacy_file(tmp_path):
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps([LEGACY]), encoding="utf-8")
    service = TaskService(store=JsonStore(path))
    assert service.restore() == 1
    assert service.status("a1f3c9e2b7d4") is TaskStatus.PENDING
    assert [t.id for t in service.queue] == ["a1f3c9e2b7d4"]
