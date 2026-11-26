import json
from datetime import datetime, timezone

import pytest

from taskq.storage import JsonStore, StorageError
from taskq.task import Task, TaskStatus


def test_missing_file_loads_nothing(tmp_path):
    assert JsonStore(tmp_path / "none.json").load() == []


def test_save_and_load(tmp_path):
    store = JsonStore(tmp_path / "tasks.json")
    created = datetime(2024, 2, 1, 7, 30, tzinfo=timezone.utc)
    tasks = [
        Task("a", {"n": 1}, priority=2, tags=["x"], created_at=created),
        Task("b", status=TaskStatus.DONE, attempts=1, created_at=created),
    ]
    store.save(tasks)
    assert store.load() == tasks


def test_saved_file_is_versioned(tmp_path):
    store = JsonStore(tmp_path / "tasks.json")
    store.save([Task("a")])
    document = json.loads(store.path.read_text(encoding="utf-8"))
    assert document["version"] == 1
    assert document["tasks"][0]["name"] == "a"


def test_bare_list_format_still_loads(tmp_path):
    path = tmp_path / "old.json"
    path.write_text(json.dumps([{"id": "t1", "name": "a", "tags": ["legacy"]}]), encoding="utf-8")
    [task] = JsonStore(path).load()
    assert task.id == "t1"
    assert task.tags == ["legacy"]


def test_corrupt_file(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(StorageError):
        JsonStore(path).load()


def test_clear(tmp_path):
    store = JsonStore(tmp_path / "tasks.json")
    store.save([])
    assert store.exists()
    store.clear()
    assert not store.exists()
