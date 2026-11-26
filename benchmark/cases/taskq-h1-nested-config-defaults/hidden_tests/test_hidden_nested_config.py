import copy
from datetime import datetime, timedelta, timezone

from taskq.api import TaskService
from taskq.clock import FixedClock
from taskq.config import DEFAULTS, load_config
from taskq.retry import RetryPolicy
from taskq.task import TaskStatus
from taskq.worker import Worker

START = datetime(2024, 3, 1, 12, 0, tzinfo=timezone.utc)
PRISTINE = copy.deepcopy(DEFAULTS)


def test_partial_retry_section_keeps_other_defaults():
    config = load_config({"retry": {"max_attempts": 3}})
    assert config["retry"] == {**PRISTINE["retry"], "max_attempts": 3}
    assert config["worker"] == PRISTINE["worker"]


def test_partial_worker_section():
    config = load_config({"worker": {"batch_size": 50}})
    assert config["worker"] == {**PRISTINE["worker"], "batch_size": 50}


def test_defaults_are_not_mutated():
    load_config({"retry": {"max_attempts": 3}, "worker": {"name": "w2"}})
    assert DEFAULTS == PRISTINE
    assert load_config()["retry"]["max_attempts"] == PRISTINE["retry"]["max_attempts"]
    assert load_config()["worker"]["name"] == PRISTINE["worker"]["name"]


def test_returned_config_is_independent():
    first = load_config({"retry": {"factor": 3.0}})
    first["retry"]["base_delay"] = 99.0
    assert load_config({"retry": {"factor": 3.0}})["retry"]["base_delay"] == PRISTINE["retry"]["base_delay"]


def test_toml_file_with_partial_sections(tmp_path):
    path = tmp_path / "taskq.toml"
    path.write_text("[retry]\nmax_attempts = 3\n\n[worker]\nbatch_size = 2\n", encoding="utf-8")
    worker = Worker.from_config(load_config(path), {"noop": lambda p: None})
    assert worker.retry == RetryPolicy(max_attempts=3)
    assert worker.batch_size == 2
    assert worker.name == PRISTINE["worker"]["name"]


def test_end_to_end_retries_use_merged_policy(tmp_path):
    path = tmp_path / "taskq.json"
    path.write_text('{"retry": {"max_attempts": 2, "base_delay": 10}}', encoding="utf-8")
    config = load_config(path)
    clock = FixedClock(START)
    service = TaskService.from_config(config, clock=clock)
    calls = []

    def always_fails(payload):
        calls.append(clock())
        raise RuntimeError("smtp down")

    worker = Worker.from_config(config, {"send_email": always_fails}, queue=service.queue, clock=clock)
    task_id = service.submit("send_email", {"to": "ada@example.com"})
    worker.run_pending()
    assert service.get(task_id).next_run == START + timedelta(seconds=10)
    clock.advance(seconds=10)
    worker.run_pending()
    assert service.status(task_id) is TaskStatus.FAILED
    assert len(calls) == 2
