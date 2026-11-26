from datetime import datetime, timezone

from taskq.api import TaskService
from taskq.clock import FixedClock
from taskq.queue import TaskQueue
from taskq.task import Task
from taskq.worker import Worker

START = datetime(2024, 3, 1, 12, 0, tzinfo=timezone.utc)


def test_equal_priority_pops_in_push_order():
    q = TaskQueue()
    ids = [f"task-{n:02d}" for n in range(20, 0, -1)]
    for task_id in ids:
        q.push(Task("job", id=task_id))
    assert [q.pop().id for _ in ids] == ids


def test_fifo_within_each_priority_level():
    q = TaskQueue()
    pushed = [("z", 1), ("y", 5), ("x", 1), ("w", 5), ("v", 1)]
    for task_id, priority in pushed:
        q.push(Task("job", id=task_id, priority=priority))
    assert [t.id for t in q.snapshot()] == ["y", "w", "z", "x", "v"]
    assert [q.pop().id for _ in pushed] == ["y", "w", "z", "x", "v"]


def test_interleaved_push_and_pop():
    q = TaskQueue()
    q.push(Task("job", id="c"))
    q.push(Task("job", id="b"))
    assert q.pop().id == "c"
    q.push(Task("job", id="a"))
    assert [q.pop().id, q.pop().id] == ["b", "a"]


def test_requeued_task_goes_behind_waiting_tasks():
    q = TaskQueue()
    q.push(Task("job", id="a-first"))
    q.push(Task("job", id="z-second"))
    q.push(q.pop())
    assert [t.id for t in q] == ["z-second", "a-first"]


def test_worker_runs_chunks_in_submission_order():
    service = TaskService(clock=FixedClock(START))
    for chunk in range(1, 7):
        service.submit("import_chunk", {"chunk": chunk}, task_id=f"{9 - chunk}-chunk")
    seen = []
    worker = Worker(service.queue, {"import_chunk": lambda p: seen.append(p["chunk"])}, clock=FixedClock(START))
    worker.run_pending()
    assert seen == [1, 2, 3, 4, 5, 6]


def test_remove_keeps_order_of_the_rest():
    q = TaskQueue()
    for task_id in ["d", "c", "b", "a"]:
        q.push(Task("job", id=task_id))
    q.remove("c")
    assert [q.pop().id for _ in range(3)] == ["d", "b", "a"]
