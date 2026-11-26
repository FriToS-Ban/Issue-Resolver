import pytest

from taskq.queue import QueueEmpty, QueueFull, TaskQueue
from taskq.task import Task


def test_higher_priority_first():
    q = TaskQueue()
    q.push(Task("low", priority=1))
    q.push(Task("high", priority=9))
    q.push(Task("mid", priority=5))
    assert [q.pop().name for _ in range(3)] == ["high", "mid", "low"]


def test_peek_does_not_remove():
    q = TaskQueue()
    q.push(Task("only"))
    assert q.peek().name == "only"
    assert len(q) == 1


def test_pop_empty():
    with pytest.raises(QueueEmpty):
        TaskQueue().pop()


def test_max_size():
    q = TaskQueue(max_size=1)
    q.push(Task("a"))
    with pytest.raises(QueueFull):
        q.push(Task("b"))


def test_remove():
    q = TaskQueue()
    a, b = Task("a", priority=1), Task("b", priority=2)
    q.push(a)
    q.push(b)
    assert q.remove(a.id) is a
    assert q.remove("missing") is None
    assert a.id not in q
    assert b.id in q
    assert len(q) == 1


def test_snapshot_matches_pop_order():
    q = TaskQueue()
    for p in (3, 1, 2):
        q.push(Task(f"p{p}", priority=p))
    assert [t.name for t in q.snapshot()] == ["p3", "p2", "p1"]
    assert bool(q)
