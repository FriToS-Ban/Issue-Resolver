# taskq

A small in-process task queue with priorities, retries with exponential backoff,
interval schedules and JSON persistence. No broker, no threads: you call
`Worker.run_pending()` from your own loop, cron job or management command.

```python
from taskq.api import TaskService
from taskq.config import load_config
from taskq.worker import Worker

config = load_config("taskq.toml")
service = TaskService.from_config(config)
task_id = service.submit("send_email", {"to": "ada@example.com"}, priority=5)

worker = Worker.from_config(config, handlers={"send_email": send_email}, queue=service.queue)
worker.run_pending()
print(service.status(task_id))
```

## Priorities

Higher `priority` values run first. Tasks with the same priority run in the order
they were submitted.

## Retries

A handler that raises is retried with exponential backoff:
`base_delay * factor ** (attempt - 1)` seconds, never more than `max_delay`.
Raise `taskq.retry.PermanentError` to fail a task immediately.

## Schedules

Intervals are written as `"90s"`, `"5m"`, `"2h"`, `"1d"` or combinations such as
`"1h30m"`. A bare number is a number of seconds.

## Configuration

`load_config()` accepts a path to a TOML or JSON file, a dict, or nothing. Every
setting has a default; anything you specify overrides just that setting.

```toml
[queue]
max_size = 10000

[retry]
max_attempts = 5
base_delay = 1.0
factor = 2.0
max_delay = 300.0

[worker]
name = "worker"
batch_size = 10
poll_interval = "5s"

[storage]
path = "tasks.json"
```

## Persistence

`TaskService.save()` writes all tasks to the JSON file configured under
`[storage]`; `TaskService.restore()` loads them back and re-queues pending ones.
