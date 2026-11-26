"""
sandbox.py — Execution hygiene for commands that run target-repository code.

Running a repository's tests or linters executes code we did not write. We cannot
fully sandbox that without containers, but we can (and do) limit the blast radius:

  * secrets are stripped from the child environment (GitHub token, LLM keys, cloud creds),
  * commands are argument lists (never ``shell=True``),
  * every command has a hard timeout,
  * commands run inside the throwaway clone only,
  * "project scripts" (``npm run``, ``make``, ``tox``) that are arbitrary by definition are
    gated behind an explicit opt-in (see validators.py / config ``allow_project_scripts``).

An optional ``sandbox_prefix`` (e.g. a ``docker run ...`` invocation) can wrap commands
for real isolation; when it is not configured that limitation is recorded in the run.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

_SECRET_NAME = re.compile(
    r"(TOKEN|SECRET|PASSWORD|PASSWD|API_KEY|APIKEY|ACCESS_KEY|PRIVATE_KEY|CREDENTIAL|_KEY$|^AWS_|^AZURE_|^GCP_|^GOOGLE_)",
    re.IGNORECASE,
)
# Variables that look secret-ish by name but are needed for tools to run.
_ALLOWLIST = {"PATH", "PATHEXT", "SYSTEMROOT", "COMSPEC", "TEMP", "TMP", "HOME", "USERPROFILE",
              "APPDATA", "LOCALAPPDATA", "PROGRAMDATA", "WINDIR", "LANG", "LC_ALL", "TERM",
              "VIRTUAL_ENV", "PYTHONPATH", "NODE_PATH", "GOPATH", "GOROOT", "GOCACHE",
              "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE", "OS", "SHELL", "USER", "USERNAME"}


def safe_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """Return a copy of the environment with secret-looking variables removed."""
    env = {
        k: v for k, v in os.environ.items()
        if k.upper() in _ALLOWLIST or not _SECRET_NAME.search(k)
    }
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env["CI"] = "1"
    env["GIT_TERMINAL_PROMPT"] = "0"
    if extra:
        env.update(extra)
    return env


def python_executable() -> str:
    """The interpreter running fix-issue (has pytest etc.), not whatever `python` is on PATH."""
    return sys.executable or "python"


@dataclass
class CommandResult:
    command: list[str]
    returncode: int | None
    output: str
    duration_seconds: float
    timed_out: bool = False
    not_found: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out


def run_command(
    command: list[str],
    cwd: Path,
    timeout: int = 300,
    prefix: list[str] | None = None,
    max_output: int = 12000,
) -> CommandResult:
    """Run an argument-list command with a sanitized env and a timeout."""
    full = [*(prefix or []), *command]
    start = time.monotonic()
    try:
        proc = subprocess.run(
            full, cwd=str(cwd), capture_output=True, text=True, timeout=timeout,
            env=safe_env(), encoding="utf-8", errors="replace",
        )
        out = "\n".join(p for p in (proc.stdout, proc.stderr) if p and p.strip())
        return CommandResult(full, proc.returncode, out[-max_output:], time.monotonic() - start)
    except subprocess.TimeoutExpired as exc:
        partial = f"{exc.stdout or ''}\n{exc.stderr or ''}"
        if isinstance(partial, bytes):  # pragma: no cover
            partial = partial.decode("utf-8", "replace")
        return CommandResult(full, None, f"Timed out after {timeout}s.\n{partial}"[-max_output:],
                             time.monotonic() - start, timed_out=True)
    except FileNotFoundError:
        return CommandResult(full, None, f"Command not found: {full[0]}", time.monotonic() - start,
                             not_found=True)
