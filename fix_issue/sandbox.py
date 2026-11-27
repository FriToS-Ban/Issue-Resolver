"""
sandbox.py — Execution hygiene for commands that run target-repository code.

Running a repository's tests or linters executes code we did not write. We cannot
fully sandbox that without containers, but we can (and do) limit the blast radius:

  * the child environment is built from an ALLOWLIST (``safe_env``): only what test runners,
    linters and git need (executable lookup, OS/system dirs, temp dirs, home and per-user
    profile dirs, locale, TLS trust-store paths and a vetted set of Python / Node / Go /
    Rust / Java variables). Everything else — GitHub and LLM tokens, database URLs, webhooks,
    agent sockets, CI job tokens, registry credentials — is dropped. ``HOME`` is NOT redirected,
    so files under it (``~/.npmrc``, ``~/.fix-issue.toml``, ...) stay readable: only a container
    (``sandbox_prefix``) isolates those,
  * commands are argument lists (never ``shell=True``), and a ``.bat``/``.cmd`` program is
    refused when an argument contains anything cmd.exe could interpret (``_batch_refusal``),
  * every command has a wall-clock timeout; the child runs in its own process group/session
    (POSIX) or Job object (Windows, falling back to ``taskkill /T /F``), and the WHOLE process
    tree is killed when the timeout expires and again when the command finishes, so no
    grandchild outlives it,
  * stdout/stderr go to temporary files, not pipes, so a grandchild that inherited the
    stream cannot keep the runner waiting after the kill,
  * commands run inside the throwaway clone only,
  * "project scripts" (``npm run``, ``make``, ``tox``) that are arbitrary by definition are
    gated behind an explicit opt-in (see validators.py / config ``allow_project_scripts``).

Extending the environment: ``safe_env(extra={...})`` adds variables for one call, and the
``FIX_ISSUE_ENV_PASSTHROUGH`` environment variable (comma/space separated NAMES, e.g.
``DATABASE_URL,JAVA_OPTS``) passes those variables from the parent environment through to
every child — use it only for values the target repository is allowed to see.

An optional ``sandbox_prefix`` (e.g. a ``docker run ...`` invocation) can wrap commands
for real isolation; when it is not configured that limitation is recorded in the run.
"""

from __future__ import annotations

import contextlib
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

_ENV_ALLOWLIST = frozenset({
    # executable lookup and the OS itself
    "PATH", "PATHEXT", "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "COMSPEC", "OS",
    "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE", "PROCESSOR_ARCHITEW6432",
    "PROCESSOR_IDENTIFIER", "PROCESSOR_LEVEL", "PROCESSOR_REVISION",
    "PROGRAMFILES", "PROGRAMFILES(X86)", "PROGRAMW6432", "COMMONPROGRAMFILES",
    "COMMONPROGRAMFILES(X86)", "COMMONPROGRAMW6432", "PROGRAMDATA", "ALLUSERSPROFILE", "PUBLIC",
    # temp, home and per-user directories (HOME is deliberately not redirected)
    "TEMP", "TMP", "TMPDIR", "HOME", "USERPROFILE", "HOMEDRIVE", "HOMEPATH", "APPDATA", "LOCALAPPDATA",
    "XDG_CACHE_HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME",
    "USER", "USERNAME", "LOGNAME", "SHELL",
    # locale / terminal (LC_* is allowed by prefix)
    "LANG", "LANGUAGE", "TZ", "TERM", "NO_COLOR",
    # TLS trust stores: paths, not secrets
    "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "NODE_EXTRA_CA_CERTS",
    # Python
    "VIRTUAL_ENV", "CONDA_PREFIX", "CONDA_DEFAULT_ENV", "PYTHONPATH", "PYTHONHOME", "PYTHONUTF8",
    "PYTHONIOENCODING", "PYTHONHASHSEED", "PYTHONWARNINGS", "PYTHONNOUSERSITE", "PYTHONUSERBASE",
    "PYTHONPYCACHEPREFIX", "PYTHONDEVMODE", "PYTHONFAULTHANDLER", "PYTHONSAFEPATH",
    # Node
    "NODE_PATH", "NODE_ENV", "NODE_OPTIONS", "NVM_DIR", "NVM_BIN", "NVM_HOME", "NVM_SYMLINK",
    # Go / Rust / Java
    "GOPATH", "GOROOT", "GOCACHE", "GOMODCACHE", "GOFLAGS", "GOOS", "GOARCH", "CGO_ENABLED",
    "GOPROXY", "GOPRIVATE", "GONOSUMDB", "GONOPROXY", "GOTOOLCHAIN",
    # Shared-library lookup: a --enable-shared Python without rpath cannot start without it.
    "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH",
    "CARGO_HOME", "RUSTUP_HOME", "JAVA_HOME",
})
_ENV_ALLOW_PREFIXES = ("LC_",)
PASSTHROUGH_VAR = "FIX_ISSUE_ENV_PASSTHROUGH"


def _env_key(name: str) -> str:
    # Windows environment names are case-insensitive (os.environ upper-cases them); POSIX are not.
    return name.upper() if os.name == "nt" else name


def _passthrough_names() -> set[str]:
    raw = os.environ.get(PASSTHROUGH_VAR, "")
    return {_env_key(n) for n in re.split(r"[\s,;]+", raw) if n}


def env_allowed(name: str) -> bool:
    """True when ``name`` is on the sandbox allowlist (or the user's passthrough list)."""
    key = _env_key(name)
    return key in _ENV_ALLOWLIST or key.startswith(_ENV_ALLOW_PREFIXES) or key in _passthrough_names()


def safe_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """The child environment: allowlisted variables from ours, plus fixed hygiene settings."""
    env = {k: v for k, v in os.environ.items() if env_allowed(k)}
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


# ---------------------------------------------------------------------------
# .bat / .cmd guard (cmd.exe re-parses the command line: BatBadBut-style injection)
# ---------------------------------------------------------------------------

_BATCH_SUFFIXES = (".bat", ".cmd")
# Plain ASCII only: no quotes, %, !, ^, &, |, <, >, (, ), newlines, and no non-ASCII characters
# that a best-fit code-page conversion could turn into one of those.
_BATCH_SAFE_ARG = re.compile(r"[A-Za-z0-9 _\-+=.,:/\\@~#\[\]{}'*?]*\Z")


def is_batch_file(program: str) -> bool:
    return str(program).lower().endswith(_BATCH_SUFFIXES)


def _batch_refusal(argv: list[str], windows: bool | None = None) -> str | None:
    """Why ``argv`` must not run (a batch file with an unsafe argument), or None."""
    windows = os.name == "nt" if windows is None else windows
    if not windows or not argv or not is_batch_file(argv[0]):
        return None
    for arg in argv[1:]:
        if not _BATCH_SAFE_ARG.match(arg):
            return (f"Refused to run {Path(argv[0]).name}: it is a .bat/.cmd file, cmd.exe would interpret "
                    f"characters in the argument {arg!r}")
    return None


# ---------------------------------------------------------------------------
# Process trees
# ---------------------------------------------------------------------------

if os.name == "nt":  # pragma: no cover - exercised on Windows only
    import ctypes
    from ctypes import wintypes

    class _IoCounters(ctypes.Structure):
        _fields_ = [(n, ctypes.c_ulonglong) for n in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class _BasicLimits(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class _ExtendedLimits(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _BasicLimits),
            ("IoInfo", _IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    _JOB_EXTENDED_LIMIT_INFORMATION = 9
    _JOB_KILL_ON_JOB_CLOSE = 0x2000
    _JOB_DIE_ON_UNHANDLED_EXCEPTION = 0x0400
    _CREATE_SUSPENDED = 0x00000004

    def _load_win_api() -> Any:
        try:
            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            ntdll = ctypes.WinDLL("ntdll")
            api = {
                "CreateJobObjectW": (k32.CreateJobObjectW, [wintypes.LPVOID, wintypes.LPCWSTR], wintypes.HANDLE),
                "SetInformationJobObject": (k32.SetInformationJobObject,
                                            [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD],
                                            wintypes.BOOL),
                "AssignProcessToJobObject": (k32.AssignProcessToJobObject, [wintypes.HANDLE, wintypes.HANDLE],
                                             wintypes.BOOL),
                "TerminateJobObject": (k32.TerminateJobObject, [wintypes.HANDLE, wintypes.UINT], wintypes.BOOL),
                "TerminateProcess": (k32.TerminateProcess, [wintypes.HANDLE, wintypes.UINT], wintypes.BOOL),
                "CloseHandle": (k32.CloseHandle, [wintypes.HANDLE], wintypes.BOOL),
                "NtResumeProcess": (ntdll.NtResumeProcess, [wintypes.HANDLE], ctypes.c_long),
            }
        except (OSError, AttributeError):
            return None
        out = {}
        for name, (fn, argtypes, restype) in api.items():
            fn.argtypes, fn.restype = argtypes, restype
            out[name] = fn
        return types.SimpleNamespace(**out)

    _WIN_API = _load_win_api()

    class _WinJob:
        """A kill-on-close Job object: holds the child and every process it starts."""

        def __init__(self, handle: int) -> None:
            self.handle = handle

        @classmethod
        def create(cls) -> "_WinJob | None":
            api = _WIN_API
            if api is None:
                return None
            handle = api.CreateJobObjectW(None, None)
            if not handle:
                return None
            info = _ExtendedLimits()
            info.BasicLimitInformation.LimitFlags = _JOB_KILL_ON_JOB_CLOSE | _JOB_DIE_ON_UNHANDLED_EXCEPTION
            if not api.SetInformationJobObject(handle, _JOB_EXTENDED_LIMIT_INFORMATION,
                                               ctypes.byref(info), ctypes.sizeof(info)):
                api.CloseHandle(handle)
                return None
            return cls(handle)

        def adopt(self, proc: subprocess.Popen) -> bool:
            """Put the (suspended) child in the job, then resume it. False: it runs outside the job."""
            api = _WIN_API
            hproc = int(getattr(proc, "_handle", 0) or 0)
            assigned = bool(hproc) and bool(api.AssignProcessToJobObject(self.handle, hproc))
            status = api.NtResumeProcess(hproc) if hproc else -1
            if status != 0:
                if hproc:
                    api.TerminateProcess(hproc, 1)
                raise OSError(f"could not resume the child process (status {status})")
            return assigned

        def kill(self) -> None:
            _WIN_API.TerminateJobObject(self.handle, 1)

        def close(self) -> None:
            if self.handle:
                _WIN_API.CloseHandle(self.handle)   # kill-on-close: anything still in the job dies
                self.handle = 0


def _taskkill(pid: int) -> None:  # pragma: no cover - Windows fallback only
    exe = os.path.join(os.environ.get("SYSTEMROOT", r"C:\Windows"), "System32", "taskkill.exe")
    with contextlib.suppress(OSError, subprocess.SubprocessError):
        subprocess.run([exe, "/T", "/F", "/PID", str(pid)], stdin=subprocess.DEVNULL,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)


class _ProcessTree:
    """A started child plus the means to kill it and all of its descendants."""

    def __init__(self, argv: list[str], cwd: Path | str, env: dict[str, str] | None,
                 stdout: IO[bytes], stderr: IO[bytes]) -> None:
        self.job = None
        kwargs: dict[str, Any] = dict(cwd=str(cwd), env=env, stdin=subprocess.DEVNULL,
                                      stdout=stdout, stderr=stderr)
        if os.name == "nt":  # pragma: no cover - platform specific
            job = _WinJob.create()
            flags = subprocess.CREATE_NEW_PROCESS_GROUP | (_CREATE_SUSPENDED if job else 0)
            try:
                self.proc = subprocess.Popen(argv, creationflags=flags, **kwargs)
            except BaseException:
                if job:
                    job.close()
                raise
            if job is not None:
                try:
                    in_job = job.adopt(self.proc)
                except BaseException:
                    job.close()
                    raise
                if in_job:
                    self.job = job
                else:
                    job.close()
        else:
            self.proc = subprocess.Popen(argv, start_new_session=True, **kwargs)

    def kill_tree(self) -> None:
        """Kill the child and every descendant still in its group/job, then reap the child."""
        if os.name == "nt":  # pragma: no cover - platform specific
            if self.job is not None:
                self.job.kill()
            elif self.proc.poll() is None:
                _taskkill(self.proc.pid)
                with contextlib.suppress(OSError):
                    self.proc.kill()
        else:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(self.proc.pid, signal.SIGKILL)
        with contextlib.suppress(subprocess.TimeoutExpired):
            self.proc.wait(timeout=10)
        if self.job is not None:
            self.job.close()
            self.job = None


def _read_back(f: IO[bytes], max_chars: int | None) -> str:
    f.flush()
    size = f.seek(0, os.SEEK_END)
    limit = size if max_chars is None else min(size, max_chars * 4 + 4)
    f.seek(size - limit)
    text = f.read(limit).decode("utf-8", "replace")
    text = text.replace("\r\n", "\n").replace("\r", "\n")   # same as text-mode universal newlines
    return text if max_chars is None else text[-max_chars:]


@dataclass
class ProcessResult:
    returncode: int | None       # None when the command timed out
    stdout: str
    stderr: str
    duration_seconds: float
    timed_out: bool = False


def run_process(argv: list[str], cwd: Path | str, *, timeout: float, env: dict[str, str] | None = None,
                max_chars: int | None = None) -> ProcessResult:
    """
    Run ``argv`` (no shell) with stdin closed and stdout/stderr captured in temporary files.

    ``timeout`` is wall-clock from the start. On timeout — and also after a normal exit — the
    whole process tree is killed (POSIX: the child's session; Windows: its Job object, else
    ``taskkill /T /F``). ``env=None`` inherits OUR environment: callers running repository
    code pass ``safe_env()``. Raises ``FileNotFoundError``/``OSError`` like ``subprocess.Popen``.
    """
    start = time.monotonic()
    with tempfile.TemporaryFile() as out_f, tempfile.TemporaryFile() as err_f:
        tree = _ProcessTree(argv, cwd, env, out_f, err_f)
        timed_out = False
        returncode: int | None = None
        try:
            returncode = tree.proc.wait(timeout=max(0.0, start + timeout - time.monotonic()))
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            tree.kill_tree()
        duration = time.monotonic() - start
        return ProcessResult(None if timed_out else returncode, _read_back(out_f, max_chars),
                             _read_back(err_f, max_chars), duration, timed_out)


# ---------------------------------------------------------------------------
# Commands that run repository code
# ---------------------------------------------------------------------------

@dataclass
class CommandResult:
    command: list[str]
    returncode: int | None
    output: str
    duration_seconds: float
    timed_out: bool = False
    not_found: bool = False
    refused: bool = False        # never started: a .bat/.cmd with arguments cmd.exe would interpret

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
    """Run an argument-list command with the allowlist env, a wall-clock timeout and tree kill."""
    full = [*(prefix or []), *command]
    refusal = _batch_refusal(full)
    if refusal:
        return CommandResult(full, None, refusal, 0.0, refused=True)
    start = time.monotonic()
    try:
        res = run_process(full, cwd, timeout=timeout, env=safe_env(), max_chars=max_output)
    except FileNotFoundError:
        return CommandResult(full, None, f"Command not found: {full[0]}", time.monotonic() - start,
                             not_found=True)
    out = "\n".join(p for p in (res.stdout, res.stderr) if p and p.strip())
    if res.timed_out:
        return CommandResult(full, None, f"Timed out after {timeout}s.\n{out}"[-max_output:],
                             res.duration_seconds, timed_out=True)
    return CommandResult(full, res.returncode, out[-max_output:], res.duration_seconds)
