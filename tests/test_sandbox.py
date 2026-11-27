"""
sandbox.py: allowlist environment, wall-clock timeouts that kill the whole process tree, no
stragglers after a normal exit, temp-file capture, and the .bat/.cmd argument guard.
"""

from __future__ import annotations

import os
import sys
import textwrap
import time
from pathlib import Path

import pytest

from fix_issue import sandbox
from fix_issue.sandbox import _batch_refusal, run_command, run_process, safe_env
from fix_issue.test_runner import run_tests

SECRETS = ["GITHUB_TOKEN", "GH_TOKEN", "NVIDIA_API_KEY", "LLM_API_KEY", "SSH_AUTH_SOCK", "DATABASE_URL",
           "MYSQL_PWD", "SLACK_WEBHOOK_URL", "SENTRY_DSN", "DOCKER_AUTH_CONFIG", "GITLAB_PAT", "KUBECONFIG",
           "NPM_CONFIG__AUTH", "GIT_ASKPASS", "GCM_CREDENTIAL_STORE", "PIP_INDEX_URL", "OPENAI_API_BASE",
           "AZURE_OPENAI_KEY", "HF_HUB_TOKEN", "BW_SESSION", "OP_SESSION_MY", "VAULT_ADDR",
           "CLAUDE_CODE_OAUTH_TOKEN", "CI_JOB_JWT", "ACTIONS_ID_TOKEN_REQUEST_URL", "ACTIONS_RUNTIME_TOKEN",
           "GIT_CONFIG_PARAMETERS", "AWS_SECRET_ACCESS_KEY", "HTTPS_PROXY", "FIX_ISSUE_HOME"]


# ---------------------------------------------------------------------------
# Environment allowlist
# ---------------------------------------------------------------------------

def test_safe_env_is_an_allowlist(monkeypatch):
    for name in SECRETS:
        monkeypatch.setenv(name, "FAKEVALUE")
    monkeypatch.setenv("VIRTUAL_ENV", "/venv")
    monkeypatch.setenv("LC_ALL", "C.UTF-8")
    env = safe_env()
    kept = [n for n in SECRETS if n in env or n.lower() in env]
    assert kept == []
    assert "FAKEVALUE" not in env.values()
    assert env["PATH"] == os.environ["PATH"]
    assert env["VIRTUAL_ENV"] == "/venv" and env["LC_ALL"] == "C.UTF-8"
    for name in ("HOME", "USERPROFILE", "SYSTEMROOT", "TEMP", "TMP", "APPDATA"):
        if name in os.environ:
            assert env[name] == os.environ[name], name   # HOME & co. are passed, not redirected
    assert env["PYTHONDONTWRITEBYTECODE"] == "1" and env["CI"] == "1" and env["GIT_TERMINAL_PROMPT"] == "0"


def test_safe_env_extension_hooks(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgres://localhost/test")
    monkeypatch.setenv("GITHUB_TOKEN", "FAKE")
    assert "DATABASE_URL" not in safe_env()
    monkeypatch.setenv(sandbox.PASSTHROUGH_VAR, "DATABASE_URL, SOMETHING_ELSE")
    env = safe_env({"EXTRA_ONE": "1"})
    assert env["DATABASE_URL"] == "postgres://localhost/test" and env["EXTRA_ONE"] == "1"
    assert "GITHUB_TOKEN" not in env


def test_children_only_see_the_allowlist(tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "FAKE_TOKEN_SHOULD_NOT_LEAK")
    monkeypatch.setenv("DATABASE_URL", "postgres://user:pw@db/prod")
    res = run_command([sys.executable, "-c", "import os; print(sorted(os.environ))"], tmp_path, timeout=60)
    assert res.ok, res.output
    assert "GITHUB_TOKEN" not in res.output and "DATABASE_URL" not in res.output


def test_run_tests_runs_real_pytest_with_the_allowlist_env(tmp_path, monkeypatch):
    """The production test command (this interpreter's pytest) works with the reduced environment."""
    monkeypatch.setenv("GITHUB_TOKEN", "FAKE")
    monkeypatch.setenv("DATABASE_URL", "postgres://x")
    proj = tmp_path / "proj"
    (proj / "tests").mkdir(parents=True)
    (proj / "pytest.ini").write_text("[pytest]\n")
    (proj / "calc.py").write_text("def add(a, b):\n    return a + b\n")
    (proj / "tests" / "test_calc.py").write_text(textwrap.dedent("""\
        import os, subprocess, sys
        from calc import add

        def test_add():
            assert add(2, 3) == 5

        def test_no_secrets_and_tmp_path_works(tmp_path):
            assert "GITHUB_TOKEN" not in os.environ and "DATABASE_URL" not in os.environ
            (tmp_path / "f.txt").write_text("ok")

        def test_subprocess_works():
            out = subprocess.run([sys.executable, "-c", "print(40 + 2)"], capture_output=True, text=True)
            assert out.stdout.strip() == "42"
    """))
    cmd = [sys.executable, "-m", "pytest", "--tb=short", "-q", "-p", "no:cacheprovider"]
    res = run_tests(cmd, proj, timeout_seconds=120, extra_args=["--rootdir", str(proj)])
    assert res.passed, res.output
    assert "3 passed" in res.output


# ---------------------------------------------------------------------------
# Timeouts and process trees
# ---------------------------------------------------------------------------

HEARTBEAT = "import sys, time\nwhile True:\n    open(sys.argv[1], 'a').write('.')\n    time.sleep(0.05)\n"


def _spawner(heartbeat_file: Path, then: str) -> str:
    """A child that starts a heartbeat grandchild (inheriting stdout), waits for it, then runs `then`."""
    return textwrap.dedent(f"""\
        import os, subprocess, sys, time
        p = {str(heartbeat_file)!r}
        subprocess.Popen([sys.executable, "-c", {HEARTBEAT!r}, p])
        deadline = time.time() + 20
        while not os.path.exists(p) and time.time() < deadline:
            time.sleep(0.01)
        print("grandchild started", flush=True)
        {then}
    """)


def _assert_heartbeat_stopped(path: Path) -> None:
    assert path.exists(), "grandchild never started"
    time.sleep(0.3)   # let a dying process finish its last write
    size = path.stat().st_size
    time.sleep(0.6)
    assert path.stat().st_size == size, "grandchild is still running"


def test_timeout_is_wall_clock_and_kills_grandchildren(tmp_path):
    beat = tmp_path / "beat.txt"
    start = time.monotonic()
    res = run_command([sys.executable, "-c", _spawner(beat, "time.sleep(30)")], tmp_path, timeout=3)
    elapsed = time.monotonic() - start
    assert res.timed_out and res.returncode is None and not res.ok
    assert elapsed < 10, f"timeout of 3s returned after {elapsed:.1f}s"
    assert "Timed out after 3s" in res.output and "grandchild started" in res.output
    _assert_heartbeat_stopped(beat)


def test_grandchild_holding_stdout_does_not_block_a_normal_exit_and_is_killed(tmp_path):
    beat = tmp_path / "beat.txt"
    start = time.monotonic()
    res = run_command([sys.executable, "-c", _spawner(beat, "sys.exit(0)")], tmp_path, timeout=60)
    assert res.ok and "grandchild started" in res.output
    assert time.monotonic() - start < 15
    _assert_heartbeat_stopped(beat)


def test_run_process_captures_both_streams_and_normalizes_newlines(tmp_path):
    code = ("import sys; sys.stdout.buffer.write(b'out1\\r\\nout2\\rout3\\n'); sys.stderr.buffer.write(b'err\\n');"
            " sys.exit(3)")
    res = run_process([sys.executable, "-c", code], tmp_path, timeout=60, env=safe_env())
    assert res.returncode == 3 and not res.timed_out
    assert res.stdout == "out1\nout2\nout3\n" and res.stderr == "err\n"   # like text-mode universal newlines
    out = run_command([sys.executable, "-c", code], tmp_path, timeout=60, max_output=8).output
    assert out.endswith("err\n") and len(out) <= 8


def test_missing_command_is_not_found(tmp_path):
    res = run_command(["definitely-not-a-real-command-xyz"], tmp_path, timeout=10)
    assert res.not_found and not res.ok


# ---------------------------------------------------------------------------
# .bat / .cmd guard
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("arg", ['a&calc', 'x|y', 'a"b', '%PATH%', 'a^b', '!x!', 'a>b', '(x)', 'n\nl',
                                 'caf\u00e9.py', 'x\uff02y'])
def test_batch_files_refuse_arguments_cmd_would_interpret(arg):
    assert _batch_refusal(["C:/x/node_modules/.bin/eslint.cmd", "-f", "json", arg], windows=True)
    assert _batch_refusal(["tool.BAT", arg], windows=True)


def test_batch_guard_allows_plain_arguments_and_other_programs():
    plain = ["npm.cmd", "run", "--silent", "lint", "--", "--passWithNoTests", "src/app.js", "C:\\a b\\c.ts"]
    assert _batch_refusal(plain, windows=True) is None
    assert _batch_refusal(["node.exe", "a&calc"], windows=True) is None
    assert _batch_refusal(["x.cmd", "a&calc"], windows=False) is None


@pytest.mark.skipif(os.name != "nt", reason="cmd.exe semantics")
def test_batch_file_injection_is_refused_end_to_end(tmp_path):
    bat = tmp_path / "shim.cmd"
    bat.write_text("@echo off\r\necho ARG=%1\r\n")
    ok = run_command([str(bat), "safe.py"], tmp_path, timeout=30)
    assert ok.ok and "ARG=safe.py" in ok.output
    bad = run_command([str(bat), "x&echo INJECTED"], tmp_path, timeout=30)
    assert bad.refused and not bad.ok and "ARG=" not in bad.output
    res = run_tests([str(bat)], tmp_path, extra_args=["a&echo INJECTED"])
    assert not res.passed and "Refused" in res.output
