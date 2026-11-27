"""Tests for scope_guard.py"""
import textwrap
from pathlib import Path

import pytest

from fix_issue.scope_guard import ScopeViolation, check


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SMALL_DIFF = textwrap.dedent("""\
    diff --git a/foo.py b/foo.py
    --- a/foo.py
    +++ b/foo.py
    @@ -1,3 +1,3 @@
    -x = 1
    +x = 2
     y = 3
""")

def _make_ignore(tmp_path: Path, patterns: list[str]) -> None:
    ignore = tmp_path / ".fix-issue-ignore"
    ignore.write_text("\n".join(patterns) + "\n")


# ---------------------------------------------------------------------------
# Size checks
# ---------------------------------------------------------------------------

def test_small_diff_passes(tmp_path):
    check(SMALL_DIFF, tmp_path, max_diff_lines=150)  # should not raise


def test_oversized_diff_raises(tmp_path):
    big_diff = "\n".join(["+line"] * 200)
    with pytest.raises(ScopeViolation, match="too large"):
        check(big_diff, tmp_path, max_diff_lines=150)


def test_exactly_at_limit_passes(tmp_path):
    diff = "\n".join(["+line"] * 150)
    check(diff, tmp_path, max_diff_lines=150)  # boundary: should pass


def test_one_over_limit_raises(tmp_path):
    diff = "\n".join(["+line"] * 151)
    with pytest.raises(ScopeViolation):
        check(diff, tmp_path, max_diff_lines=150)


def test_header_lines_not_counted(tmp_path):
    """--- and +++ header lines must not count toward the size limit."""
    diff = textwrap.dedent("""\
        --- a/file.py
        +++ b/file.py
        +actual change
    """)
    # Only 1 real changed line; should pass even with a tight limit
    check(diff, tmp_path, max_diff_lines=1)


# ---------------------------------------------------------------------------
# Ignored paths
# ---------------------------------------------------------------------------

def test_no_ignore_file_passes(tmp_path):
    check(SMALL_DIFF, tmp_path, max_diff_lines=150)


def test_unmatched_pattern_passes(tmp_path):
    _make_ignore(tmp_path, ["auth/**", "payments/**"])
    check(SMALL_DIFF, tmp_path, max_diff_lines=150)  # touches foo.py → passes


def test_matching_glob_raises(tmp_path):
    _make_ignore(tmp_path, ["auth/**"])
    diff = textwrap.dedent("""\
        diff --git a/auth/login.py b/auth/login.py
        --- a/auth/login.py
        +++ b/auth/login.py
        @@ -1 +1 @@
        -old
        +new
    """)
    with pytest.raises(ScopeViolation, match="auth/login.py"):
        check(diff, tmp_path, max_diff_lines=150)


def test_comment_lines_in_ignore_ignored(tmp_path):
    _make_ignore(tmp_path, ["# this is a comment", "foo.py"])
    with pytest.raises(ScopeViolation):
        check(SMALL_DIFF, tmp_path, max_diff_lines=150)


def test_blank_lines_in_ignore_ignored(tmp_path):
    _make_ignore(tmp_path, ["", "  ", "auth/**"])
    # SMALL_DIFF touches foo.py — should still pass
    check(SMALL_DIFF, tmp_path, max_diff_lines=150)


def test_wildcard_extension_pattern(tmp_path):
    _make_ignore(tmp_path, ["*.env"])
    diff = textwrap.dedent("""\
        --- a/.env
        +++ b/.env
        @@ -1 +1 @@
        -SECRET=old
        +SECRET=new
    """)
    with pytest.raises(ScopeViolation, match=r"\.env"):
        check(diff, tmp_path, max_diff_lines=150)


# ===========================================================================
# evaluate(): path- and severity-aware policy
# ===========================================================================

import difflib  # noqa: E402

from fix_issue.scope_guard import (  # noqa: E402
    ScopeConstraints,
    ScopeReport,
    classify_path,
    evaluate,
    load_rules,
    path_matches,
)


def _gdiff(path: str, plus: int = 1, minus: int = 1, *, new: bool = False, deleted: bool = False) -> str:
    """A git-style diff touching `path` with `plus` added and `minus` removed lines."""
    head = [f"diff --git a/{path} b/{path}"]
    if new:
        head.append("new file mode 100644")
    if deleted:
        head.append("deleted file mode 100644")
    head += [
        f"--- {'/dev/null' if new else 'a/' + path}",
        f"+++ {'/dev/null' if deleted else 'b/' + path}",
        f"@@ -{1 if minus else 0},{minus} +{1 if plus else 0},{plus} @@",
    ]
    body = [f"-old line {i}" for i in range(minus)] + [f"+new line {i}" for i in range(plus)]
    return "\n".join(head + body) + "\n"


def _by(report: ScopeReport, category: str) -> list:
    return [f for f in report.findings if f.category == category]


@pytest.mark.parametrize("path,category", [
    (".env", "secrets"), ("config/.env.production", "secrets"), ("deploy/certs/server.pem", "secrets"),
    ("keys/id.key", "secrets"), ("ops/secrets/db.yaml", "secrets"), ("aws_credentials.json", "secrets"),
    (".github/workflows/ci.yml", "ci"), (".gitlab-ci.yml", "ci"), (".circleci/config.yml", "ci"),
    ("Jenkinsfile", "ci"),
    ("poetry.lock", "lockfile"), ("web/package-lock.json", "lockfile"), ("yarn.lock", "lockfile"),
    ("pnpm-lock.yaml", "lockfile"), ("Pipfile.lock", "lockfile"), ("uv.lock", "lockfile"),
    ("Cargo.lock", "lockfile"), ("go.sum", "lockfile"),
    ("app/auth/views.py", "auth"), ("src/oauth_client.py", "auth"), ("app/permissions.py", "auth"),
    ("app/oauth/flow.js", "auth"), ("app/security/csrf.py", "auth"), ("web/login.html", "auth"),
    ("App/Auth/Views.py", "auth"),
    ("shop/payments/charge.py", "payments"), ("src/payment_gateway.py", "payments"),
    ("billing/models.py", "payments"), ("app/stripe/hooks.py", "payments"), ("app/invoices.py", "payments"),
    ("app/migrations/0001_initial.py", "migrations"), ("db/alembic/versions/abc.py", "migrations"),
    ("tests/test_auth.py", "tests"), ("pkg/foo_test.py", "tests"), ("test_util.py", "tests"),
    ("web/__tests__/a.js", "tests"), ("web/a.spec.ts", "tests"), ("web/b.test.js", "tests"),
    ("docs/index.rst", "docs"), ("README.md", "docs"), ("docs/auth/setup.md", "docs"),
    ("src/app.py", "source"), ("fix_issue/config.py", "source"),
])
def test_classify_path_defaults(path, category):
    assert classify_path(path) == category


def test_path_matches_semantics():
    assert path_matches("a/b/key.pem", "*.pem")            # no slash -> any component
    assert path_matches("src/auth/login.py", "**/auth/**")  # **/ -> any depth, incl. zero
    assert path_matches("auth/login.py", "**/auth/**")
    assert path_matches("src/core/x.py", "src/core/")       # directory prefix
    assert not path_matches("lib/src/core/x.py", "src/core")  # slash -> anchored at root
    assert not path_matches("src/coreutils.py", "src/core")


def test_evaluate_clean_source_change_is_ok(tmp_path):
    report = evaluate(SMALL_DIFF, tmp_path)
    assert report.ok and not report.requires_review
    assert report.findings == []
    assert report.raw_lines == 2 and report.effective_lines == 2.0 and report.limit == 150
    assert report.categories == {"foo.py": "source"}


@pytest.mark.parametrize("path,category", [
    (".env", "secrets"), ("certs/prod.pem", "secrets"), (".github/workflows/release.yml", "ci"),
    ("Jenkinsfile", "ci"), ("poetry.lock", "lockfile"), ("package-lock.json", "lockfile"),
])
def test_secrets_ci_lockfiles_block(tmp_path, path, category):
    report = evaluate(_gdiff(path), tmp_path)
    assert not report.ok
    blocks = [f for f in report.findings if f.severity == "block"]
    assert [f.category for f in blocks] == [category]
    assert path in report.blocking_message()


def test_ignored_path_blocks(tmp_path):
    _make_ignore(tmp_path, ["shopcart/payments/**"])
    report = evaluate(_gdiff("shopcart/payments/cart.py"), tmp_path)
    assert not report.ok
    assert _by(report, "ignored")[0].severity == "block"
    assert "shopcart/payments/**" in report.blocking_message()


@pytest.mark.parametrize("path,category", [("app/auth/tokens.py", "auth"), ("shop/payments/charge.py", "payments")])
def test_sensitive_paths_require_review_within_budget(tmp_path, path, category):
    report = evaluate(_gdiff(path, plus=5, minus=5), tmp_path)
    assert report.ok and report.requires_review
    assert [f.severity for f in _by(report, category)] == ["review"]


@pytest.mark.parametrize("path,category", [("app/auth/tokens.py", "auth"), ("billing/charge.py", "payments")])
def test_sensitive_paths_over_budget_block(tmp_path, path, category):
    report = evaluate(_gdiff(path, plus=11, minus=10), tmp_path)  # 21 > 20
    assert not report.ok
    assert any(f.severity == "block" and "budget of 20" in f.message for f in _by(report, category))


def test_sensitive_budget_is_summed_across_files(tmp_path):
    diff = _gdiff("app/auth/a.py", plus=6, minus=6) + _gdiff("app/auth/b.py", plus=6, minus=6)
    report = evaluate(diff, tmp_path)
    assert not report.ok
    assert "24 changed lines in auth paths" in report.blocking_message()


def test_weights_tests_and_docs(tmp_path):
    diff = (_gdiff("tests/test_x.py", plus=100, minus=0, new=True)
            + _gdiff("docs/guide.md", plus=40, minus=0)
            + _gdiff("src/app.py", plus=5, minus=5))
    report = evaluate(diff, tmp_path)
    assert report.raw_lines == 150
    assert report.effective_lines == pytest.approx(100 * 0.5 + 40 * 0.25 + 10)
    assert report.categories == {"tests/test_x.py": "tests", "docs/guide.md": "docs", "src/app.py": "source"}
    assert report.ok


def test_weighted_size_limit(tmp_path):
    assert evaluate(_gdiff("tests/test_big.py", plus=250, minus=0), tmp_path).ok   # 125 effective
    report = evaluate(_gdiff("src/big.py", plus=151, minus=0), tmp_path)
    assert not report.ok and _by(report, "size")[0].severity == "block"
    assert "too large" in report.blocking_message()


def test_headerless_diff_counts_as_source(tmp_path):
    report = evaluate("\n".join(["+line"] * 200), tmp_path)
    assert report.raw_lines == 200 and not report.ok


def test_triage_can_tighten_size_limit(tmp_path):
    report = evaluate(_gdiff("src/app.py", plus=15, minus=15), tmp_path, 150, ScopeConstraints(max_diff_lines=20))
    assert report.limit == 20 and not report.ok
    assert "triage" in _by(report, "size")[-1].message


def test_triage_cannot_loosen_size_limit(tmp_path):
    report = evaluate(_gdiff("src/app.py", plus=200, minus=0), tmp_path, 150, ScopeConstraints(max_diff_lines=500))
    assert report.limit == 150 and not report.ok
    assert any(f.severity == "info" and "only tighten" in f.message for f in _by(report, "size"))


def test_triage_forbidden_paths_block(tmp_path):
    report = evaluate(_gdiff("src/core/engine.py"), tmp_path,
                      constraints=ScopeConstraints(forbidden_paths=["src/core/"]))
    assert not report.ok
    finding = _by(report, "triage")[0]
    assert finding.severity == "block" and finding.path == "src/core/engine.py"


def test_triage_allowed_paths_review_outside(tmp_path):
    diff = _gdiff("src/app.py") + _gdiff("src/other.py") + _gdiff("tests/test_app.py") + _gdiff("docs/a.md")
    report = evaluate(diff, tmp_path, constraints=ScopeConstraints(allowed_paths=["src/app.py"]))
    assert report.ok and report.requires_review
    triage = _by(report, "triage")
    assert [f.path for f in triage] == ["src/other.py"]   # tests/docs exempt
    assert triage[0].severity == "review" and "human review" in triage[0].message


def test_migration_add_is_review(tmp_path):
    report = evaluate(_gdiff("app/migrations/0002_add_col.py", plus=10, minus=0, new=True), tmp_path)
    assert report.ok and report.requires_review
    assert [f.severity for f in _by(report, "migrations")] == ["review"]


def test_migration_modify_existing_blocks(tmp_path):
    report = evaluate(_gdiff("app/migrations/0001_initial.py"), tmp_path)
    assert not report.ok
    assert "existing migration" in report.blocking_message()


def test_migration_delete_existing_blocks(tmp_path):
    report = evaluate(_gdiff("db/alembic/versions/abc.py", plus=0, minus=3, deleted=True), tmp_path)
    assert not report.ok
    assert "deletes the existing migration" in report.blocking_message()


def test_migration_base_contents_decide_existence(tmp_path):
    # difflib-style diff has no "new file" marker; base_contents says the file did not exist.
    diff = "--- a/app/migrations/0003.py\n+++ b/app/migrations/0003.py\n@@ -0,0 +1 @@\n+x = 1\n"
    report = evaluate(diff, tmp_path, base_contents={"app/migrations/0003.py": None},
                      new_contents={"app/migrations/0003.py": "x = 1\n"})
    assert report.ok
    assert [f.severity for f in _by(report, "migrations")] == ["review"]


def _api(tmp_path, base: str, new: str, path: str = "pkg/core.py") -> list:
    report = evaluate(_gdiff(path), tmp_path, base_contents={path: base}, new_contents={path: new})
    return _by(report, "public_api")


def test_public_api_body_only_change_is_not_api_change(tmp_path):
    base = "def foo(a, b=1):\n    return a\n\nclass C:\n    def m(self, x):\n        return x\n"
    new = "def foo(a, b=1):\n    return a + b\n\nclass C:\n    def m(self, x):\n        return x * 2\n"
    assert _api(tmp_path, base, new) == []


def test_public_api_signature_change_detected(tmp_path):
    findings = _api(tmp_path, "def foo(a, b=1):\n    pass\n", "def foo(a):\n    pass\n")
    assert len(findings) == 1 and findings[0].severity == "review"
    assert "`foo`" in findings[0].message


@pytest.mark.parametrize("old,new,flagged", [
    ("def f(a, b): pass", "def f(b, a): pass", True),            # reordered
    ("def f(a, b=1): pass", "def f(a, b): pass", True),          # default removed
    ("def f(a): pass", "def f(a, b): pass", True),               # new required param
    ("def f(a): pass", "def f(a, b=None): pass", False),         # optional param appended
    ("def f(a, *, k=1): pass", "def f(a): pass", True),          # kw-only removed
    ("def f(a, **kw): pass", "def f(a): pass", True),            # **kwargs removed
    ("def f(a: int) -> int: pass", "def f(a: str) -> str: pass", False),  # annotations only
])
def test_public_api_signature_rules(tmp_path, old, new, flagged):
    assert bool(_api(tmp_path, old + "\n", new + "\n")) is flagged


def test_public_api_removed_symbols(tmp_path):
    base = "def keep(): pass\ndef gone(): pass\ndef _private(): pass\nclass Old: pass\n"
    new = "def keep(): pass\n"
    messages = " ".join(f.message for f in _api(tmp_path, base, new))
    assert "`gone`" in messages and "`Old`" in messages and "_private" not in messages


def test_public_api_method_change_and_removal(tmp_path):
    base = "class C:\n    def m(self, x): pass\n    def n(self): pass\n    def _p(self): pass\n"
    new = "class C:\n    def m(self, x, y): pass\n"
    messages = [f.message for f in _api(tmp_path, base, new)]
    assert any("`C.m`" in m for m in messages)
    assert any("`C.n`" in m and "Removes" in m for m in messages)
    assert not any("_p" in m for m in messages)
    compatible = "class C:\n    def m(self, x, y=0): pass\n    def n(this): pass\n    def _p(self, z): pass\n"
    assert _api(tmp_path, base, compatible) == []


def test_public_api_all_change(tmp_path):
    findings = _api(tmp_path, "__all__ = ['a']\ndef a(): pass\n", "__all__ = ['a', 'b']\ndef a(): pass\nb = 1\n")
    assert any("__all__" in f.message and "b" in f.message for f in findings)


def test_public_api_package_init_export_change(tmp_path):
    findings = _api(tmp_path, "from .core import a\n", "from .core import a, b\n", path="pkg/__init__.py")
    assert len(findings) == 1 and "exports" in findings[0].message and "b" in findings[0].message


def test_public_api_ignores_test_files(tmp_path):
    assert _api(tmp_path, "def helper(a): pass\n", "", path="tests/test_core.py") == []


def test_public_api_module_deletion(tmp_path):
    path = "pkg/util.py"
    report = evaluate(_gdiff(path, plus=0, minus=2, deleted=True), tmp_path,
                      base_contents={path: "def public(): pass\n"}, new_contents={})
    assert any("public" in f.message for f in _by(report, "public_api"))
    assert _by(report, "deletion")[0].severity == "review"


def test_deletion_requires_review(tmp_path):
    report = evaluate(_gdiff("src/old.py", plus=0, minus=4, deleted=True), tmp_path)
    assert report.ok and report.requires_review
    assert _by(report, "deletion")[0].path == "src/old.py"


def test_rename_requires_review_and_checks_old_path(tmp_path):
    diff = ("diff --git a/app/auth/x.py b/misc/x.py\nsimilarity index 100%\n"
            "rename from app/auth/x.py\nrename to misc/x.py\n")
    report = evaluate(diff, tmp_path)
    assert report.requires_review
    assert report.categories == {"misc/x.py": "auth"}   # stricter of old/new path
    assert _by(report, "deletion")


@pytest.mark.parametrize("diff", [
    "diff --git a/logo.png b/logo.png\nindex 1..2 100644\nBinary files a/logo.png and b/logo.png differ\n",
    "diff --git a/logo.png b/logo.png\nindex 1..2 100644\nGIT binary patch\nliteral 5\nMcmZ?wbhEHbDXa\n\n"
    "literal 0\nHcmV?d00001\n",
])
def test_binary_patch_blocks(tmp_path, diff):
    report = evaluate(diff, tmp_path)
    assert not report.ok
    assert _by(report, "binary")[0].path == "logo.png"


def test_difflib_style_diff_is_parsed(tmp_path):
    old = ["a = 1\n", "b = 2\n", "-- sql comment\n"]
    new = ["a = 1\n", "b = 3\n", "select 1\n"]
    with_prefix = "".join(difflib.unified_diff(old, new, "a/app/auth/q.py", "b/app/auth/q.py"))
    no_prefix = "".join(difflib.unified_diff(old, new, "docs/q.md", "docs/q.md"))
    r1 = evaluate(with_prefix, tmp_path)
    assert r1.categories == {"app/auth/q.py": "auth"}
    assert r1.raw_lines == 4   # the removed "-- sql comment" line ("--- sql comment") is content
    r2 = evaluate(no_prefix, tmp_path)
    assert r2.categories == {"docs/q.md": "docs"} and r2.effective_lines == pytest.approx(1.0)


def test_custom_scope_rules_toml(tmp_path):
    (tmp_path / ".fix-issue-scope.toml").write_text(textwrap.dedent("""\
        [[rule]]
        category = "generated"
        patterns = ["src/generated/**"]
        severity = "block"

        [[rule]]
        category = "fixtures"
        patterns = ["tests/fixtures/**"]
        weight = 0.1
    """))
    rules = load_rules(tmp_path)
    assert classify_path("src/generated/api.py", rules) == "generated"
    assert classify_path("tests/fixtures/data.json", rules) == "fixtures"
    assert classify_path(".env", rules) == "secrets"            # defaults still apply
    report = evaluate(_gdiff("src/generated/api.py"), tmp_path)
    assert not report.ok and _by(report, "generated")[0].severity == "block"
    report = evaluate(_gdiff("tests/fixtures/data.json", plus=100, minus=0), tmp_path)
    assert report.ok and report.effective_lines == pytest.approx(10.0)


def test_invalid_scope_rules_toml_falls_back_to_defaults(tmp_path):
    (tmp_path / ".fix-issue-scope.toml").write_text("[[rule]\ncategory = ")
    report = evaluate(_gdiff(".env"), tmp_path)
    assert not report.ok
    assert _by(report, "config")[0].severity == "info"


def test_report_to_dict_and_messages(tmp_path):
    diff = _gdiff(".env") + _gdiff("app/auth/x.py")
    report = evaluate(diff, tmp_path)
    data = report.to_dict()
    assert data["ok"] is False and data["requires_review"] is True
    assert {f["category"] for f in data["findings"]} == {"secrets", "auth"}
    assert ".env" in report.blocking_message() and "auth" not in report.blocking_message()
    assert "app/auth/x.py" in report.review_message()


# ===========================================================================
# Hardening: alias paths, policy files, base-commit policy, coverage, char budget
# ===========================================================================

import subprocess  # noqa: E402

from fix_issue.scope_guard import DEFAULT_RULES, load_policy  # noqa: E402

PERMISSIVE = (
    '[[rule]]\ncategory = "anything"\n'
    'patterns = [".github/**", ".fix-issue-scope.toml", ".fix-issue-ignore", "**"]\nseverity = "info"\n'
)


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True)


def _repo(root: Path, files: dict[str, str]) -> Path:
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@example.invalid")
    _git(root, "config", "user.name", "t")
    _git(root, "config", "commit.gpgsign", "false")
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "base", "--allow-empty")
    return root


@pytest.mark.parametrize("path", [
    ".github/workflows./ci.yml", ".github/workflows /ci.yml", "GITHUB~1/WORKFL~1/ci.yml",
    ".git./hooks/pre-push", ".git/hooks/pre-push", "sub/.GIT/config", "src/app.py:stream",
    "PRE-CO~1.YAM", "../outside.py",
])
def test_alias_paths_are_unsafe_and_block(tmp_path, path):
    assert classify_path(path) == "unsafe_path"
    report = evaluate(_gdiff(path), tmp_path)
    assert not report.ok
    assert [f.category for f in report.findings if f.severity == "block"] == ["unsafe_path"]
    assert set(report.categories.values()) == {"unsafe_path"}


@pytest.mark.parametrize("path", [".fix-issue-scope.toml", ".fix-issue-ignore", "sub/.FIX-ISSUE-IGNORE"])
def test_policy_files_always_block_even_when_repo_rules_say_info(tmp_path, path):
    (tmp_path / ".fix-issue-scope.toml").write_text(PERMISSIVE)
    assert classify_path(path, load_rules(tmp_path)) == "policy"
    report = evaluate(_gdiff(path), tmp_path)
    assert not report.ok and _by(report, "policy")[0].severity == "block"


def test_policy_is_read_from_base_commit_not_working_tree(tmp_path):
    # Reviewer's repro: attempt 1 writes a permissive rule (on disk), attempt 2 edits CI.
    _repo(tmp_path, {"app.py": "x = 1\n", ".github/workflows/ci.yml": "on: push\n"})
    (tmp_path / ".fix-issue-scope.toml").write_text(PERMISSIVE)
    policy = load_policy(tmp_path)
    assert policy.source == "git:HEAD" and len(policy.rules) == len(DEFAULT_RULES)
    report = evaluate(_gdiff(".github/workflows/ci.yml"), tmp_path)
    assert not report.ok and _by(report, "ci")[0].severity == "block"


def test_committed_repo_rules_still_apply(tmp_path):
    rule = '[[rule]]\ncategory = "generated"\npatterns = ["src/generated/**"]\nseverity = "block"\n'
    _repo(tmp_path, {".fix-issue-scope.toml": rule})
    (tmp_path / ".fix-issue-scope.toml").write_text("")      # emptied in the working tree
    report = evaluate(_gdiff("src/generated/api.py"), tmp_path)
    assert not report.ok and _by(report, "generated")[0].severity == "block"


def test_base_contents_take_precedence_over_head(tmp_path):
    # e.g. test code committed a permissive policy on top of the base and restored the file:
    # the orchestrator's base_contents (read from the base SHA) are what counts.
    _repo(tmp_path, {".fix-issue-scope.toml": PERMISSIVE})
    report = evaluate(_gdiff(".github/workflows/ci.yml"), tmp_path,
                      base_contents={".fix-issue-scope.toml": None})
    assert not report.ok and _by(report, "ci")


def test_explicit_policy_snapshot_pins_the_rules(tmp_path):
    _repo(tmp_path, {"app.py": "x = 1\n"})
    policy = load_policy(tmp_path)                            # taken before any repo code runs
    (tmp_path / ".fix-issue-scope.toml").write_text(PERMISSIVE)
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "test code commits a new policy")
    report = evaluate(_gdiff(".github/workflows/ci.yml"), tmp_path, policy=policy)
    assert not report.ok and _by(report, "ci")


def test_working_tree_ignore_patterns_only_add_restrictions(tmp_path):
    _repo(tmp_path, {".fix-issue-ignore": "vendor/**\n"})
    (tmp_path / ".fix-issue-ignore").write_text("legacy/**\n")    # base pattern removed on disk
    assert load_policy(tmp_path).ignore_patterns == ["vendor/**", "legacy/**"]
    assert _by(evaluate(_gdiff("vendor/lib.py"), tmp_path), "ignored")
    assert _by(evaluate(_gdiff("legacy/old.py"), tmp_path), "ignored")


def test_unreadable_git_policy_is_flagged_and_working_tree_not_trusted(tmp_path):
    (tmp_path / ".git").write_text("gitdir: does-not-exist\n")
    (tmp_path / ".fix-issue-scope.toml").write_text(PERMISSIVE)
    report = evaluate(_gdiff(".github/workflows/ci.yml"), tmp_path)
    assert not report.ok and _by(report, "ci")
    assert report.requires_review and _by(report, "config")[0].severity == "review"


def test_legacy_check_blocks_policy_and_alias_paths(tmp_path):
    for path in (".fix-issue-ignore", ".fix-issue-scope.toml", ".git./hooks/pre-commit", "GIT~1/config"):
        with pytest.raises(ScopeViolation):
            check(_gdiff(path), tmp_path, max_diff_lines=150)


def test_legacy_check_reads_ignore_file_from_base_commit(tmp_path):
    _repo(tmp_path, {".fix-issue-ignore": "auth/**\n"})
    (tmp_path / ".fix-issue-ignore").write_text("# emptied by an earlier attempt\n")
    with pytest.raises(ScopeViolation):
        check(_gdiff("auth/login.py"), tmp_path, max_diff_lines=150)


@pytest.mark.parametrize("path,category,severity", [
    (".github/actions/setup/action.yml", "ci", "block"), (".github/CODEOWNERS", "ci", "block"),
    (".github/dependabot.yml", "ci", "block"), ("CODEOWNERS", "ci", "block"),
    ("docs/CODEOWNERS", "ci", "block"), (".pre-commit-config.yaml", "ci", "block"),
    (".travis.yml", "ci", "block"), (".gitlab-ci.yml", "ci", "block"), ("azure-pipelines.yml", "ci", "block"),
    ("Jenkinsfile", "ci", "block"), (".circleci/config.yml", "ci", "block"),
    (".npmrc", "secrets", "block"), ("web/.npmrc", "secrets", "block"), (".pypirc", "secrets", "block"),
    (".gitattributes", "git_metadata", "block"), (".gitmodules", "git_metadata", "block"),
    ("package.json", "build", "review"), ("setup.py", "build", "review"), ("setup.cfg", "build", "review"),
    ("tox.ini", "build", "review"), ("noxfile.py", "build", "review"), ("Makefile", "build", "review"),
    ("pyproject.toml", "build", "review"), ("requirements-dev.txt", "build", "review"),
])
def test_automation_credentials_git_metadata_and_build_files(tmp_path, path, category, severity):
    assert classify_path(path) == category
    report = evaluate(_gdiff(path), tmp_path)
    assert _by(report, category)[0].severity == severity
    assert report.ok is (severity != "block")


def test_fixture_manifests_under_tests_stay_tests():
    assert classify_path("tests/fixtures/package.json") == "tests"
    assert classify_path("docs/Makefile") == "docs"


def test_single_huge_line_exceeds_char_budget(tmp_path):
    big = ("diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-x = 1\n+x = 1; "
           + "y = 2; " * 50000 + "\n")
    report = evaluate(big, tmp_path, 10, ScopeConstraints(max_diff_lines=10))
    assert report.raw_lines == 2 and report.raw_chars > 350_000 and report.char_limit == 2000
    assert not report.ok
    assert any("characters" in f.message for f in _by(report, "size") if f.severity == "block")
    assert report.to_dict()["char_limit"] == 2000


def test_char_budget_allows_normal_diffs_and_follows_the_tightened_limit(tmp_path):
    assert evaluate(_gdiff("src/app.py", plus=100, minus=40), tmp_path).ok
    head = "diff --git a/src/app.py b/src/app.py\n--- a/src/app.py\n+++ b/src/app.py\n@@ -1,2 +1,2 @@\n"
    long_lines = head + "".join(f"-{'a' * 300}\n" for _ in range(2)) + "".join(f"+{'b' * 300}\n" for _ in range(2))
    assert evaluate(long_lines, tmp_path).ok                                    # 1200 chars <= 30000
    tight = evaluate(long_lines, tmp_path, 150, ScopeConstraints(max_diff_lines=5))
    assert not tight.ok and tight.char_limit == 1000


def test_triage_forbidden_globs_do_not_block_test_files(tmp_path):
    constraints = ScopeConstraints(forbidden_paths=["tests/**", "src/core/**"])
    assert evaluate(_gdiff("tests/test_cart.py", new=True), tmp_path, constraints=constraints).ok
    report = evaluate(_gdiff("src/core/engine.py"), tmp_path, constraints=constraints)
    assert not report.ok and _by(report, "triage")[0].severity == "block"


def test_test_files_still_subject_to_hard_and_repo_blocks(tmp_path):
    (tmp_path / ".fix-issue-ignore").write_text("tests/golden/**\n")
    constraints = ScopeConstraints(forbidden_paths=["tests/**"])
    for path, category in [("tests/golden/out.txt", "ignored"), ("tests/.env", "secrets"),
                           ("tests/.fix-issue-scope.toml", "policy"), ("tests/.git./x", "unsafe_path")]:
        report = evaluate(_gdiff(path), tmp_path, constraints=constraints)
        assert not report.ok and _by(report, category), path


@pytest.mark.parametrize("header", [
    "new file mode 120000\nindex 0000000..1111111\n",       # new symlink
    "index 1111111..2222222 120000\n",                     # symlink retargeted
    "old mode 100644\nnew mode 160000\n",                  # becomes a submodule gitlink
])
def test_symlink_and_gitlink_entries_block(tmp_path, header):
    diff = (f"diff --git a/docs/link.md b/docs/link.md\n{header}--- a/docs/link.md\n+++ b/docs/link.md\n"
            "@@ -1 +1 @@\n-x\n+/home/user/.ssh/id_rsa\n")
    report = evaluate(diff, tmp_path)
    assert not report.ok and _by(report, "special_file")[0].severity == "block"
