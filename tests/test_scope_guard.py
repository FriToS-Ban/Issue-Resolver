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
