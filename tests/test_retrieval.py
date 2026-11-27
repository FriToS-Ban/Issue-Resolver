"""Retrieval tests — independent of patch generation. Synthetic repos with known answers."""
from pathlib import Path

import pytest

from fix_issue.retrieval import RepoIndex, RepositoryRetriever, extract_mentions, retrieval_metrics
from fix_issue.retrieval.embeddings import EmbeddingCache, LocalNgramEmbedder, cosine
from fix_issue.retrieval.lexical import BM25
from fix_issue.retrieval.mentions import resolve_module, resolve_path
from fix_issue.retrieval.text import tokenize


def _w(root: Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


@pytest.fixture
def shop(tmp_path: Path) -> Path:
    _w(tmp_path, "shop/__init__.py", "")
    _w(tmp_path, "shop/utils/__init__.py", "")
    _w(tmp_path, "shop/utils/dates.py",
       "from datetime import date\n\n"
       "def is_expired(expiry: date, today: date) -> bool:\n"
       "    \"\"\"True when the expiry date is in the past.\"\"\"\n"
       "    return expiry > today\n")
    _w(tmp_path, "shop/discounts.py",
       "from shop.utils.dates import is_expired\n\n"
       "class Coupon:\n"
       "    def __init__(self, code, percent, expiry):\n"
       "        self.code, self.percent, self.expiry = code, percent, expiry\n\n"
       "    def is_valid(self, today):\n"
       "        return not is_expired(self.expiry, today)\n")
    _w(tmp_path, "shop/checkout.py",
       "from shop.discounts import Coupon\n\n"
       "def apply_coupon(total, coupon: Coupon, today):\n"
       "    if coupon.is_valid(today):\n"
       "        return total * (1 - coupon.percent / 100)\n"
       "    return total\n")
    _w(tmp_path, "shop/inventory.py",
       "class Inventory:\n"
       "    def __init__(self):\n"
       "        self._stock = {}\n\n"
       "    def available(self, sku):\n"
       "        return self._stock[sku]\n")
    _w(tmp_path, "shop/reports.py", "def sales_report(orders):\n    return sum(o.total for o in orders)\n")
    _w(tmp_path, "shop/pricing.py", "def price_with_tax(price, rate):\n    return price * (1 + rate)\n")
    _w(tmp_path, "tests/__init__.py", "")
    _w(tmp_path, "tests/test_checkout.py",
       "from shop.checkout import apply_coupon\n\ndef test_apply():\n    assert apply_coupon\n")
    _w(tmp_path, "README.md", "# shop\nA coupon and inventory library.\n")
    return tmp_path


# --- tokenisation / mentions --------------------------------------------------------

def test_tokenize_splits_identifiers():
    toks = tokenize("isExpired is_expired HTTPServer")
    assert "isexpired" in toks and "expired" in toks and "http" in toks and "server" in toks


def test_resolve_absolute_and_windows_paths():
    files = ["shop/inventory.py", "shop/utils/dates.py", "pkg/__init__.py", "other/__init__.py"]
    assert resolve_path("/home/u/proj/shop/inventory.py", files) == "shop/inventory.py"
    assert resolve_path(r"C:\work\shop\utils\dates.py", files) == "shop/utils/dates.py"
    assert resolve_path("dates.py", files) == "shop/utils/dates.py"
    assert resolve_path("__init__.py", files) is None          # ambiguous bare name
    assert resolve_module("shop.utils.dates", files) == "shop/utils/dates.py"


def test_extract_python_traceback_frames(shop):
    files = RepoIndex.build(shop).all_paths
    text = ('Traceback (most recent call last):\n'
            '  File "/srv/app/shop/checkout.py", line 4, in apply_coupon\n'
            '  File "/srv/app/shop/inventory.py", line 6, in available\n'
            "KeyError: 'X1'\n")
    m = extract_mentions(text, files)
    assert [f.path for f in m.frames] == ["shop/checkout.py", "shop/inventory.py"]
    assert m.frames[-1].function == "available"
    assert "KeyError" in m.error_types
    assert "available" in m.identifiers


def test_extract_identifiers_and_dotted(shop):
    files = RepoIndex.build(shop).all_paths
    m = extract_mentions("Calling `Coupon.is_valid()` returns True; see shop.utils.dates", files)
    assert "Coupon.is_valid" in m.identifiers
    assert "shop/utils/dates.py" in m.paths


# --- index ----------------------------------------------------------------------

def test_index_symbols_imports_and_refs(shop):
    idx = RepoIndex.build(shop)
    assert {s.path for s in idx.lookup("is_expired")} == {"shop/utils/dates.py"}
    assert idx.lookup("Coupon.is_valid")[0].kind == "method"
    assert "shop/utils/dates.py" in idx.imports["shop/discounts.py"]
    assert "shop/discounts.py" in idx.importers["shop/utils/dates.py"]
    callers = {p for p, _ in idx.callers("is_expired")}
    assert "shop/discounts.py" in callers
    # local variables must not create references
    inv = idx.lookup("Inventory.available")[0]
    assert "sku" not in inv.refs


def test_test_runner_module_is_not_a_test(tmp_path):
    _w(tmp_path, "pkg/test_runner.py", "def run_tests():\n    pass\n")
    _w(tmp_path, "pkg/test_math.py", "def test_add():\n    assert 1\n")
    idx = RepoIndex.build(tmp_path)
    assert not idx.files["pkg/test_runner.py"].is_test
    assert idx.files["pkg/test_math.py"].is_test


# --- channels ----------------------------------------------------------------------

def test_bm25_prefers_matching_doc():
    bm = BM25([["coupon", "expiry"], ["inventory", "stock"], ["report"]])
    s = bm.scores(["coupon", "expired"])
    assert s[0] > s[1] and s[0] > s[2]


def test_local_embedder_morphology():
    e = LocalNgramEmbedder(dim=512)
    e.fit(["expired coupon", "inventory stock", "is_expired expiry date"])
    q, a, b = e.embed(["coupon expiry", "def is_expired(expiry)", "inventory stock level"], "query")
    assert cosine(q, a) > cosine(q, b)


def test_embedding_cache_reuses_vectors(tmp_path):
    class Fake:
        name, semantic, calls = "fake", True, 0

        def embed(self, texts, kind):
            Fake.calls += 1
            return [[1.0, 0.0] for _ in texts]

    cache = EmbeddingCache(tmp_path / "c.jsonl")
    cache.embed(Fake(), ["a", "b"])
    cache.embed(Fake(), ["a", "b"])
    assert Fake.calls == 1
    assert EmbeddingCache(tmp_path / "c.jsonl").vec  # persisted


# --- retriever ------------------------------------------------------------------------

def test_traceback_file_always_included_and_first(shop):
    r = RepositoryRetriever(shop)
    res = r.retrieve("KeyError for unknown SKU",
                     'File "/x/shop/inventory.py", line 6, in available\nKeyError: \'X\'', top_n=3)
    assert res.files[0].path == "shop/inventory.py"
    assert any("stack trace" in reason for reason in res.files[0].reasons)
    assert "Inventory.available" in res.files[0].symbols


def test_two_hop_root_cause_found_through_call_graph(shop):
    r = RepositoryRetriever(shop)
    res = r.retrieve("Expired coupons are still accepted at checkout",
                     "A coupon whose expiry date passed last week still gets applied by apply_coupon.",
                     top_n=4)
    paths = res.paths()
    assert "shop/checkout.py" in paths
    assert "shop/utils/dates.py" in paths      # reached via apply_coupon -> Coupon.is_valid -> is_expired
    dates = next(f for f in res.files if f.path == "shop/utils/dates.py")
    assert dates.reasons                        # explained


def test_unrelated_files_cut_off(shop):
    r = RepositoryRetriever(shop)
    res = r.retrieve("Inventory.available crashes", "`Inventory.available('nope')` raises KeyError", top_n=6)
    assert "shop/reports.py" not in res.paths()


def test_tests_listed_separately(shop):
    r = RepositoryRetriever(shop)
    res = r.retrieve("apply_coupon applies expired coupon", "apply_coupon is wrong", top_n=4)
    assert all(not f.is_test for f in res.files)
    assert any(t.path == "tests/test_checkout.py" for t in res.tests)


def test_hints_from_plan_add_files(shop):
    r = RepositoryRetriever(shop)
    res = r.retrieve("something about prices", "", top_n=3, hint_paths=["shop/pricing.py"])
    assert "shop/pricing.py" in res.paths()


def test_render_context_live_reads_disk(shop):
    r = RepositoryRetriever(shop)
    res = r.retrieve("Inventory.available", "", top_n=2)
    (shop / "shop/inventory.py").write_text("class Inventory:\n    CHANGED = 1\n", encoding="utf-8")
    assert "CHANGED" in r.render_context(res, 10_000, live=True)
    assert "CHANGED" not in r.render_context(res, 10_000, live=False)


def test_retrieval_metrics():
    from fix_issue.retrieval.mentions import Mentions
    from fix_issue.retrieval.retriever import RetrievalResult, RetrievedFile

    res = RetrievalResult([RetrievedFile("a.py", 3, symbols=["f"]), RetrievedFile("b.py", 2)], [], Mentions(),
                          ["g"], "x", False)
    m = retrieval_metrics(res, ["b.py"], ["f", "zz"])
    assert m["recall@1"] == 0 and m["recall@3"] == 1.0 and m["mrr"] == 0.5 and m["symbol_recall"] == 0.5


# --- containment: nothing outside the clone reaches a prompt ---------------------------

import os  # noqa: E402

from fix_issue.retrieval.index import safe_repo_file  # noqa: E402


def _symlink(link: Path, target: Path, is_dir: bool = False) -> None:
    try:
        link.symlink_to(target, target_is_directory=is_dir)
    except (OSError, NotImplementedError):
        pytest.skip("cannot create symlinks here")


def _outside_secret(tmp_path_factory) -> Path:
    secret = tmp_path_factory.mktemp("outside") / "host-secret.md"
    secret.write_text("HOST_SECRET_VALUE\n", encoding="utf-8")
    return secret


def test_index_skips_symlinked_file_pointing_outside(shop, tmp_path_factory):
    secret = _outside_secret(tmp_path_factory)
    (shop / "docs").mkdir()
    _symlink(shop / "docs" / "notes.md", secret)
    idx = RepoIndex.build(shop)
    assert "docs/notes.md" not in idx.files and "docs/notes.md" not in idx.all_paths
    assert all("HOST_SECRET_VALUE" not in fi.text for fi in idx.files.values())
    assert "shop/checkout.py" in idx.files                       # ordinary files still indexed


def test_index_does_not_walk_symlinked_directories(shop, tmp_path_factory):
    secret = _outside_secret(tmp_path_factory)
    _symlink(shop / "vendored_docs", secret.parent, is_dir=True)
    idx = RepoIndex.build(shop)
    assert not any(p.startswith("vendored_docs/") for p in idx.all_paths)


def test_index_does_not_walk_junctions(shop, tmp_path_factory):
    winapi = pytest.importorskip("_winapi")
    if not hasattr(winapi, "CreateJunction"):
        pytest.skip("no junction support")
    secret = _outside_secret(tmp_path_factory)
    junction = shop / "linked"
    winapi.CreateJunction(str(secret.parent), str(junction))
    try:
        idx = RepoIndex.build(shop)
        assert not any(p.startswith("linked/") for p in idx.all_paths)
        assert safe_repo_file(shop, "linked/host-secret.md") is None
    finally:
        os.rmdir(junction)                                        # removes the junction only


def test_safe_repo_file_rejects_traversal_absolute_and_links(shop, tmp_path_factory):
    assert safe_repo_file(shop, "shop/checkout.py") == shop / "shop" / "checkout.py"
    for bad in ("../x.py", "shop/../../x.py", "/etc/passwd", "C:/Windows/win.ini", "", "shop", "missing.py"):
        assert safe_repo_file(shop, bad) is None, bad
    secret = _outside_secret(tmp_path_factory)
    _symlink(shop / "link.md", secret)
    assert safe_repo_file(shop, "link.md") is None


def test_render_context_never_reads_plan_paths_outside_the_clone(shop, tmp_path_factory):
    secret = _outside_secret(tmp_path_factory)
    r = RepositoryRetriever(shop)
    res = r.retrieve("Inventory.available", "", top_n=2)
    rel = os.path.relpath(secret, shop).replace(os.sep, "/")     # "../outside0/host-secret.md"
    for live in (True, False):
        out = r.render_context(res, 10_000, extra_paths=[rel, str(secret)], live=live)
        assert "HOST_SECRET_VALUE" not in out


def test_live_render_skips_a_file_swapped_for_a_symlink(shop, tmp_path_factory):
    secret = _outside_secret(tmp_path_factory)
    r = RepositoryRetriever(shop)
    res = r.retrieve("Inventory.available", "", top_n=2)
    assert "shop/inventory.py" in res.paths()
    (shop / "shop" / "inventory.py").unlink()
    _symlink(shop / "shop" / "inventory.py", secret)              # e.g. done by test code later
    assert "HOST_SECRET_VALUE" not in r.render_context(res, 10_000, live=True)


def test_render_context_has_no_phantom_blank_last_line(shop):
    r = RepositoryRetriever(shop)
    res = r.retrieve("Inventory.available", "", top_n=2)
    out = r.render_context(res, 10_000)
    assert "        return self._stock[sku]\n```" in out
    assert "\n\n```" not in out


# --- pytest location lines ------------------------------------------------------------

def test_extract_pytest_short_and_long_frames(shop):
    files = RepoIndex.build(shop).all_paths
    out = ("tests/test_checkout.py:4: in test_apply\n    assert apply_coupon(1, c, d)\n"
           + r"shop\inventory.py:6: in available" + "\n    return self._stock[sku]\n"
           + "E   KeyError: 'X1'\n\nshop/checkout.py:5: KeyError\n")
    got = [(f.path, f.line, f.function) for f in extract_mentions(out, files).frames]
    assert ("tests/test_checkout.py", 4, "test_apply") in got
    assert ("shop/inventory.py", 6, "available") in got
    assert ("shop/checkout.py", 5, None) in got


def test_pytest_frames_do_not_duplicate_native_frames(shop):
    files = RepoIndex.build(shop).all_paths
    text = '  File "/srv/shop/inventory.py", line 6, in available\nshop/inventory.py:6: in available\n'
    assert len(extract_mentions(text, files).frames) == 1
