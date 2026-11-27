"""
retrieval/retriever.py — Layered, explainable repository retrieval.

Signals, in priority order (each normalised to 0..1 and weighted):

  1. explicit     files named in the issue text (paths, dotted modules)          w=3.0
     traceback    files in stack-trace frames (innermost frame strongest)        w=3.0
  2. symbol_def   files defining identifiers the issue mentions                   w=2.0
  3. lexical      BM25 over code-aware tokens (path tokens up-weighted)           w=1.0
     semantic     embedding similarity, max over the file's symbol chunks         w=1.0
  4. callee       definitions called by focus symbols                             w=0.9
     caller       files referencing focus symbols (hub names damped)              w=0.7
     imports      modules imported by strong seeds                                w=0.35
     imported_by  modules importing strong seeds                                  w=0.3
     hint         files/symbols suggested by triage or the planner               w=1.5

Stage 1 scores seeds from channels 1–3; stage 2 expands through the symbol/import graph,
scaling every expansion by the strength of the seed it came from, so a neighbour of a
strong seed outranks a weak lexical hit. Files below 20% of the top score are dropped
(explicit/traceback files are always kept) to avoid flooding the model with unrelated
code. Every selected file carries the human-readable reasons it was chosen.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

from fix_issue.retrieval.embeddings import EmbeddingBackend, EmbeddingCache, LocalNgramEmbedder, cosine
from fix_issue.retrieval.index import RepoIndex, Symbol, file_tokens, safe_repo_file, symbol_document
from fix_issue.retrieval.lexical import BM25
from fix_issue.retrieval.mentions import Mentions, extract_mentions
from fix_issue.retrieval.text import COMMON_CODE_NAMES, tokenize

WEIGHTS = {
    "explicit": 3.0, "traceback": 3.0, "symbol_def": 2.0, "hint": 1.5,
    "lexical": 1.0, "semantic": 1.0,
    "callee": 0.9, "caller": 0.7, "imports": 0.35, "imported_by": 0.3,
}
CUTOFF_RATIO = 0.2
MAX_ALWAYS = 8


@dataclass
class RetrievedFile:
    path: str
    score: float
    reasons: list[str] = field(default_factory=list)
    signals: dict[str, float] = field(default_factory=dict)
    symbols: list[str] = field(default_factory=list)   # focus symbols (qualnames) in this file
    is_test: bool = False

    def to_dict(self) -> dict:
        return {
            "path": self.path, "score": round(self.score, 3), "reasons": self.reasons,
            "signals": {k: round(v, 3) for k, v in self.signals.items() if v},
            "symbols": self.symbols, "is_test": self.is_test,
        }


@dataclass
class RetrievalResult:
    files: list[RetrievedFile]
    tests: list[RetrievedFile]
    mentions: Mentions
    focus_symbols: list[str]
    backend: str
    semantic: bool
    stats: dict = field(default_factory=dict)
    candidates: list[RetrievedFile] = field(default_factory=list)  # full ranking (diagnostics)

    def paths(self) -> list[str]:
        return [f.path for f in self.files]

    def rank_of(self, path: str) -> int | None:
        for i, f in enumerate(self.candidates, 1):
            if f.path == path:
                return i
        return None

    def to_dict(self) -> dict:
        return {
            "files": [f.to_dict() for f in self.files],
            "tests": [f.to_dict() for f in self.tests],
            "mentions": self.mentions.to_dict(),
            "focus_symbols": self.focus_symbols,
            "embedding_backend": self.backend,
            "embedding_semantic": self.semantic,
            "stats": self.stats,
            "ranking_top20": [(f.path, round(f.score, 3)) for f in self.candidates[:20]],
        }


class _Cand:
    __slots__ = ("path", "signals", "reasons", "symbols")

    def __init__(self, path: str) -> None:
        self.path = path
        self.signals: dict[str, float] = {}
        self.reasons: list[str] = []
        self.symbols: dict[str, float] = {}

    def bump(self, signal: str, value: float, reason: str | None = None, symbol: str | None = None,
             sym_weight: float = 1.0) -> None:
        if value > self.signals.get(signal, 0.0):
            self.signals[signal] = value
        if reason and reason not in self.reasons:
            self.reasons.append(reason)
        if symbol:
            self.symbols[symbol] = max(self.symbols.get(symbol, 0.0), sym_weight)

    def stage1(self) -> float:
        return sum(WEIGHTS[k] * v for k, v in self.signals.items()
                   if k in ("explicit", "traceback", "symbol_def", "lexical", "semantic", "hint"))

    def total(self) -> float:
        return sum(WEIGHTS[k] * v for k, v in self.signals.items())


class RepositoryRetriever:
    def __init__(
        self,
        root: Path,
        embedder: EmbeddingBackend | None = None,
        cache: EmbeddingCache | None = None,
        index: RepoIndex | None = None,
        use_embeddings: bool = True,
    ) -> None:
        self.root = Path(root)
        self.index = index or RepoIndex.build(self.root)
        self.use_embeddings = use_embeddings
        self.embedder = embedder if embedder is not None else (LocalNgramEmbedder() if use_embeddings else None)
        self.cache = cache or EmbeddingCache()
        self._chunks: list[tuple[str, str | None, str]] | None = None   # (path, qualname, text)
        self._chunk_vecs: list[list[float]] | None = None
        self._bm25: BM25 | None = None
        self._bm25_paths: list[str] = []
        self.embedding_error: str | None = None

    # ------------------------------------------------------------------
    # Lazily built channels
    # ------------------------------------------------------------------
    def _lexical(self) -> tuple[BM25, list[str]]:
        if self._bm25 is None:
            paths = self.index.source_paths() or list(self.index.files)
            self._bm25_paths = paths
            self._bm25 = BM25([file_tokens(self.index.files[p]) for p in paths])
        return self._bm25, self._bm25_paths

    def _ensure_chunks(self) -> None:
        if self._chunks is not None or not self.embedder:
            return
        chunks: list[tuple[str, str | None, str]] = []
        for path in self.index.source_paths():
            fi = self.index.files[path]
            head = "\n".join(fi.lines[:40])
            chunks.append((path, None, f"{path}\n{head}"[:1500]))
            for sym in fi.symbols:
                if sym.kind == "constant":
                    continue
                chunks.append((path, sym.qualname, symbol_document(self.index, sym)))
        self._chunks = chunks
        texts = [c[2] for c in chunks]
        try:
            if isinstance(self.embedder, LocalNgramEmbedder):
                self.embedder.fit(texts)
            self._chunk_vecs = self.cache.embed(self.embedder, texts, "passage")
        except Exception as exc:  # embedding service down → degrade gracefully, record why
            self.embedding_error = f"{type(exc).__name__}: {exc}"[:300]
            self._chunk_vecs = None

    # ------------------------------------------------------------------
    def retrieve(
        self,
        title: str,
        body: str,
        top_n: int = 6,
        hint_paths: list[str] | None = None,
        hint_symbols: list[str] | None = None,
        max_tests: int = 4,
    ) -> RetrievalResult:
        text = f"{title}\n\n{body or ''}"
        idx = self.index
        mentions = extract_mentions(text, idx.all_paths)
        cands: dict[str, _Cand] = {}

        def cand(path: str) -> _Cand:
            if path not in cands:
                cands[path] = _Cand(path)
            return cands[path]

        always: list[str] = []

        # 1a. stack-trace frames (Python prints outermost first → last frame is where it raised)
        frames = mentions.frames
        for i, fr in enumerate(frames):
            depth_weight = 1.0 if i == len(frames) - 1 else 0.75
            sym = idx.symbol_at(fr.path, fr.line) if fr.line else None
            where = f"line {fr.line}" + (f" in {fr.function}()" if fr.function else "")
            cand(fr.path).bump("traceback", depth_weight, f"stack trace frame: {where}",
                               sym.qualname if sym else None, 1.0)
            if fr.path not in always:
                always.append(fr.path)
        # 1b. explicit file mentions
        frame_paths = {f.path for f in frames}
        for p in mentions.paths:
            if p not in frame_paths:
                cand(p).bump("explicit", 1.0, "file named in the issue text")
                if p not in always:
                    always.append(p)

        # 2. symbol definitions for mentioned identifiers
        focus: dict[str, float] = {}
        idents = list(mentions.identifiers) + [s for s in (hint_symbols or []) if s]
        for ident in idents:
            syms = [s for s in idx.lookup(ident) if s.kind != "constant" or ident.isupper()]
            if not syms or len(syms) > 6:
                continue
            exact = any(s.qualname == ident or s.name == ident for s in syms)
            strength = (1.0 if exact else 0.7) / math.sqrt(len(syms))
            from_hint = ident in (hint_symbols or []) and ident not in mentions.identifiers
            for s in syms:
                label = f"defines `{s.qualname}`" + (" (suggested by plan/triage)" if from_hint else " (mentioned in issue)")
                cand(s.path).bump("hint" if from_hint else "symbol_def", strength, label, s.qualname, strength)
                focus[s.key] = max(focus.get(s.key, 0.0), strength)

        for hp in hint_paths or []:
            if hp in idx.files:
                cand(hp).bump("hint", 1.0, "suggested by triage/plan")

        # 3a. lexical (BM25)
        query = tokenize(title) * 2 + tokenize(body or "")[:400]
        bm25, bm_paths = self._lexical()
        lex_scores = bm25.scores(query) if query else []
        top_lex = max(lex_scores) if lex_scores else 0.0
        if top_lex > 0:
            ranked = sorted(range(len(bm_paths)), key=lambda i: -lex_scores[i])[:15]
            for i in ranked:
                norm = lex_scores[i] / top_lex
                if norm < 0.15:
                    break
                terms = bm25.matched_terms(query, i)
                cand(bm_paths[i]).bump("lexical", norm, f"lexical match on {', '.join(terms)} (BM25 {norm:.2f})")

        # 3b. embeddings (max over chunks per file)
        backend_name = getattr(self.embedder, "name", "none") if self.embedder else "none"
        semantic = bool(getattr(self.embedder, "semantic", False))
        if self.embedder:
            self._ensure_chunks()
        if self.embedder and self._chunk_vecs:
            try:
                qv = self.cache.embed(self.embedder, [text[:3000]], "query")[0]
                per_file: dict[str, tuple[float, str | None]] = {}
                for (path, qual, _), v in zip(self._chunks or [], self._chunk_vecs):
                    s = cosine(qv, v)
                    if s > per_file.get(path, (-1.0, None))[0]:
                        per_file[path] = (s, qual)
                sims = sorted(per_file.values(), key=lambda x: -x[0])
                if sims:
                    top_sim = sims[0][0]
                    median = sims[len(sims) // 2][0]
                    span = (top_sim - median) or 1e-6
                    for path, (s, qual) in sorted(per_file.items(), key=lambda kv: -kv[1][0])[:15]:
                        norm = max(0.0, min(1.0, (s - median) / span))
                        if norm < 0.2:
                            continue
                        kind = "semantic" if semantic else "n-gram vector"
                        cand(path).bump("semantic", norm,
                                        f"{kind} similarity {s:.2f} ({backend_name})"
                                        + (f" via `{qual}`" if qual else ""),
                                        qual, 0.5 * norm)
            except Exception as exc:
                self.embedding_error = f"{type(exc).__name__}: {exc}"[:300]

        # Focus symbols from strong lexical/semantic seeds: best symbol-level match per seed file.
        seeds = sorted(cands.values(), key=lambda c: -c.stage1())[:8]
        max_seed = max((c.stage1() for c in seeds), default=0.0) or 1.0
        for c in seeds[:4]:
            for qual, w in c.symbols.items():
                for s in idx.files.get(c.path).symbols if c.path in idx.files else []:
                    if s.qualname == qual:
                        focus[s.key] = max(focus.get(s.key, 0.0), w * c.stage1() / max_seed)
            if not c.symbols and c.path in idx.files and query:
                best = self._best_symbol(c.path, query)
                if best is not None:
                    c.symbols[best.qualname] = 0.4
                    focus[best.key] = max(focus.get(best.key, 0.0), 0.4 * c.stage1() / max_seed)

        # 4. graph expansion
        sym_by_key = {s.key: s for fi in idx.files.values() for s in fi.symbols}
        for key, fstrength in sorted(focus.items(), key=lambda kv: -kv[1])[:12]:
            s = sym_by_key.get(key)
            if not s:
                continue
            seed_strength = min(1.0, fstrength * (cands[s.path].stage1() / max_seed if s.path in cands else 0.5) + 0.3 * fstrength)
            # callees
            for r in s.refs:
                if r in COMMON_CODE_NAMES or len(r) < 3:
                    continue
                defs = [d for d in idx.defs.get(r, []) if d.path != s.path and d.kind != "constant"]
                if not defs or len(defs) > 3:
                    continue
                for d in defs:
                    cand(d.path).bump("callee", seed_strength / len(defs),
                                      f"`{s.qualname}` ({s.path}) uses `{d.qualname}` defined here",
                                      d.qualname, 0.6 * seed_strength)
            # callers
            if s.name in COMMON_CODE_NAMES or s.kind == "constant":
                continue
            callers = {(p, q) for p, q in idx.callers(s.name) if p != s.path}
            caller_files = {p for p, _ in callers}
            if not callers or len(caller_files) > 12:
                continue
            damp = 1.0 / math.sqrt(len(caller_files))
            for p, q in sorted(callers):
                cand(p).bump("caller", seed_strength * damp,
                             f"references `{s.qualname}`" + (f" from `{q}`" if q else " at module level"),
                             q or None, 0.5 * seed_strength)

        for c in seeds[:5]:
            strength = c.stage1() / max_seed
            for imp in idx.imports.get(c.path, ()):
                cand(imp).bump("imports", 0.8 * strength, f"imported by {c.path}")
            for imp in idx.importers.get(c.path, ()):
                cand(imp).bump("imported_by", 0.8 * strength, f"imports {c.path}")

        # Ranking
        ranked_all = sorted(cands.values(), key=lambda c: (-c.total(), c.path))
        result_files: list[RetrievedFile] = []
        test_files: list[RetrievedFile] = []
        always = always[:MAX_ALWAYS]

        def to_rf(c: _Cand) -> RetrievedFile:
            fi = idx.files.get(c.path)
            syms = [q for q, _ in sorted(c.symbols.items(), key=lambda kv: -kv[1])][:8]
            return RetrievedFile(c.path, c.total(), c.reasons[:6], dict(c.signals), syms,
                                 bool(fi.is_test) if fi else False)

        all_rf = [to_rf(c) for c in ranked_all]
        top_score = next((rf.score for rf in all_rf if not rf.is_test), 0.0) or 1.0
        for rf in all_rf:
            if rf.path in always:
                if rf.is_test and not rf.signals.get("traceback"):
                    test_files.append(rf)
                else:
                    result_files.append(rf)
        for rf in all_rf:
            if rf.path in always:
                continue
            if rf.is_test:
                if len(test_files) < max_tests:
                    test_files.append(rf)
                continue
            if len(result_files) >= max(top_n, len([p for p in always])):
                continue
            if rf.score < CUTOFF_RATIO * top_score:
                continue
            if rf.path not in idx.files or idx.files[rf.path].language == "text":
                if not rf.signals.get("explicit"):
                    continue
            result_files.append(rf)
        result_files.sort(key=lambda r: -r.score)

        # Related tests: test files referencing focus symbol names or importing selected files.
        selected = {r.path for r in result_files}
        focus_names = {sym_by_key[k].name for k in focus if k in sym_by_key}
        for path, fi in idx.files.items():
            if not fi.is_test or any(t.path == path for t in test_files) or len(test_files) >= max_tests:
                continue
            hits = (focus_names & fi.module_refs) | {n for s in fi.symbols for n in s.refs if n in focus_names}
            if hits or (idx.imports.get(path, set()) & selected):
                reason = f"tests reference {', '.join(sorted(hits)[:3])}" if hits else "imports a selected file"
                test_files.append(RetrievedFile(path, 0.0, [reason], {}, [], True))

        focus_list = [sym_by_key[k].qualname for k, _ in sorted(focus.items(), key=lambda kv: -kv[1])
                      if k in sym_by_key][:12]
        return RetrievalResult(
            files=result_files,
            tests=test_files[:max_tests],
            mentions=mentions,
            focus_symbols=focus_list,
            backend=backend_name,
            semantic=semantic,
            stats={**idx.stats(), "candidates": len(cands),
                   **({"embedding_error": self.embedding_error} if self.embedding_error else {}),
                   **({"embedding_rejected_inputs": getattr(self.embedder, "rejected", 0)}
                      if getattr(self.embedder, "rejected", 0) else {})},
            candidates=[rf for rf in all_rf if not rf.is_test],
        )

    def _best_symbol(self, path: str, query: list[str]) -> Symbol | None:
        fi = self.index.files.get(path)
        if not fi or not fi.symbols:
            return None
        qset = set(query)
        best, best_score = None, 0.0
        for s in fi.symbols:
            toks = tokenize(f"{s.qualname} {s.signature} {s.doc} {self.index.symbol_text(s)[:800]}")
            if not toks:
                continue
            overlap = len(qset & set(toks)) / math.sqrt(len(set(toks)))
            if overlap > best_score:
                best, best_score = s, overlap
        return best

    # ------------------------------------------------------------------
    # Prompt context rendering
    # ------------------------------------------------------------------
    def render_context(self, result: RetrievalResult, budget_chars: int = 60_000,
                       extra_paths: list[str] | None = None, include_tests: int = 1,
                       live: bool = False) -> str:
        """Render selected files for an LLM prompt: whole files when they fit, otherwise an
        outline plus the full text of the focus symbols (always real, contiguous source).
        ``live=True`` re-reads every file from disk so the prompt reflects edits made so far."""
        entries: list[tuple[str, list[str], list[str]]] = [(f.path, f.reasons, f.symbols) for f in result.files]
        for p in extra_paths or []:   # plan output (LLM-written): must stay inside the clone
            if (p in self.index.files or safe_repo_file(self.root, p)) and all(p != e[0] for e in entries):
                entries.insert(0, (p, ["file the plan expects to change"], []))
        for t in result.tests[:include_tests]:
            if all(t.path != e[0] for e in entries):
                entries.append((t.path, t.reasons, []))
        parts: list[str] = []
        remaining = budget_chars
        for i, (path, reasons, symbols) in enumerate(entries):
            fi = self.index.files.get(path)
            if fi is None or live:
                # Re-read from disk: test code may have swapped a file for a symlink since indexing.
                p = safe_repo_file(self.root, path)
                if p is None:
                    continue
                try:
                    text = p.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
            else:
                text = fi.text
            share = max(2000, remaining // max(1, len(entries) - i))
            why = "; ".join(reasons[:2])
            if len(text) <= share * 1.3:
                # No trailing newline inside the fence: an extra "\n" renders as a phantom empty
                # last line that models copy into SEARCH (which then cannot match at EOF).
                shown = text.rstrip("\n")
                block = f"### {path}\n(why: {why})\n```\n{shown}\n```"
            else:
                block = self._excerpt(path, text, symbols, share, why)
            parts.append(block)
            remaining -= len(block)
            if remaining <= 1500:
                break
        return "\n\n".join(parts) if parts else "(no relevant files found)"

    def _excerpt(self, path: str, text: str, symbols: list[str], budget: int, why: str) -> str:
        fi = self.index.files.get(path)
        lines = text.splitlines()
        outline = []
        spans: list[tuple[int, int]] = []
        if fi:
            for s in fi.symbols:
                if s.kind in ("class", "function", "method"):
                    outline.append(f"  L{s.start}-{s.end}: {s.signature or s.qualname}")
                if s.qualname in symbols:
                    spans.append((s.start, s.end))
        header_end = next((i for i, ln in enumerate(lines) if ln.startswith(("def ", "class ", "async def "))), min(30, len(lines)))
        spans.insert(0, (1, max(1, header_end)))
        spans.sort()
        merged: list[list[int]] = []
        for a, b in spans:
            if merged and a <= merged[-1][1] + 3:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        body_parts, used = [], 0
        for a, b in merged:
            chunk = "\n".join(lines[a - 1:b])
            if used + len(chunk) > budget and body_parts:
                break
            body_parts.append(f"# --- lines {a}-{b} ---\n{chunk}")
            used += len(chunk)
        return (
            f"### {path}\n(why: {why}; file has {len(lines)} lines — EXCERPT: outline + relevant symbols; "
            f"SEARCH blocks must copy text exactly from the lines shown)\nOutline:\n" + "\n".join(outline[:60])
            + "\n```\n" + "\n".join(body_parts) + "\n```"
        )


def retrieval_metrics(result: RetrievalResult, expected_files: list[str], expected_symbols: list[str],
                      ks: tuple[int, ...] = (1, 3, 5)) -> dict:
    """Recall@k / MRR / symbol recall of one retrieval against ground truth."""
    ranked = [f.path for f in result.files]
    out: dict = {}
    exp = [e for e in expected_files]
    for k in ks:
        top = set(ranked[:k])
        out[f"recall@{k}"] = (sum(1 for e in exp if e in top) / len(exp)) if exp else None
    rr = 0.0
    for i, p in enumerate(ranked, 1):
        if p in exp:
            rr = 1.0 / i
            break
    out["mrr"] = rr if exp else None
    out["all_found"] = all(e in ranked for e in exp) if exp else None
    out["returned"] = len(ranked)
    if expected_symbols:
        got = set(result.focus_symbols) | {s for f in result.files for s in f.symbols}
        got_short = {g.split(".")[-1] for g in got} | got
        hit = sum(1 for s in expected_symbols if s in got or s.split(".")[-1] in got_short)
        out["symbol_recall"] = hit / len(expected_symbols)
    else:
        out["symbol_recall"] = None
    return out
