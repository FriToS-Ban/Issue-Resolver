"""
file_selector.py — Score each file in a repo for relevance to an issue.

Approach:
  - Extract keywords from the issue title + body
  - Score each file path and a snippet of its content against those keywords
  - Return the top-N files (path + truncated content) for use as LLM context
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


# File extensions worth reading
_TEXT_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".rb", ".rs",
    ".c", ".cpp", ".h", ".hpp", ".cs", ".php", ".swift", ".kt", ".scala",
    ".md", ".rst", ".txt", ".yaml", ".yml", ".json", ".toml", ".cfg",
    ".ini", ".sh", ".bash", ".html", ".css",
}

# Paths almost never relevant — skip to save tokens
_SKIP_PATTERNS = re.compile(
    r"(node_modules|\.git|__pycache__|\.pytest_cache|dist/|build/|"
    r"\.egg-info|venv|\.venv|coverage|\.nyc_output)",
    re.IGNORECASE,
)

_MAX_FILE_CHARS = 100_000   # max chars to read per file for scoring/context
_MAX_SNIPPET_CHARS = 50_000  # chars included in the LLM prompt per file



@dataclass
class RelevantFile:
    path: str
    score: float
    snippet: str  # truncated content for LLM context


def select_relevant_files(
    issue_title: str,
    issue_body: str,
    repo_root: Path,
    top_n: int = 5,
) -> list[RelevantFile]:
    """
    Walk `repo_root` and return the `top_n` most relevant files.
    """
    keywords = _extract_keywords(issue_title + " " + (issue_body or ""))
    if not keywords:
        return []

    candidates: list[RelevantFile] = []

    for path in repo_root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(repo_root).as_posix()
        if _SKIP_PATTERNS.search(rel):
            continue
        if path.suffix.lower() not in _TEXT_EXTENSIONS:
            continue

        try:
            content = path.read_text(encoding="utf-8", errors="replace")[:_MAX_FILE_CHARS]
        except (OSError, PermissionError):
            continue

        score = _score(rel, content, keywords)
        if score > 0:
            candidates.append(
                RelevantFile(
                    path=rel,
                    score=score,
                    snippet=content[:_MAX_SNIPPET_CHARS],
                )
            )

    candidates.sort(key=lambda f: f.score, reverse=True)
    return candidates[:top_n]


def format_for_prompt(files: list[RelevantFile]) -> str:
    """
    Render the selected files into a block suitable for inclusion in an LLM prompt.
    """
    if not files:
        return "(no relevant files found)"
    parts: list[str] = []
    for f in files:
        parts.append(f"### {f.path}\n```\n{f.snippet}\n```")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

_STOP_WORDS = {
    "the", "a", "an", "is", "it", "in", "on", "at", "of", "to", "and",
    "or", "but", "for", "with", "this", "that", "not", "be", "are",
    "was", "has", "have", "from", "by", "as", "i", "we", "you", "they",
    "error", "issue", "bug", "problem", "when", "how", "can", "should",
}


def _extract_keywords(text: str) -> set[str]:
    tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{2,}", text)
    return {t.lower() for t in tokens if t.lower() not in _STOP_WORDS}


def _score(rel_path: str, content: str, keywords: set[str]) -> float:
    """
    Simple TF-style scoring:
      - +3 for each keyword hit in the file *path*
      - +1 for each keyword hit in the file *content* (capped per keyword)
    """
    path_lower = rel_path.lower()
    content_lower = content.lower()
    score = 0.0
    for kw in keywords:
        if kw in path_lower:
            score += 3.0
        hits = content_lower.count(kw)
        if hits:
            score += min(hits, 5)  # cap at 5 per keyword to avoid log-file dominance
    return score
