"""
patch_generator.py — Generate a unified diff that fixes a GitHub issue using Search/Replace blocks.

Takes an LLMProvider instance — fully provider-neutral. No direct API imports.
"""

from __future__ import annotations

import difflib
import re
from pathlib import Path
from typing import Any

from fix_issue.llm_provider import LLMError, LLMProvider


_SYSTEM = """\
You are an expert software engineer. Your job is to produce a minimal, correct patch that fixes the reported GitHub issue.

STRICT OUTPUT FORMAT:
Output your changes as one or more SEARCH/REPLACE blocks.

Format for each block:
FILE: <relative_path_to_file>
<<<<<<< SEARCH
<exact existing code block from the file>
=======
<replacement code block>
>>>>>>> REPLACE

Rules:
- The SEARCH block MUST match the existing code in the file EXACTLY line-for-line, including all whitespace and indentation.
- Keep SEARCH blocks as concise as necessary to uniquely identify the section to modify.
- Do NOT output any prose, explanations, or markdown commentary outside the SEARCH/REPLACE blocks.
- If the fix requires modifying multiple files or adding a test, use multiple SEARCH/REPLACE blocks.

If you cannot produce a correct fix with high confidence, output:
CANNOT_FIX: <one sentence reason>
"""


def generate_patch(
    llm: LLMProvider,
    issue: dict[str, Any],
    file_context: str,
    prior_failures: list[str] | None = None,
    repo_root: Path | None = None,
) -> str | None:
    """
    Ask the LLM for a fix and synthesize a clean, valid unified diff.
    """
    prompt = _build_prompt(issue, file_context, prior_failures or [])
    response = llm.generate(prompt, system=_SYSTEM)
    return _parse_response(response, repo_root, file_context)


def _build_prompt(
    issue: dict[str, Any],
    file_context: str,
    prior_failures: list[str],
) -> str:
    body = (issue.get("body") or "").strip()[:4000]
    attempt = len(prior_failures) + 1

    failure_section = ""
    if prior_failures:
        recent = prior_failures[-1][:3000]
        failure_section = f"""

### Previous Attempt Failed — Test / Execution Output
Attempt {attempt - 1} produced a patch that did not apply or make tests pass.
Here is the failure output. Use it to diagnose what went wrong and produce a better patch:

```
{recent}
```
"""

    return f"""\
Fix the following GitHub issue using SEARCH/REPLACE blocks.

### Issue #{issue.get('number')} — {issue.get('title', '')}
{body or '(no body)'}

### Relevant Source Files
{file_context}
{failure_section}
Attempt: {attempt}

Remember: output ONLY SEARCH/REPLACE blocks (or CANNOT_FIX: <reason>).
"""


def _parse_response(response: str, repo_root: Path | None, file_context: str) -> str | None:
    text = response.strip()

    if text.startswith("CANNOT_FIX"):
        return None

    # Strip markdown code fences if wrapped
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # Match SEARCH/REPLACE blocks
    pattern = re.compile(
        r"FILE:\s*(?P<path>[^\n\r]+)[\r\n]+"
        r"<<<<<<< SEARCH[\r\n]+"
        r"(?P<search>.*?)"
        r"=======[\r\n]+"
        r"(?P<replace>.*?)"
        r">>>>>>> REPLACE",
        re.DOTALL,
    )
    blocks = pattern.findall(text)

    if not blocks:
        raise LLMError(
            "Model response did not contain required SEARCH/REPLACE blocks. "
            "Output must use the format:\n"
            "FILE: <path>\n"
            "<<<<<<< SEARCH\n"
            "<exact existing code>\n"
            "=======\n"
            "<replacement code>\n"
            ">>>>>>> REPLACE"
        )

    diff_parts: list[str] = []

    for path_str, search, replace in blocks:
        rel_path = path_str.strip().replace("\\", "/")
        original_content: str | None = None

        # 1. Try reading from repo_root
        if repo_root is not None:
            target_path = repo_root / rel_path
            if target_path.exists() and target_path.is_file():
                try:
                    original_content = target_path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    pass

        # 2. Fallback: try parsing from file_context if repo_root wasn't provided
        if original_content is None:
            original_content = _extract_file_from_context(file_context, rel_path)

        if original_content is None:
            original_content = ""

        # Perform replacement (raises ValueError if search block is not found)
        new_content = _fuzzy_replace(original_content, search, replace, rel_path)

        old_lines = original_content.replace("\r\n", "\n").splitlines(keepends=True)
        new_lines = new_content.replace("\r\n", "\n").splitlines(keepends=True)

        diff_lines = list(
            difflib.unified_diff(
                old_lines,
                new_lines,
                fromfile=f"a/{rel_path}",
                tofile=f"b/{rel_path}",
            )
        )
        if diff_lines:
            part_lines: list[str] = []
            for l in diff_lines:
                if l.startswith(("---", "+++", "@@")):
                    if not l.endswith("\n"):
                        l += "\n"
                    part_lines.append(l)
                else:
                    if l.endswith("\n"):
                        part_lines.append(l)
                    else:
                        part_lines.append(l + "\n")
                        part_lines.append("\\ No newline at end of file\n")
            diff_parts.append("".join(part_lines))

    if not diff_parts:
        raise LLMError("SEARCH/REPLACE blocks produced no changes (SEARCH and REPLACE content are identical).")

    return "".join(diff_parts)


def _fuzzy_replace(content: str, search: str, replace: str, path: str) -> str:
    search_norm = search.replace("\r\n", "\n")
    content_norm = content.replace("\r\n", "\n")
    replace_norm = replace.replace("\r\n", "\n")

    if search_norm in content_norm:
        return content_norm.replace(search_norm, replace_norm, 1)

    # Line-by-line sliding window search ignoring leading/trailing space differences
    c_lines = content_norm.splitlines(keepends=True)
    s_lines = search_norm.splitlines()
    s_clean = [l.strip() for l in s_lines if l.strip()]

    if not s_clean:
        raise ValueError(f"SEARCH block for '{path}' was empty or invalid.")

    for i in range(len(c_lines) - len(s_lines) + 1):
        window = c_lines[i : i + len(s_lines)]
        w_clean = [l.strip() for l in window if l.strip()]
        if w_clean == s_clean:
            r_lines = replace_norm.splitlines(keepends=True)
            if not replace_norm.endswith("\n") and window and window[-1].endswith("\n"):
                if r_lines:
                    r_lines[-1] += "\n"
            return "".join(c_lines[:i] + r_lines + c_lines[i + len(s_lines) :])

    raise ValueError(f"SEARCH block for '{path}' could not be matched in the target file content.")


def _extract_file_from_context(file_context: str, path: str) -> str | None:
    pattern = re.compile(
        rf"###\s+{re.escape(path)}\s*\n```[^\n]*\n(?P<content>.*?)\n```",
        re.DOTALL,
    )
    match = pattern.search(file_context)
    if match:
        return match.group("content")
    return None
