"""
redaction.py — Remove credentials from text before it is logged, recorded or posted.

Two layers:
  * exact values registered at runtime (``register_secret``) — the API keys and tokens this
    process was configured with — are replaced wherever they appear, also base64-encoded
    (HTTP basic auth headers carry ``x-access-token:<token>`` in base64);
  * well-known credential shapes (GitHub tokens, NVIDIA/OpenAI/Anthropic/Google keys, bearer
    and basic auth headers, ``key=``/``token=`` query parameters, credentials in URLs).

This is a safety net for error messages and tool output, not a guarantee: callers should still
avoid putting secrets into text in the first place.
"""

from __future__ import annotations

import base64
import re
import threading

_lock = threading.Lock()
_secrets: set[str] = set()

_PLACEHOLDER = "[REDACTED]"

_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bnvapi-[A-Za-z0-9_\-]{20,}"),
    re.compile(r"\bsk-(?:ant-)?[A-Za-z0-9_\-]{20,}"),
    re.compile(r"\bAIza[0-9A-Za-z_\-]{30,}"),
    re.compile(r"(?i)\b(authorization\s*:\s*(?:bearer|basic|token)\s+)[A-Za-z0-9._~+/=\-]{8,}"),
    re.compile(r"(?i)\b(bearer\s+)[A-Za-z0-9._~+/=\-]{16,}"),
    re.compile(r"(?i)([?&](?:key|api_key|apikey|access_token|token)=)[^&\s'\"]+"),
    re.compile(r"(?i)(https?://)[^/\s:@]+:[^/\s@]+@"),          # user:password@host
]


def register_secret(value: str | None) -> None:
    """Remember a credential value so ``redact`` removes it (and its base64 forms) everywhere."""
    if not value or len(value) < 8:
        return
    with _lock:
        _secrets.add(value)
        for prefix in ("x-access-token:", ""):
            _secrets.add(base64.b64encode(f"{prefix}{value}".encode()).decode())


def _sub(match: re.Match[str]) -> str:
    # Keep the label part of the patterns that capture one (e.g. "Authorization: Bearer ").
    return (match.group(1) if match.re.groups else "") + _PLACEHOLDER


def redact(text: str | None) -> str:
    if not text:
        return text or ""
    with _lock:
        values = sorted(_secrets, key=len, reverse=True)
    for v in values:
        if v in text:
            text = text.replace(v, _PLACEHOLDER)
    for pat in _PATTERNS:
        text = pat.sub(_sub, text)
    return text
