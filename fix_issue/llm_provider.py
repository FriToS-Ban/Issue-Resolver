"""
llm_provider.py — Provider-neutral LLM abstraction.

All prompts throughout the codebase are plain strings.
Each provider implementation translates them into the correct wire format
using plain `requests` — no vendor SDK.

Supported providers (set via LLM_PROVIDER config key):
    anthropic           — Anthropic Messages API
    openai_compatible   — OpenAI chat-completions schema (OpenAI, Ollama, NVIDIA NIM, LM Studio …)
    gemini              — Google Generative Language API (different schema, own class)

Transport errors, HTTP 429 and 5xx responses are retried a bounded number of times with
exponential backoff. ``RecordingLLM`` wraps any provider to make every call inspectable.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Callable, Iterator

import requests

_TIMEOUT = 120  # seconds per API call
_RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}
_MAX_TRANSPORT_RETRIES = 3


class LLMError(Exception):
    """Raised when an LLM API call fails after retries."""


class LLMProvider(ABC):
    """Abstract interface — one method, provider-neutral prompts."""

    @abstractmethod
    def generate(self, prompt: str, system: str | None = None) -> str:
        """
        Send `prompt` (and optional `system` instruction) to the LLM.
        Returns the model's text response as a plain string.
        """


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------

class AnthropicProvider(LLMProvider):
    """
    Calls POST https://api.anthropic.com/v1/messages.
    Uses the Messages API format directly via `requests`.
    """

    BASE_URL = "https://api.anthropic.com/v1/messages"
    API_VERSION = "2023-06-01"

    def __init__(
        self,
        api_key: str,
        model: str = "claude-opus-4-5",
        temperature: float | None = None,
        max_tokens: int = 4096,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    def generate(self, prompt: str, system: str | None = None) -> str:
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": self.API_VERSION,
            "content-type": "application/json",
        }
        body: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            body["system"] = system
        if self._temperature is not None:
            body["temperature"] = self._temperature

        resp = _post_with_retries(
            lambda: requests.post(self.BASE_URL, headers=headers, json=body, timeout=_TIMEOUT),
            "Anthropic",
        )
        data = resp.json()
        return "".join(
            block.get("text", "") for block in data.get("content", []) if block.get("type", "text") == "text"
        )


# ---------------------------------------------------------------------------
# OpenAI-compatible  (OpenAI, Ollama, NVIDIA NIM, LM Studio, …)
# ---------------------------------------------------------------------------

class OpenAICompatibleProvider(LLMProvider):
    """
    Calls POST {base_url}/chat/completions using the OpenAI chat-completions schema.
    Works with any server that speaks that schema — no OpenAI SDK required.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        base_url: str = "https://api.openai.com/v1",
        temperature: float | None = None,
        max_tokens: int = 4096,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._temperature = temperature
        self._max_tokens = max_tokens

    def generate(self, prompt: str, system: str | None = None) -> str:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        body: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "max_tokens": self._max_tokens,
        }
        if self._temperature is not None:
            body["temperature"] = self._temperature

        url = f"{self._base_url}/chat/completions"
        resp = _post_with_retries(
            lambda: requests.post(url, headers=headers, json=body, timeout=_TIMEOUT),
            "OpenAI-compatible",
        )
        data = resp.json()
        try:
            content = data["choices"][0]["message"].get("content")
        except (KeyError, IndexError, TypeError, AttributeError) as e:
            raise LLMError(f"OpenAI-compatible API returned an unexpected body: {str(data)[:300]}") from e
        if not content:
            raise LLMError(
                "OpenAI-compatible API returned an empty message (possibly truncated by max_tokens)."
            )
        return content


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------

class GeminiProvider(LLMProvider):
    """
    Calls the Google Generative Language API.
    Uses a different schema to OpenAI — separate class, not a base_url hack.
    """

    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-pro",
        temperature: float | None = None,
        max_tokens: int = 4096,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    def generate(self, prompt: str, system: str | None = None) -> str:
        url = f"{self.BASE_URL}/{self._model}:generateContent?key={self._api_key}"
        contents: list[dict[str, Any]] = [
            {"role": "user", "parts": [{"text": prompt}]}
        ]
        body: dict[str, Any] = {"contents": contents}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        gen_cfg: dict[str, Any] = {"maxOutputTokens": self._max_tokens}
        if self._temperature is not None:
            gen_cfg["temperature"] = self._temperature
        body["generationConfig"] = gen_cfg

        resp = _post_with_retries(lambda: requests.post(url, json=body, timeout=_TIMEOUT), "Gemini")
        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_provider(
    provider: str,
    api_key: str,
    model: str,
    base_url: str = "",
    temperature: float | None = None,
    max_tokens: int = 4096,
) -> LLMProvider:
    """Instantiate the correct provider from config values."""
    p = provider.lower()
    kw: dict[str, Any] = {"temperature": temperature, "max_tokens": max_tokens}
    if p == "anthropic":
        return AnthropicProvider(api_key=api_key, model=model, **kw)
    if p == "openai_compatible":
        if not base_url:
            base_url = "https://api.openai.com/v1"
        return OpenAICompatibleProvider(api_key=api_key, model=model, base_url=base_url, **kw)
    if p == "gemini":
        return GeminiProvider(api_key=api_key, model=model, **kw)
    raise ValueError(
        f"Unknown LLM_PROVIDER '{provider}'. "
        "Valid options: anthropic | openai_compatible | gemini"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _raise_for_llm_error(resp: requests.Response, provider_name: str) -> None:
    if not resp.ok:
        try:
            detail = json.dumps(resp.json(), indent=2)
        except Exception:
            detail = resp.text[:500]
        raise LLMError(
            f"{provider_name} API returned HTTP {resp.status_code}:\n{detail}"
        )


def _post_with_retries(send: Callable[[], requests.Response], provider_name: str) -> requests.Response:
    """POST with bounded retries on transport errors, 429 and 5xx (exponential backoff)."""
    for attempt in range(_MAX_TRANSPORT_RETRIES + 1):
        try:
            resp = send()
        except requests.exceptions.RequestException as e:
            if attempt == _MAX_TRANSPORT_RETRIES:
                raise LLMError(f"{provider_name} API request failed ({type(e).__name__}): {e}") from e
            time.sleep(min(30.0, 2.0 * 2 ** attempt))
            continue
        if resp.status_code in _RETRY_STATUS and attempt < _MAX_TRANSPORT_RETRIES:
            retry_after = resp.headers.get("retry-after") if resp.headers else None
            try:
                delay = float(retry_after) if retry_after else 2.0 * 2 ** attempt
            except ValueError:
                delay = 2.0 * 2 ** attempt
            time.sleep(min(60.0, delay))
            continue
        _raise_for_llm_error(resp, provider_name)
        return resp
    raise LLMError(f"{provider_name} API request failed after retries")  # pragma: no cover


# ---------------------------------------------------------------------------
# Observability wrapper
# ---------------------------------------------------------------------------

class RecordingLLM(LLMProvider):
    """
    Wraps any provider and records every call (purpose tag, sizes, latency, errors).
    Prompt tails and responses are kept (truncated) so a run can be inspected afterwards.
    """

    def __init__(self, inner: LLMProvider, keep_chars: int = 6000) -> None:
        self.inner = inner
        self.calls: list[dict[str, Any]] = []
        self._tag = "untagged"
        self._keep = keep_chars

    @contextmanager
    def tag(self, purpose: str) -> Iterator[None]:
        previous, self._tag = self._tag, purpose
        try:
            yield
        finally:
            self._tag = previous

    def generate(self, prompt: str, system: str | None = None) -> str:
        start = time.monotonic()
        entry: dict[str, Any] = {
            "purpose": self._tag,
            "prompt_chars": len(prompt) + len(system or ""),
            "prompt_tail": prompt[-self._keep:],
        }
        try:
            out = self.inner.generate(prompt, system=system)
        except Exception as e:
            entry.update(error=f"{type(e).__name__}: {e}"[:500], seconds=round(time.monotonic() - start, 2))
            self.calls.append(entry)
            raise
        entry.update(
            response_chars=len(out or ""),
            response=(out or "")[: self._keep],
            seconds=round(time.monotonic() - start, 2),
        )
        self.calls.append(entry)
        return out

    def summary(self) -> dict[str, Any]:
        by: dict[str, int] = {}
        for c in self.calls:
            by[c["purpose"]] = by.get(c["purpose"], 0) + 1
        return {
            "calls": len(self.calls),
            "by_purpose": by,
            "errors": sum(1 for c in self.calls if "error" in c),
            "seconds": round(sum(c.get("seconds", 0) for c in self.calls), 1),
            "prompt_chars": sum(c.get("prompt_chars", 0) for c in self.calls),
            "response_chars": sum(c.get("response_chars", 0) for c in self.calls),
        }


def tagged(llm: LLMProvider, purpose: str):
    """Context manager: tag calls on a RecordingLLM; a no-op for other providers."""
    if isinstance(llm, RecordingLLM):
        return llm.tag(purpose)
    from contextlib import nullcontext
    return nullcontext()
