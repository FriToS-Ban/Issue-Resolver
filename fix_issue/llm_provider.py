"""
llm_provider.py — Provider-neutral LLM abstraction.

All prompts throughout the codebase are plain strings.
Each provider implementation translates them into the correct wire format
using plain `requests` — no vendor SDK.

Supported providers (set via LLM_PROVIDER config key):
    anthropic           — Anthropic Messages API
    openai_compatible   — OpenAI chat-completions schema (OpenAI, Ollama, NVIDIA NIM, LM Studio …)
    gemini              — Google Generative Language API (different schema, own class)
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any

import requests

_TIMEOUT = 120  # seconds per API call


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

    def __init__(self, api_key: str, model: str = "claude-opus-4-5") -> None:
        self._api_key = api_key
        self._model = model

    def generate(self, prompt: str, system: str | None = None) -> str:
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": self.API_VERSION,
            "content-type": "application/json",
        }
        body: dict[str, Any] = {
            "model": self._model,
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            body["system"] = system

        try:
            resp = requests.post(self.BASE_URL, headers=headers, json=body, timeout=_TIMEOUT)
        except requests.exceptions.RequestException as e:
            raise LLMError(f"Anthropic API request failed ({type(e).__name__}): {e}") from e
        _raise_for_llm_error(resp, "Anthropic")
        data = resp.json()
        return data["content"][0]["text"]


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
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")

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
            "max_tokens": 4096,
        }

        url = f"{self._base_url}/chat/completions"
        try:
            resp = requests.post(url, headers=headers, json=body, timeout=_TIMEOUT)
        except requests.exceptions.RequestException as e:
            raise LLMError(f"OpenAI-compatible API request failed ({type(e).__name__}): {e}") from e
        _raise_for_llm_error(resp, "OpenAI-compatible")
        data = resp.json()
        return data["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------

class GeminiProvider(LLMProvider):
    """
    Calls the Google Generative Language API.
    Uses a different schema to OpenAI — separate class, not a base_url hack.
    """

    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(self, api_key: str, model: str = "gemini-2.5-pro") -> None:
        self._api_key = api_key
        self._model = model

    def generate(self, prompt: str, system: str | None = None) -> str:
        url = f"{self.BASE_URL}/{self._model}:generateContent?key={self._api_key}"
        contents: list[dict[str, Any]] = [
            {"role": "user", "parts": [{"text": prompt}]}
        ]
        body: dict[str, Any] = {"contents": contents}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}

        try:
            resp = requests.post(url, json=body, timeout=_TIMEOUT)
        except requests.exceptions.RequestException as e:
            raise LLMError(f"Gemini API request failed ({type(e).__name__}): {e}") from e
        _raise_for_llm_error(resp, "Gemini")
        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]



# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_provider(provider: str, api_key: str, model: str, base_url: str = "") -> LLMProvider:
    """Instantiate the correct provider from config values."""
    p = provider.lower()
    if p == "anthropic":
        return AnthropicProvider(api_key=api_key, model=model)
    if p == "openai_compatible":
        if not base_url:
            base_url = "https://api.openai.com/v1"
        return OpenAICompatibleProvider(api_key=api_key, model=model, base_url=base_url)
    if p == "gemini":
        return GeminiProvider(api_key=api_key, model=model)
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
