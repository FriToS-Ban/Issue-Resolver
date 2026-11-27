"""Credential routing in config.py and redaction of secrets from error text."""

from __future__ import annotations

import base64

import pytest

from fix_issue import config
from fix_issue.redaction import redact, register_secret


@pytest.mark.parametrize("url,ok", [
    ("https://integrate.api.nvidia.com/v1", True),
    ("https://nvidia.com/v1", True),
    ("https://integrate.api.nvidia.com.evil.example/v1", False),
    ("https://evil.example/nvidia.com/v1", False),
    ("http://evil.example/v1?h=nvidia.com", False),
    ("http://integrate.api.nvidia.com/v1", False),          # plain http never gets the key
    ("https://notnvidia.com/v1", False),
    ("", False),
])
def test_is_nvidia_endpoint(url, ok):
    assert config.is_nvidia_endpoint(url) is ok


def _cfg(tmp_path, monkeypatch, toml: str, env: dict[str, str]) -> config.Config:
    for k in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_PROVIDER", "LLM_MODEL", "NVIDIA_API_KEY", "GITHUB_TOKEN"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    p = tmp_path / "c.toml"
    p.write_text(toml, encoding="utf-8")
    return config.load(p, dotenv=False)


def test_nvidia_key_not_sent_to_lookalike_host(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, monkeypatch, '[llm]\nprovider="openai_compatible"\n'
               'base_url="https://integrate.api.nvidia.com.evil.example/v1"\nmodel="m"\n',
               {"NVIDIA_API_KEY": "nvapi-test-value-0000000000"})
    assert cfg.llm.api_key == ""


def test_secondary_endpoints_do_not_inherit_a_foreign_key(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, monkeypatch, '[llm]\nprovider="openai_compatible"\nbase_url="https://api.example.com/v1"\n'
               'model="m"\napi_key="sk-test-llm-key-000000000000"\n'
               '[critic]\nprovider="openai_compatible"\nbase_url="https://other.example.org/v1"\nmodel="c"\n'
               '[retrieval]\nembedding_backend="openai_compatible"\nembedding_base_url="https://emb.example.net/v1"\n'
               'embedding_model="e"\n', {})
    assert config.critic_settings(cfg)[1] == ""
    assert config.embedding_settings(cfg)[2] == ""


def test_same_endpoint_reuses_the_llm_key(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, monkeypatch, '[llm]\nprovider="openai_compatible"\nbase_url="https://integrate.api.nvidia.com/v1"\n'
               'model="m"\n[critic]\nmodel="c"\n[retrieval]\nembedding_backend="openai_compatible"\nembedding_model="e"\n',
               {"NVIDIA_API_KEY": "nvapi-test-value-1111111111"})
    assert cfg.llm.api_key == "nvapi-test-value-1111111111"
    assert config.critic_settings(cfg)[1] == cfg.llm.api_key
    assert config.embedding_settings(cfg)[2] == cfg.llm.api_key


def test_dotenv_only_loads_fix_issue_settings(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("GIT_SSH_COMMAND=evil\nPATH=/evil\nFIX_ISSUE_PROFILE=baseline\nNVIDIA_API_KEY=nvapi-x\n",
                   encoding="utf-8")
    for k in ("GIT_SSH_COMMAND", "FIX_ISSUE_PROFILE", "NVIDIA_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    loaded = config.load_dotenv(env)
    assert sorted(loaded) == ["FIX_ISSUE_PROFILE", "NVIDIA_API_KEY"]
    import os
    assert "GIT_SSH_COMMAND" not in os.environ


def test_toml_string_false_is_false(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, monkeypatch, '[pipeline]\ncritique="false"\n', {})
    assert cfg.pipeline.critique is False


def test_redact_registered_secret_and_its_basic_auth_form():
    token = "ghp_" + "a1B2c3D4e5" * 4
    register_secret(token)
    basic = base64.b64encode(f"x-access-token:{token}".encode()).decode()
    text = f"git push https://x-access-token:{token}@github.com/o/r failed; header AUTHORIZATION: basic {basic}"
    out = redact(text)
    assert token not in out and basic not in out and "[REDACTED]" in out


def test_redact_known_shapes():
    out = redact("GET https://generativelanguage.googleapis.com/v1/m:generateContent?key=AIzaSyA-THIS-IS-NOT-REAL-000000"
                 " Authorization: Bearer abcdefghijklmnopqrstuvwxyz nvapi-ABCDEFGHIJKLMNOPQRSTUVWXYZ012345")
    assert "AIzaSyA" not in out and "abcdefghijklmnop" not in out and "nvapi-ABCDEF" not in out


def test_gemini_key_travels_in_a_header_not_the_url(monkeypatch):
    import requests

    from fix_issue.llm_provider import GeminiProvider

    seen = {}

    class R:
        status_code, ok, headers = 200, True, {}

        def json(self):
            return {"candidates": [{"content": {"parts": [{"text": "hi"}]}}]}

    def fake_post(url, json=None, headers=None, timeout=None):
        seen.update(url=url, headers=headers or {})
        return R()

    monkeypatch.setattr(requests, "post", fake_post)
    assert GeminiProvider(api_key="AIza-test-key-not-real").generate("x") == "hi"
    assert "key=" not in seen["url"] and seen["headers"].get("x-goog-api-key") == "AIza-test-key-not-real"


def test_critic_on_another_provider_gets_neither_llm_endpoint_nor_key(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, monkeypatch, '[llm]\nprovider="openai_compatible"\nbase_url="https://integrate.api.nvidia.com/v1"\n'
               'model="m"\n[critic]\nprovider="anthropic"\nmodel="c"\n',
               {"NVIDIA_API_KEY": "nvapi-test-value-2222222222"})
    provider, key, base, _ = config.critic_settings(cfg)
    assert provider == "anthropic" and key == "" and base == ""


def test_llm_timeout_is_configurable(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, monkeypatch, '[llm]\nprovider="openai_compatible"\nbase_url="https://x.example/v1"\n'
               'model="m"\ntimeout_seconds=42\n[critic]\nmodel="c"\ntimeout_seconds=7\n', {})
    from fix_issue.cli import build_llms

    llm, critic = build_llms(cfg)
    assert llm._timeout == 42 and critic._timeout == 7
