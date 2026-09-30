"""
smoke_test_nim.py — Isolated smoke test for OpenAICompatibleProvider against NVIDIA NIM.

Reads ~/.fix-issue.toml for api_key, base_url, model.
No triage, no CLI, no GitHub — just a raw LLMProvider.generate() call.

Usage:
    python smoke_test_nim.py

Before running:
    1. Edit ~/.fix-issue.toml: replace nvapi-... with your real NIM key
    2. Edit ~/.fix-issue.toml: replace PLACEHOLDER with a real NIM model name
       (e.g. "meta/llama-3.1-8b-instruct" or "mistralai/mixtral-8x7b-instruct-v0.1")
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Make sure the package is importable when run from the project root
sys.path.insert(0, str(Path(__file__).parent))

from fix_issue import config as cfg_module
from fix_issue.llm_provider import LLMError, build_provider

PROMPT = "Say hello in one sentence."


def main() -> None:
    # ── Load config ──────────────────────────────────────────────────
    toml_path = Path.home() / ".fix-issue.toml"
    cfg = cfg_module.load(toml_path)

    print("=" * 60)
    print("NIM Smoke Test")
    print("=" * 60)
    print(f"  provider : {cfg.llm.provider}")
    print(f"  base_url : {cfg.llm.base_url}")
    print(f"  model    : {cfg.llm.model}")
    print(f"  api_key  : {cfg.llm.api_key[:12]}{'...' if len(cfg.llm.api_key) > 12 else ''}")
    print()

    # ── Pre-flight checks ────────────────────────────────────────────
    if cfg.llm.provider != "openai_compatible":
        print(f"[WARN] provider is '{cfg.llm.provider}', not 'openai_compatible'.")
        print("       NIM uses the OpenAI-compatible schema. Update ~/.fix-issue.toml.")

    if "PLACEHOLDER" in cfg.llm.model or not cfg.llm.model:
        print("[ERROR] model is still set to PLACEHOLDER in ~/.fix-issue.toml.")
        print("        Set it to a real NIM model name and re-run.")
        print()
        print("  Common NIM models:")
        print("    meta/llama-3.1-8b-instruct")
        print("    meta/llama-3.1-70b-instruct")
        print("    mistralai/mixtral-8x7b-instruct-v0.1")
        print("    nvidia/llama-3.1-nemotron-70b-instruct")
        sys.exit(1)

    if not cfg.llm.api_key or "nvapi-..." in cfg.llm.api_key:
        print("[ERROR] api_key is still a placeholder in ~/.fix-issue.toml.")
        print("        Paste your real nvapi-... key and re-run.")
        sys.exit(1)

    # ── Build provider + call ────────────────────────────────────────
    provider = build_provider(
        provider=cfg.llm.provider,
        api_key=cfg.llm.api_key,
        model=cfg.llm.model,
        base_url=cfg.llm.base_url,
    )

    print(f"Sending prompt: \"{PROMPT}\"")
    print(f"To: POST {cfg.llm.base_url}/chat/completions")
    print()

    try:
        response = provider.generate(PROMPT)
    except LLMError as e:
        print("[ERROR] LLM call failed:")
        print(str(e))
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Unexpected error: {type(e).__name__}: {e}")
        sys.exit(1)

    print("=" * 60)
    print("Response:")
    print("=" * 60)
    print(response)
    print()
    print("[OK] Connection works. Provider is ready to use with fix-issue.")


if __name__ == "__main__":
    main()
