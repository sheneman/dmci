############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# llm_providers.py: Multi-provider LLM client for the Experiment F *thinking-mode* re-run. Adds, over the original `llm_client`: -...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Multi-provider LLM client for the Experiment F *thinking-mode* re-run.

Adds, over the original `llm_client`:
  - reasoning/"thinking" mode (the original run used `/nothink`, i.e. reasoning OFF);
  - a configurable model + provider (MindRouter qwen, or the OpenAI API);
  - a large `max_completion_tokens` so a reasoning model has room to think;
  - per-(label) on-disk caching, keyed by (model, thinking, system, user), so the thinking
    run caches separately from the original no-think cache and reruns replay instantly.

Provider-specific switches follow the MindRouter docs:
  - MindRouter/Qwen: `extra_body={"chat_template_kwargs": {"enable_thinking": true}}`
  - OpenAI/GPT-5.x: reasoning is native; we pass `reasoning_effort` (in extra_body so the
    request body carries it regardless of the installed SDK version).
Non-standard params go through `extra_body` so an older `openai` SDK won't reject them — the
server interprets them. The `<think>...</think>` block (Qwen) is stripped before parsing.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import dotenv

dotenv.load_dotenv(Path(__file__).parent.parent.parent / ".env")
CACHE_ROOT = Path(__file__).parent / "llm_cache"


@dataclass
class LLMSpec:
    label: str                      # cache subdir + result tag, e.g. "qwen27b_think"
    provider: str                   # "mindrouter" | "openai"
    model: str                      # e.g. "qwen/qwen3.6-27b" | "gpt-5.5"
    thinking: bool = True
    max_completion_tokens: int = 32768
    temperature: float | None = 0.7
    reasoning_effort: str = "high"  # OpenAI reasoning models


def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


def _client(spec: LLMSpec):
    import openai
    if spec.provider == "mindrouter":
        return openai.OpenAI(
            base_url=os.environ.get("MINDROUTER_BASE_URL", "https://mindrouter.uidaho.edu/v1"),
            api_key=os.environ["MINDROUTER_API_KEY"])
    if spec.provider == "openai":
        return openai.OpenAI(
            base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            api_key=os.environ["OPENAI_API_KEY"])
    raise ValueError(f"unknown provider {spec.provider!r}")


def _cache_path(spec: LLMSpec, system: str, user: str) -> Path:
    h = hashlib.sha256(
        f"{spec.model}\x00{spec.thinking}\x00{system}\x00{user}".encode()).hexdigest()[:16]
    return CACHE_ROOT / spec.label / f"{h}.json"


def complete(spec: LLMSpec, system: str, user: str) -> str:
    """One cached chat completion. Returns the (think-stripped) text."""
    path = _cache_path(spec, system, user)
    if path.exists():
        return json.loads(path.read_text())["response"]

    extra: dict = {"max_completion_tokens": spec.max_completion_tokens}
    kwargs: dict = {
        "model": spec.model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
    }
    if spec.provider == "mindrouter":
        extra["chat_template_kwargs"] = {"enable_thinking": spec.thinking}
        if spec.temperature is not None:
            kwargs["temperature"] = spec.temperature
    else:  # openai reasoning models: native reasoning, default temperature only
        if spec.thinking:
            extra["reasoning_effort"] = spec.reasoning_effort
    kwargs["extra_body"] = extra

    resp = _client(spec).chat.completions.create(**kwargs)
    msg = resp.choices[0].message
    content = msg.content or getattr(msg, "reasoning_content", None) or ""
    content = _strip_think(content)
    if not content:
        raise ValueError(f"empty response from {spec.label} ({spec.model})")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        {"label": spec.label, "model": spec.model, "thinking": spec.thinking,
         "system": system, "user": user, "response": content}, indent=2))
    return content


# Provider specs requested for the thinking-mode re-run.
SPECS = {
    "qwen27b_think": LLMSpec(
        label="qwen27b_think", provider="mindrouter", model="qwen/qwen3.6-27b",
        thinking=True, max_completion_tokens=16384, temperature=0.7),  # 16384: avoid thinking-trace timeouts
    "gpt55_think": LLMSpec(
        label="gpt55_think", provider="openai", model="gpt-5.5",
        thinking=True, max_completion_tokens=32768, temperature=None),
}
