############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# llm_client.py: LLM client for Experiment F: iterative model discovery.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""LLM client for Experiment F: iterative model discovery."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import dotenv

dotenv.load_dotenv(Path(__file__).parent.parent.parent / ".env")


CACHE_DIR = Path(__file__).parent / "llm_cache"

SYSTEM_PROMPT = """\
You output ONLY a single Scheme S-expression. No explanation, no prose, \
no markdown, no LaTeX.

CRITICAL: You MUST use named parameters a, b, c, d (not literal numbers) \
for all coefficients and rates. The parameters will be optimized by gradient \
descent. Only use literal numbers for structural constants like 0, 1.0, 2.

Rules:
- Parameters: a, b, c, d (use as few as needed). Input variable: x
- ALL arithmetic operators take exactly TWO arguments: (* a (* b c)) not (* a b c)
- Primitives: +, -, *, /, exp, log, sin, cos, pow, sqrt, abs
- Negate with (- 0 val) not (- val)
- No defines, no lambdas

Examples of CORRECT output (note: named params, not numbers):
(* a (exp (* (- 0 b) x)))
(+ (* a (sin (* b x))) (* c (exp (* (- 0 d) x))))
(/ a (+ 1.0 (exp (* (- 0 b) (- x c)))))
(* a (* (sin (* b x)) (exp (* (- 0 c) x))))
"""


def _cache_key(target_name: str, seed: int, iteration: int,
               prompt: str) -> str:
    h = hashlib.sha256(prompt.encode()).hexdigest()[:12]
    return f"{target_name}_s{seed}_i{iteration}_{h}"


def _load_cache(key: str) -> str | None:
    path = CACHE_DIR / f"{key}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)["response"]


def _save_cache(key: str, prompt: str, response: str):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_DIR / f"{key}.json", "w") as f:
        json.dump({"prompt": prompt, "response": response}, f, indent=2)


def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


def call_llm(prompt: str, target_name: str, seed: int, iteration: int,
             temperature: float = 0.7, max_tokens: int = 16384) -> str:
    key = _cache_key(target_name, seed, iteration, prompt)
    cached = _load_cache(key)
    if cached is not None:
        return cached

    base_url = os.environ.get(
        "MINDROUTER_BASE_URL", "https://mindrouter.uidaho.edu/v1")
    api_key = os.environ.get("MINDROUTER_API_KEY")
    model = os.environ.get("MINDROUTER_MODEL", "qwen/qwen3.6-35b")
    if not api_key:
        raise ValueError("MINDROUTER_API_KEY not set")

    import openai
    if hasattr(openai, "OpenAI"):
        client = openai.OpenAI(base_url=base_url, api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        content = response.choices[0].message.content
        if not content:
            content = getattr(response.choices[0].message,
                              "reasoning_content", None)
    else:
        openai.api_base = base_url
        openai.api_key = api_key
        response = openai.ChatCompletion.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        content = response["choices"][0]["message"]["content"]
    if not content:
        raise ValueError("Empty LLM response")
    content = _strip_think(content)

    _save_cache(key, prompt, content)
    return content


def extract_scheme(response: str) -> str:
    code_match = re.search(
        r"```(?:scheme|lisp)?\s*\n(.*?)```", response, re.DOTALL)
    if code_match:
        return code_match.group(1).strip()

    param_pat = re.compile(r'\b[abcd]\b')
    best = ""

    for m in re.finditer(r'\(', response):
        start = m.start()
        depth, end = 0, 0
        for i in range(start, len(response)):
            ch = response[i]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            if depth == 0:
                end = i + 1
                break
        if end <= start:
            continue
        candidate = response[start:end]
        if not param_pat.search(candidate):
            continue
        has_op = any(op in candidate for op in
                     ["+", "-", "*", "/", "exp", "sin", "cos", "pow",
                      "sqrt", "log", "abs"])
        if has_op and len(candidate) > len(best):
            best = candidate
    if best:
        return best

    for line in response.strip().split("\n"):
        line = line.strip()
        if line.startswith("("):
            return line
    return response.strip()


def format_data_for_llm(xs, ys, n_samples: int = 12) -> str:
    import torch
    n = len(xs)
    indices = torch.linspace(0, n - 1, n_samples).long()
    lines = [f"  x={xs[i].item():.3f}, y={ys[i].item():.4f}" for i in indices]
    return "\n".join(lines)


def make_initial_prompt(xs, ys) -> str:
    data_str = format_data_for_llm(xs, ys)
    n = len(xs)
    return (
        f"Data ({n} points, x in [0,5]):\n"
        f"{data_str}\n\n"
        f"Output a single Scheme expression. /nothink"
    )


def make_refinement_prompt(prev_expr: str, prev_mse: float,
                           fitted_params: dict[str, float],
                           residual_summary: str) -> str:
    param_str = ", ".join(f"{k}={v:.4f}" for k, v in fitted_params.items())
    return (
        f"Previous: {prev_expr}\n"
        f"Params: {param_str}\n"
        f"MSE: {prev_mse:.6f}\n"
        f"Residuals:\n{residual_summary}\n\n"
        f"Propose an improved Scheme expression. /nothink"
    )


def make_retry_prompt(expr: str, error: str) -> str:
    return (
        f"Failed: {expr}\nError: {error}\n\n"
        f"Fix it. Binary operators only, (- 0 val) for negation, "
        f"params a/b/c/d, input x. /nothink"
    )
