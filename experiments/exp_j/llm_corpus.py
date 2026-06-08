############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# llm_corpus.py: LLM-generated program corpus for Exp J — the *external-validity* companion to the synthetic `corpus.py`....
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""LLM-generated program corpus for Exp J — the *external-validity* companion to the
synthetic `corpus.py`.

`corpus.py` generates structurally-distinct programs from random expression trees. A fair
reviewer asks: is that synthetic distribution representative of what an LLM actually emits?
This module answers that empirically. It asks MindRouter (qwen) to author scientific-model
programs in the supported Scheme subset, caches the responses (so the measurement is
reproducible and not gated on live LLM latency), and turns each *real* program string into an
Exp J `Program` that runs through the identical three arms (DMCI / B1-lambdify / B2-handport).

Two families are requested:
  - closed-form expressions (y = f(x; a,b,c,d)) — lambdify CAN ingest these, so they test
    whether real LLM programs reproduce the synthetic corpus's compile/coverage/engineering/
    recovery curves. Ground truth is an independent Python evaluator (`sexp_eval`).
  - iterative/recursive programs (a `define` with tail recursion) — lambdify CANNOT ingest
    these (coverage collapse), demonstrating the recursive-fraction point on genuine LLM
    output. Ground truth comes from DMCI itself (the only evaluator that runs them here); B1
    fails coverage; B2's port is costed (LOC) but not auto-emitted (recovery via DMCI).

Each returned program is classified by what actually parses (not by what we asked for), so LLM
disobedience degrades gracefully. Programs are de-duplicated by canonical string and
rejection-tested for finiteness, exactly like the synthetic corpus.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import dotenv
import torch

from neural_compiler import evaluate_program
from neural_compiler.runtime.tagged_value import make_float, unwrap_number
from experiments.exp_f.exp_f import detect_used_params
from experiments.exp_f.llm_client import _strip_think, extract_scheme
from . import sexp_eval
from .corpus import Program, _XR, _PR

dotenv.load_dotenv(Path(__file__).parent.parent.parent / ".env")
CACHE_DIR = Path(__file__).parent / "llm_cache"

_PARAM_POOL = ["a", "b", "c", "d"]

_CLOSED_SYSTEM = """\
You output ONLY a single Scheme S-expression modelling y = f(x). No prose, no markdown.
Rules:
- Input variable: x. Learnable parameters: a, b, c, d (use as few as needed, at least one).
- Use named parameters for ALL coefficients/rates (they are optimized by gradient descent);
  literal numbers only for structural constants like 0, 1.0, 2.
- ALL operators take exactly TWO arguments: (* a (* b c)), never (* a b c).
- Primitives: +, -, *, /, exp, log, sin, cos, pow, sqrt, abs. Negate with (- 0 v).
- No defines, no lambdas, no if.
Examples:
(* a (exp (* (- 0 b) x)))
(+ (* a (sin (* b x))) (* c (exp (* (- 0 d) x))))
(/ a (+ 1.0 (exp (* (- 0 b) (- x c)))))
"""

_REC_SYSTEM = """\
You output ONLY Scheme code: a single recursive `define` followed by one call. No prose, no markdown.
This models an ITERATIVE scientific computation (a fixed number of update steps).
Rules:
- Input variable: x. Learnable parameters: a, b, c, d (at least one; optimized by gradient descent).
- Use a tail-recursive helper with an explicit step counter that counts down to 0.
- Comparisons allowed: = < > <= >=. Arithmetic: +, -, *, /, exp, log, sin, cos, pow, sqrt, abs.
- ALL operators take exactly TWO arguments. Negate with (- 0 v).
- The recursion MUST terminate in a fixed number of steps (10-20). Keep updates damped/stable.
Example (a damped relaxation, 15 steps):
(define (f y n) (if (= n 0) y (f (+ y (* 0.1 (- (* a (sin x)) (* b y)))) (- n 1)))) (f 1.0 15)
"""

_PHENOMENA = [
    "exponential decay toward a baseline", "logistic / sigmoid saturation",
    "a damped oscillation", "a resonance peak", "power-law growth",
    "a Gaussian bump", "Michaelis-Menten saturation kinetics",
    "a sum of two decaying modes", "a sinusoid with growing amplitude",
    "hyperbolic saturation", "a double-exponential rise-and-decay",
    "a skewed unimodal pulse", "log-linear growth", "an inverse-square falloff",
    "a thresholded soft ramp", "a beat pattern (two close frequencies)",
]


# --- cached MindRouter call -------------------------------------------------
# Reasoning ("thinking") ENABLED with a large completion budget, matching Experiment F's
# config (qwen reasoning via chat_template_kwargs; max_completion_tokens=16384). The cache key
# includes the model + thinking flag so thinking responses never collide with the old
# non-thinking cache.
# Pinned to 27B (matches Experiment F). We deliberately ignore the MINDROUTER_MODEL env var
# (set to 35B on HPC) because qwen3.6-35b intermittently 404s under thinking mode; 27B+thinking
# is validated working.
_MODEL = "qwen/qwen3.6-27b"
_THINKING = True
_MAX_COMPLETION_TOKENS = 16384   # 16384 (not 32768): cap thinking-trace length to avoid timeouts


def _cached_call(system: str, user: str, key: str) -> str:
    path = CACHE_DIR / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())["response"]

    base_url = os.environ.get("MINDROUTER_BASE_URL", "https://mindrouter.uidaho.edu/v1")
    api_key = os.environ.get("MINDROUTER_API_KEY")
    if not api_key:
        raise ValueError("MINDROUTER_API_KEY not set (need it to populate the LLM cache)")
    import openai
    client = openai.OpenAI(base_url=base_url, api_key=api_key)
    resp = client.chat.completions.create(
        model=_MODEL, temperature=0.7,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        extra_body={"max_completion_tokens": _MAX_COMPLETION_TOKENS,
                    "chat_template_kwargs": {"enable_thinking": _THINKING}})
    content = resp.choices[0].message.content or getattr(
        resp.choices[0].message, "reasoning_content", "") or ""
    content = _strip_think(content)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        {"model": _MODEL, "thinking": _THINKING, "system": system,
         "user": user, "response": content}, indent=2))
    return content


def _ask(system: str, user: str) -> str:
    key = hashlib.sha256(
        f"{_MODEL}\x00{_THINKING}\x00{system}\x00{user}".encode()).hexdigest()[:16]
    return extract_scheme(_cached_call(system, user, key))


# --- Program construction ---------------------------------------------------

def _finite(fn, n=8) -> bool:
    for i in range(n):
        x = _XR[0] + (_XR[1] - _XR[0]) * (i / (n - 1))
        try:
            v = float(fn(x))
        except Exception:
            return False
        if v != v or abs(v) > 1e4:
            return False
    return True


def _build_closed(pid, scheme, tree, g) -> Program | None:
    params = sorted(s for s in sexp_eval.free_symbols(tree) if s != "x")
    params = [p for p in params if p != "x"]
    if not params or any(p not in _PARAM_POOL for p in params):
        return None
    targets = {p: float(_PR[0] + (_PR[1] - _PR[0]) * torch.rand((), generator=g)) for p in params}

    def gt(x, _t=tree, **P):
        return sexp_eval.eval_python(_t, {"x": x, **P})
    if not _finite(lambda x: gt(x, **targets)):
        return None
    return Program(
        pid=pid, kind="closed_form", scheme=scheme, input_names=["x"],
        param_names=params, targets=targets, bounds={p: _PR for p in params},
        input_ranges={"x": _XR}, ground_truth=gt,
        jax_forward=sexp_eval.make_jax_forward(tree), port_loc=sexp_eval.node_count(tree))


def _build_recursive(pid, scheme, interp, g) -> Program | None:
    params = detect_used_params(scheme, _PARAM_POOL)
    if not params:
        return None
    targets = {p: float(_PR[0] + (_PR[1] - _PR[0]) * torch.rand((), generator=g)) for p in params}

    def gt(x, _s=scheme, _i=interp, **P):
        binds = {"x": float(x), **{k: float(v) for k, v in P.items()}}
        return float(unwrap_number(evaluate_program(_i, _s, binds)))
    if not _finite(lambda x: gt(x, **targets)):
        return None
    return Program(
        pid=pid, kind="recursive", scheme=scheme, input_names=["x"],
        param_names=params, targets=targets, bounds={p: _PR for p in params},
        input_ranges={"x": _XR}, ground_truth=gt,
        jax_forward=None,                       # no auto-port; B2 costed by LOC, recovery skipped
        port_loc=sexp_eval.token_count(scheme) + 4)


def _classify_and_build(pid, scheme, interp, g) -> Program | None:
    try:
        tree = sexp_eval.parse(scheme)
        if sexp_eval.is_closed_form(tree):
            return _build_closed(pid, scheme, tree, g)
    except ValueError:
        pass  # multi-form / non-closed -> recursive path
    return _build_recursive(pid, scheme, interp, g)


def _candidate_prompt(family: str, k: int) -> str:
    """Deterministic prompt for candidate index `k` — independent of acceptance state, so
    candidates can be fetched concurrently and the cache key is stable across runs."""
    phen = _PHENOMENA[k % len(_PHENOMENA)]
    variant = k // len(_PHENOMENA)
    nudge = f" Variant {variant}: use a distinct functional form." if variant else ""
    return f"Model {family} #{k}: {phen}.{nudge} Output only the Scheme code. /nothink"


def _gen_family(family, system, count, g, interp, seen, pid_start, max_workers=8):
    """Fetch a deterministic candidate stream CONCURRENTLY (MindRouter is I/O-bound), and
    accept programs in candidate-index order until `count` distinct/finite ones are built.

    Acceptance is processed strictly in index order (we read futures in submission order),
    so the accepted set — and the RNG draws for target values — stay deterministic regardless
    of which HTTP call returns first. Caching is per-candidate, so reruns are instant."""
    from concurrent.futures import ThreadPoolExecutor

    progs: list[Program] = []
    max_candidates = count * 6        # headroom for dedup/parse/finite rejections
    pid = pid_start
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        k = 0
        while len(progs) < count and k < max_candidates:
            batch = list(range(k, min(k + max_workers, max_candidates)))
            futures = [ex.submit(_ask, system, _candidate_prompt(family, j)) for j in batch]
            for fut in futures:                       # in submission (index) order
                if len(progs) >= count:
                    break
                try:
                    scheme = fut.result()
                except Exception as e:
                    print(f"  [{family}] LLM call failed: {e}", flush=True)
                    continue
                canon = re.sub(r"\s+", " ", scheme).strip()
                if not canon or canon in seen:
                    continue
                prog = _classify_and_build(pid, scheme, interp, g)
                if prog is None:
                    continue
                seen.add(canon)
                progs.append(prog)
                pid += 1
                if len(progs) % 10 == 0 or len(progs) == count:
                    print(f"  [{family}] accepted {len(progs)}/{count} "
                          f"(through candidate {k + len(batch)})", flush=True)
            k += len(batch)
    return progs


def generate_llm_corpus(n_closed: int, n_recursive: int, interp, seed: int = 0,
                        max_workers: int = 8) -> list[Program]:
    """Generate up to `n_closed` + `n_recursive` distinct, finite LLM programs.

    Candidates are fetched with `max_workers` concurrent MindRouter calls (the LLM round-trip,
    not compute, is the bottleneck). Deterministic + cached: a rerun replays from cache."""
    g = torch.Generator().manual_seed(seed)
    seen: set[str] = set()
    closed = _gen_family("closed", _CLOSED_SYSTEM, n_closed, g, interp, seen, 0, max_workers)
    recursive = _gen_family("recursive", _REC_SYSTEM, n_recursive, g, interp, seen,
                            len(closed), max_workers)
    return closed + recursive
