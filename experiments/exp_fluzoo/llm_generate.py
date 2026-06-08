############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# llm_generate.py: The outer loop: LLM-driven search of the discrete influenza-model program space. A language model (MindRouter...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""The outer loop: LLM-driven search of the discrete influenza-model program space.

A language model (MindRouter qwen by default) writes whole Scheme programs encoding
influenza transmission + observation models; each proposal runs the VALIDATE -> REPAIR
loop against the validity funnel (which compiles it, checks finite/nonzero gradients, a
stable rollout, and that it forecasts), and accepted programs are cached. Recipes sampled
across model families (compartmental / seasonal / observation / regional-coupling /
hybrid-closure) steer the LLM toward a semantically diverse zoo; a canonical structural
hash collapses duplicates so we count genuinely distinct structures.

This is the discrete half of the co-search: every accepted program is then handed, as data,
to the SAME frozen DMCI interpreter for gradient calibration -- no per-program reimplementation.

MindRouter is campus-only; run generation on-campus/HPC. `--offline` validates the
hand-written reference programs through the identical funnel + cache machinery so the harness
can be exercised anywhere (no API call).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import DEFAULT, GATE

# Raise BEFORE neural_compiler is imported (here and in every spawned worker).
sys.setrecursionlimit(DEFAULT.recursion_limit)

import torch

from .programs import REFERENCE_PROGRAMS
from .validity import screen, STAGES
from .data.build_data import load_processed
from .forecast import season_matrix

try:
    from dotenv import load_dotenv
    _ROOT = Path(__file__).resolve().parents[2]
    load_dotenv(_ROOT / ".env")
except Exception:  # noqa: BLE001
    pass

HERE = Path(__file__).resolve().parent
CACHE_DIR = HERE / "llm_cache"

# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #

_PROMPT_RULES = r"""You write differentiable influenza forecasting models as Scheme programs for a
compiled interpreter (DMCI). Output EXACTLY two top-level S-expressions and NOTHING ELSE (no prose,
no markdown headers, no comments) inside a single ```scheme code block:
  1. (params (name kind init [scale]) ...)   -- declare EVERY learnable parameter (ALL SCALARS)
  2. (loop ((k 0) <state...> (yhat (zeros 11)) (L 0.0)) (if (= k NWEEKS) L (let* (...) (recur ...))))

Match the WORKED EXAMPLES below EXACTLY in structure and parenthesization. Change only the
compartments, the transition algebra, the seasonal/observation terms, and the parameter list to
realize the requested design. Do not invent new syntax.

CONTRACT (every rule load-bearing):
- `obs` is a [T,11] matrix bound at run time (national + 10 HHS regions). (ref obs k) is the observed
  11-vector for week k. Your model predicts an 11-vector `ypred` of weighted %ILI as a PROPORTION
  (~0.0005..0.13). Compartments (S,E,I,R,...) are 11-VECTORS, one entry per region.
- Parameters are SCALARS ONLY. kind = positive (>0) | unit (0..1; optional 4th `scale` -> 0..scale) |
  signed-unit (-1..1) | free. Per-region variation comes from vector STATE and ops, NEVER vector params.
- The loop MUST carry `yhat` (set to the current `ypred` in recur) and `L` (accumulated NLL). The base
  case (k = NWEEKS) returns the symbol `L`. Use the LITERAL token NWEEKS as the loop bound, never a number.
- Seasonal transmission is built from the integer WEEK COUNTER k ONLY, e.g. (cos (+ (* 0.12083 k) phase)).
- Likelihood is Gaussian with a floored variance (+ s2 1e-6); `s2` is a positive parameter.
- Initial-condition parameters MUST be named starting with i0/e0 (e.g. i0, e0) so they can be re-estimated.

OP SURFACE -- use ONLY these (any other head SILENTLY computes 0 and the program is REJECTED):
  arithmetic + - * /   ** STRICTLY BINARY: (* a (* b c)), NEVER (* a b c) **
  scalar math: sin cos exp sqrt log abs pow min max ; comparison (counter only): = < > <= >=
  tensors: vec mat ref dot scale matvec matmul transpose inv det logdet outer eye zeros ones vsum vlen norm normalize
  special forms: if cond let let* letrec lambda begin define ; and (loop ((v init)..) body)/(recur ..)

COMMON MISTAKES that get programs REJECTED -- avoid ALL of these:
- Closing the (let* (...) ...) BINDING LIST early. Each binding is (name expr); the list of bindings is
  wrapped in ONE pair of parens, then the SINGLE body expression, then close. Count your parens.
- `(scale s v)` scales a vector by SCALAR s. `(* u v)` is the ELEMENTWISE product of two vectors. `(dot u v)`
  is the inner product of two vectors returning a SCALAR -- NEVER apply dot to a matrix. For the squared
  error use exactly (dot resid resid).
- 3-argument arithmetic or 3-argument scale. Everything is binary.
- Vector-valued parameters, named-let, or reading obs at any index other than k.
- Dividing by a quantity that can be 0 (always add a small floor like 1e-6)."""

_RULES_TAIL = r"""Now produce a SEMANTICALLY DISTINCT, runnable model for the requested design, following the
structure of the examples EXACTLY. Output ONLY the two S-expressions in one ```scheme block."""

SYSTEM_PROMPT = (_PROMPT_RULES
                 + "\n\nWORKED EXAMPLE 1 -- a complete, valid regional SEIR (copy this structure):\n```scheme\n"
                 + REFERENCE_PROGRAMS["seir_regional"]
                 + "\n```\n\nWORKED EXAMPLE 2 -- a regional SEIRS adding waning immunity R->S (note how a"
                 + " compartment term and a parameter are added without breaking the structure):\n```scheme\n"
                 + REFERENCE_PROGRAMS["seirs_regional"]
                 + "\n```\n\n" + _RULES_TAIL)

# --------------------------------------------------------------------------- #
# Recipes: sampled across model families to drive zoo diversity
# --------------------------------------------------------------------------- #

_FAMILIES = {
    "compartmental": [
        "an SIR model (no exposed compartment)",
        "an SEIR model",
        "an SEIRS model with waning immunity R->S",
        "an SIRS model with waning immunity",
        "an SEIR model with an asymptomatic compartment splitting off from E",
        "an SEIR model with an additional hospitalized compartment fed from I",
        "an SEIIR model with two infectious sub-compartments (Erlang-distributed infectious period)",
    ],
    "seasonal": [
        "single-harmonic seasonal transmission cos(omega*k+phase)",
        "two-harmonic seasonal transmission (a fundamental plus a second harmonic)",
        "seasonal transmission with both amplitude and a learned baseline offset",
        "a time-varying reproduction number driven by a seasonal cosine",
    ],
    "observation": [
        "a constant reporting fraction rho mapping prevalence to %ILI",
        "a reporting fraction plus a small additive baseline (non-influenza ILI)",
        "a multiplicative regional reporting bias (per-region scale around a shared rho)",
        "incidence-based reporting (report new infections rather than prevalence)",
    ],
    "coupling": [
        "11 independent regional epidemics sharing the transmission parameters",
        "regions sharing a single seasonal forcing but with per-region transmission scale",
        "low-rank regional coupling that mixes a small amount of neighbouring prevalence",
        "a learned rank-1 contact correction added to the transmission term",
    ],
    "closure": [
        "a purely mechanistic structure (no learned closure)",
        "an SEIR core plus a small learned low-rank correction to the force of infection",
        "an SEIR core with a learned multiplicative seasonal closure on transmission",
    ],
}

_AXES = ("compartmental", "seasonal", "observation", "coupling", "closure")


@dataclass(frozen=True)
class Recipe:
    idx: int
    choices: dict

    @property
    def id(self) -> str:
        return f"zoo_{self.idx:04d}"

    def describe(self) -> str:
        return "; ".join(self.choices[a] for a in _AXES)


def sample_recipes(n: int, seed: int = 0) -> list[Recipe]:
    """Deterministic diverse recipes spanning the family cross-product."""
    g = torch.Generator().manual_seed(int(seed))
    recipes, seen = [], set()
    tries = 0
    while len(recipes) < n and tries < n * 50:
        tries += 1
        choices = {}
        for a in _AXES:
            opts = _FAMILIES[a]
            j = int(torch.randint(len(opts), (1,), generator=g))
            choices[a] = opts[j]
        sig = tuple(choices[a] for a in _AXES)
        if sig in seen:
            continue
        seen.add(sig)
        recipes.append(Recipe(idx=len(recipes), choices=dict(choices)))
    return recipes


def _user_prompt(recipe: Recipe) -> str:
    return ("Write a weekly influenza model with this design:\n"
            f"  - structure: {recipe.choices['compartmental']}\n"
            f"  - seasonality: {recipe.choices['seasonal']}\n"
            f"  - observation: {recipe.choices['observation']}\n"
            f"  - regional structure: {recipe.choices['coupling']}\n"
            f"  - closure: {recipe.choices['closure']}\n"
            "Return ONLY the two S-expressions in one ```scheme block, following every rule.")


def _repair_suffix(stage: str, detail: str, hint: str) -> str:
    return ("\n\nYOUR PREVIOUS ATTEMPT FAILED the validity check at stage "
            f"'{stage}': {detail}\nREPAIR: {hint}\nRe-emit the COMPLETE corrected program "
            "as two S-expressions in one ```scheme block.")


# --------------------------------------------------------------------------- #
# LLM call + extraction (MindRouter, reused from exp_b/exp_lim_enso)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class LLMSpec:
    """A generation backend: provider + model + decoding knobs."""
    key: str                       # short id used in cache keys + records
    provider: str                  # "mindrouter" | "openai"
    model: str
    thinking: bool = False         # MindRouter qwen: keep OFF so programs aren't truncated
    reasoning_effort: str | None = None   # OpenAI: "minimal" keeps it fast for code
    max_tokens: int = 32768
    temperature: float | None = DEFAULT.gen_temperature


#: The model zoo's operators. qwen via campus MindRouter; GPT-5.5 via OpenAI.
SPECS: dict[str, LLMSpec] = {
    "qwen35": LLMSpec("qwen35", "mindrouter", "qwen/qwen3.6-35b", thinking=False),
    "qwen27": LLMSpec("qwen27", "mindrouter", "qwen/qwen3.6-27b", thinking=False),
    "gpt55":  LLMSpec("gpt55", "openai", "gpt-5.5", reasoning_effort="minimal", temperature=None),
}
DEFAULT_MODELS = ("qwen35",)


def _call_llm(user: str, spec: LLMSpec, system: str = SYSTEM_PROMPT) -> str:
    import openai
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    # Bound every request so one stalled call can never hang a whole generation barrier.
    _TIMEOUT, _RETRIES = 150.0, 2
    if spec.provider == "mindrouter":
        client = openai.OpenAI(
            base_url=os.environ.get("MINDROUTER_BASE_URL", "https://mindrouter.uidaho.edu/v1"),
            api_key=os.environ["MINDROUTER_API_KEY"], timeout=_TIMEOUT, max_retries=_RETRIES)
        kwargs = dict(model=spec.model, max_tokens=spec.max_tokens, messages=msgs)
        if spec.temperature is not None:
            kwargs["temperature"] = spec.temperature
        # disable qwen thinking so the FULL program is emitted (not truncated behind reasoning)
        try:
            resp = client.chat.completions.create(
                **kwargs, extra_body={"chat_template_kwargs": {"enable_thinking": spec.thinking}})
        except Exception:  # noqa: BLE001
            resp = client.chat.completions.create(**kwargs)
    else:  # openai (GPT-5.5)
        client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"],
                               base_url=os.environ.get("OPENAI_BASE_URL") or None,
                               timeout=_TIMEOUT, max_retries=_RETRIES)
        kwargs = dict(model=spec.model, max_completion_tokens=spec.max_tokens, messages=msgs)
        if spec.temperature is not None:
            kwargs["temperature"] = spec.temperature
        extra = {"reasoning_effort": spec.reasoning_effort} if spec.reasoning_effort else None
        try:
            resp = (client.chat.completions.create(**kwargs, extra_body=extra) if extra
                    else client.chat.completions.create(**kwargs))
        except Exception:  # noqa: BLE001
            resp = client.chat.completions.create(**kwargs)
    msg = resp.choices[0].message
    return msg.content or getattr(msg, "reasoning_content", "") or ""


def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)


def extract_scheme(response: str) -> str:
    """Pull the two-form program out of an LLM response, cleaned of prose."""
    text = _strip_think(response)
    m = re.search(r"```(?:scheme|lisp)?\s*\n(.*?)```", text, flags=re.DOTALL)
    block = m.group(1) if m else text
    i = block.find("(params")
    if i == -1:
        return block.strip()
    from neural_compiler.dmci import split_top_level_forms
    forms = split_top_level_forms(block[i:])
    return "\n\n".join(forms[:2]) if len(forms) >= 2 else block[i:].strip()


# --------------------------------------------------------------------------- #
# Generation + caching
# --------------------------------------------------------------------------- #

@dataclass
class ZooRecord:
    recipe_id: str
    recipe: dict
    status: str                 # accepted | funnel-failed | api-error | duplicate
    stage: str                  # furthest funnel stage reached
    source: str
    canonical: str = ""
    n_params: int = 0
    n_attempts: int = 0
    detail: str = ""
    grad_norm: float = 0.0
    model: str = DEFAULT.llm_model


def _cache_key(model: str, user: str) -> str:
    return hashlib.sha256(f"{model}\x00{SYSTEM_PROMPT}\x00{user}".encode()).hexdigest()[:20]


def _load_cache(key: str) -> ZooRecord | None:
    p = CACHE_DIR / f"{key}.json"
    if p.exists():
        return ZooRecord(**json.loads(p.read_text()))
    return None


def _save_cache(key: str, rec: ZooRecord) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / f"{key}.json").write_text(json.dumps(asdict(rec), indent=2))


def generate_program(recipe: Recipe, probe_obs: torch.Tensor, cfg=DEFAULT, gate=GATE,
                     *, spec_key: str = "qwen35", force: bool = False) -> ZooRecord:
    """Propose -> validate -> repair for one recipe with one model; cache and return."""
    spec = SPECS[spec_key]
    user = _user_prompt(recipe)
    key = _cache_key(spec_key, user)
    if not force:
        hit = _load_cache(key)
        if hit is not None:
            return hit

    last = None
    rec = None
    for attempt in range(1, cfg.max_repairs + 2):
        prompt = user if last is None else user + _repair_suffix(*last)
        try:
            raw = _call_llm(prompt, spec)
        except Exception as exc:  # noqa: BLE001
            rec = ZooRecord(recipe.id, recipe.choices, "api-error", "proposed", "",
                            detail=f"{type(exc).__name__}: {exc}", n_attempts=attempt, model=spec_key)
            _save_cache(key, rec)
            return rec
        src = extract_scheme(raw)
        res = screen(src, cfg=cfg, gate=gate, probe_obs=probe_obs, name=recipe.id)
        rec = ZooRecord(recipe.id, recipe.choices,
                        "accepted" if res.ok else "funnel-failed", res.stage, src,
                        canonical=res.canonical, n_params=res.n_params, n_attempts=attempt,
                        detail=res.detail, grad_norm=res.grad_norm, model=spec_key)
        if res.ok:
            break
        last = (res.stage, res.detail, res.repair_hint)
    _save_cache(key, rec)
    return rec


def validate_source(name: str, source: str, probe_obs: torch.Tensor,
                    cfg=DEFAULT, gate=GATE) -> ZooRecord:
    """Offline path: run a hand-provided program through the identical funnel + cache."""
    res = screen(source, cfg=cfg, gate=gate, probe_obs=probe_obs, name=name)
    rec = ZooRecord(name, {"source": "reference"},
                    "accepted" if res.ok else "funnel-failed", res.stage, source,
                    canonical=res.canonical, n_params=res.n_params, n_attempts=1,
                    detail=res.detail, grad_norm=res.grad_norm, model="offline")
    _save_cache(_cache_key("offline", name), rec)
    return rec


def _probe_obs(cfg=DEFAULT) -> torch.Tensor:
    """A small real-data probe ([12,11]) for the funnel."""
    try:
        data = load_processed()
        return torch.tensor(season_matrix(data, cfg.test_seasons[0])[:12], dtype=torch.float32)
    except Exception:  # noqa: BLE001  (data not built yet)
        return 0.01 * torch.rand(12, cfg.n_regions, dtype=torch.float32)


def funnel_summary(records: list[ZooRecord]) -> dict:
    """Counts at each funnel stage + unique-structure count (the central figure)."""
    counts = {s: 0 for s in STAGES}
    counts["proposed"] = len(records)
    accepted = [r for r in records if r.status == "accepted"]
    for r in records:
        # credit every stage strictly before the one it failed at, plus its own if accepted
        reached = STAGES.index(r.stage) if r.stage in STAGES else 0
        for i, s in enumerate(STAGES[1:], start=1):
            if i <= reached and (r.status == "accepted" or i < reached):
                counts[s] += 1
    counts["accepted"] = len(accepted)
    counts["unique_structures"] = len({r.canonical for r in accepted if r.canonical})
    return counts


# Module-level probe cache + picklable worker for process-parallel generation.
_PROBE = None


def _get_probe_cached():
    global _PROBE
    if _PROBE is None:
        _PROBE = _probe_obs()
    return _PROBE


def _gen_cell(payload: dict) -> dict:
    import sys as _sys
    _sys.setrecursionlimit(payload["recursion_limit"])
    recipe = Recipe(idx=payload["idx"], choices=payload["choices"])
    try:
        rec = generate_program(recipe, _get_probe_cached(),
                               spec_key=payload["spec_key"], force=payload["force"])
    except Exception as exc:  # noqa: BLE001  (one bad program must never crash the pool)
        rec = ZooRecord(recipe.id, recipe.choices, "gen-error", "proposed", "",
                        detail=f"{type(exc).__name__}: {exc}", n_attempts=0,
                        model=payload["spec_key"])
    return asdict(rec)


def main():
    ap = argparse.ArgumentParser(description="Generate / validate the FluZoo program zoo")
    ap.add_argument("--n", type=int, default=DEFAULT.n_programs)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS),
                    help="comma-separated generation backends (round-robin over recipes): "
                         + ", ".join(SPECS))
    ap.add_argument("--workers", type=int, default=1,
                    help="process-parallel generation (LLM call + funnel screen per program)")
    ap.add_argument("--offline", action="store_true",
                    help="validate the hand-written reference programs (no API call)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    records: list[ZooRecord] = []

    if args.offline:
        probe = _probe_obs()
        for name, src in REFERENCE_PROGRAMS.items():
            R = 1 if name.endswith("_national") else DEFAULT.n_regions
            rec = validate_source(name, src, probe[:, :R])
            records.append(rec)
            print(f"  [{rec.status:13s}] {name:16s} stage={rec.stage:14s} "
                  f"nparams={rec.n_params} canon={rec.canonical[:8]}")
    else:
        models = [m.strip() for m in args.models.split(",") if m.strip()]
        for m in models:
            if m not in SPECS:
                raise SystemExit(f"unknown model {m!r}; choices: {list(SPECS)}")
        recipes = sample_recipes(args.n, seed=args.seed)
        payloads = [dict(idx=r.idx, choices=r.choices,
                         spec_key=models[j % len(models)], force=args.force,
                         recursion_limit=DEFAULT.recursion_limit) for j, r in enumerate(recipes)]
        if args.workers <= 1:
            for pl in payloads:
                records.append(ZooRecord(**_gen_cell(pl)))
                _log_gen(records[-1], len(records), len(payloads))
        else:
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
                for i, d in enumerate(ex.map(_gen_cell, payloads), start=1):
                    records.append(ZooRecord(**d))
                    _log_gen(records[-1], i, len(payloads))

    # zoo-level structural de-duplication (after collection)
    seen: set[str] = set()
    for rec in records:
        if rec.status == "accepted" and rec.canonical in seen:
            rec.status = "duplicate"
        if rec.canonical:
            seen.add(rec.canonical)

    summary = funnel_summary(records)
    (CACHE_DIR / "_funnel_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n[funnel]", json.dumps(summary, indent=2))


def _log_gen(rec: "ZooRecord", i: int, n: int) -> None:
    rid = rec.recipe_id
    print(f"  [{i:4d}/{n}] [{rec.status:13s}] {rid} stage={rec.stage:14s} "
          f"attempts={rec.n_attempts} canon={rec.canonical[:8]}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
