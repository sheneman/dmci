############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# models.py: FATES-anchored ecological response models for Experiment I. Every model provides (mirroring...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""FATES-anchored ecological response models for Experiment I.

Every model provides (mirroring experiments/exp_c/models.py):
  - interp_source: bootstrap evaluator + program as quoted data  (DMCI path)
  - direct_source: the same expression compiled directly         (fast twin)
  - ground_truth:  Python function matching the Scheme exactly
plus param/driver schema, targets, init priors, and DE bounds.

All forms are `if`-free and strictly binary-nested:
  - the bootstrap interpreter reads ONLY the first two args of +,-,*,/ (a 3-term
    sum silently drops terms), so every sum/product of >2 terms is nested;
  - tagged_if is a HARD selector (sel=(truth!=0).float()), so a comparison's
    threshold LOCATION gets ZERO gradient -> every regime switch is a smooth
    sigmoid, never an (if (< ...) ...).

Named FATES/ecology subsystems used:
  light(Q)  : leaf_biophys light response, rectangular hyperbola (Falge 2001)
  temp(T)   : photosynthetic temperature optimum, Gaussian
  water(psi): btran water-stress downregulation, sigmoid relaxation of the hard ramp
  pool      : (optional) leaf/wood carbon pool, recursive Euler integration (Exp C idiom)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .config import DRIVERS, DRIVER_RANGES


BOOTSTRAP_DIR = Path(__file__).parent.parent.parent / "bootstrap"
EVALUATOR_SOURCE = (BOOTSTRAP_DIR / "compiler.scm").read_text()

PFT_PARAMS = ["alpha", "Amax", "Topt", "w", "s", "psi50"]

# Domain-plausible, target-free bounds for the black-box baseline (pre-registered,
# ~1 order of magnitude span; do NOT derive these from target_values).
PARAM_BOUNDS = {
    "alpha": (0.001, 0.5),
    "Amax": (1.0, 40.0),
    "Topt": (5.0, 40.0),
    "w": (1.0, 30.0),
    "s": (0.1, 20.0),
    "psi50": (-4.0, 0.0),
}

# Distinct per-PFT ground-truth targets (synthetic recovery).
PFT_TARGETS = [
    {"alpha": 0.06, "Amax": 18.0, "Topt": 24.0, "w": 8.0, "s": 2.5, "psi50": -1.2},
    {"alpha": 0.04, "Amax": 12.0, "Topt": 30.0, "w": 6.0, "s": 3.5, "psi50": -0.8},
    {"alpha": 0.08, "Amax": 22.0, "Topt": 21.0, "w": 10.0, "s": 1.8, "psi50": -1.6},
    {"alpha": 0.03, "Amax": 9.0, "Topt": 33.0, "w": 5.0, "s": 4.5, "psi50": -0.5},
    {"alpha": 0.05, "Amax": 15.0, "Topt": 27.0, "w": 7.0, "s": 3.0, "psi50": -1.0},
]


def _pft_target(k: int) -> dict[str, float]:
    """Deterministic, domain-plausible ground-truth target for PFT index ``k``.

    The first five reuse the hand-set pilot targets (so d<=30 stays bit-comparable to
    earlier runs); for k>=5 (the d-scaling regime up to 21 PFTs / d=126) targets are
    drawn deterministically from the interior (10--90%) of each parameter's
    pre-registered PARAM_BOUNDS, seeded by k for reproducibility. Staying off the bound
    edges keeps every synthetic-recovery problem well posed for all fitters."""
    if k < len(PFT_TARGETS):
        return PFT_TARGETS[k]
    g = random.Random(1000 + k)
    out = {}
    for s in PFT_PARAMS:
        lo, hi = PARAM_BOUNDS[s]
        out[s] = round(lo + (0.1 + 0.8 * g.random()) * (hi - lo), 4)
    return out


@dataclass
class ModelSpec:
    name: str
    interp_source: str
    direct_source: str
    expression: str                  # the bare GPP expression (fed to lambdify->JAX)
    param_names: list[str]
    target_values: dict[str, float]
    input_names: list[str]
    ground_truth: Callable[..., float]
    driver_ranges: dict[str, tuple[float, float]]
    param_bounds: dict[str, tuple[float, float]]
    is_recursive: bool = False
    init_prior: dict[str, float] = field(default_factory=dict)


def _make_env(names: list[str]) -> str:
    """Scheme association-list env: (list (cons 'name name) ...)."""
    pairs = " ".join(f"(cons '{n} {n})" for n in names)
    return f"(list {pairs})"


# --- Static composite GPP model ---------------------------------------------

def _pft_expr(prefix: str) -> str:
    """Per-PFT GPP = light(Q) * temp(T) * water(psi), all binary-nested, if-free."""
    p = prefix
    light = (f"(/ (* {p}alpha (* Q {p}Amax)) "
             f"(+ (* {p}alpha Q) {p}Amax))")
    temp = (f"(exp (/ (- 0 (* (- T {p}Topt) (- T {p}Topt))) "
            f"(* 2.0 (* {p}w {p}w))))")
    water = f"(/ 1.0 (+ 1.0 (exp (* (- 0 {p}s) (- psi {p}psi50)))))"
    return f"(* {light} (* {temp} {water}))"


def _community_expr(n_pft: int, covers: list[float]) -> str:
    """Community GPP = sum_k cover_k * GPP_k, right-nested binary +."""
    terms = [f"(* {covers[k]!r} {_pft_expr(f'p{k+1}_')})" for k in range(n_pft)]
    expr = terms[-1]
    for t in reversed(terms[:-1]):
        expr = f"(+ {t} {expr})"
    return expr


def _pft_gpp(Q, T, psi, alpha, Amax, Topt, w, s, psi50) -> float:
    light = (alpha * Q * Amax) / (alpha * Q + Amax)
    temp = math.exp(-((T - Topt) ** 2) / (2.0 * w * w))
    water = 1.0 / (1.0 + math.exp(-s * (psi - psi50)))
    return light * temp * water


def build_static_model(n_pft: int = 2) -> ModelSpec:
    covers = [round(c, 4) for c in _default_covers(n_pft)]
    param_names = [f"p{k+1}_{s}" for k in range(n_pft) for s in PFT_PARAMS]
    pft_tgts = [_pft_target(k) for k in range(n_pft)]
    targets = {f"p{k+1}_{s}": pft_tgts[k][s]
               for k in range(n_pft) for s in PFT_PARAMS}
    bounds = {f"p{k+1}_{s}": PARAM_BOUNDS[s]
              for k in range(n_pft) for s in PFT_PARAMS}

    expression = _community_expr(n_pft, covers)
    input_names = list(DRIVERS)
    env = _make_env(input_names + param_names)
    interp_source = (EVALUATOR_SOURCE +
                     f"\n(scheme-eval '{expression} {env})\n")
    # Pure-arithmetic direct twin: no defines, no prelude -> untagged fast graph.
    direct_source = expression

    def gt(**kw) -> float:
        total = 0.0
        for k in range(n_pft):
            p = f"p{k+1}_"
            total += covers[k] * _pft_gpp(
                kw["Q"], kw["T"], kw["psi"],
                kw[p + "alpha"], kw[p + "Amax"], kw[p + "Topt"],
                kw[p + "w"], kw[p + "s"], kw[p + "psi50"])
        return total

    return ModelSpec(
        name=f"I_gpp_{n_pft}pft",
        interp_source=interp_source,
        direct_source=direct_source,
        expression=expression,
        param_names=param_names,
        target_values=targets,
        input_names=input_names,
        ground_truth=gt,
        driver_ranges={d: DRIVER_RANGES[d] for d in input_names},
        param_bounds=bounds,
        is_recursive=False,
    )


def _default_covers(n_pft: int) -> list[float]:
    if n_pft == 2:
        return [0.6, 0.4]
    eq = 1.0 / n_pft
    return [eq] * n_pft


# --- Optional recursive carbon-pool model (Exp C idiom) ---------------------
# Used in the pilot ONLY to demonstrate that lambdify->JAX cannot ingest a
# recursive/stateful program without manual unrolling (pillar 2). A full DMCI
# fit of this model belongs to the main experiment, not the <2h pilot.

_POOL_STEPS = 15
_POOL_DT = 1.0


def build_recursive_pool_model() -> ModelSpec:
    """Annual leaf-carbon pool: C_{n+1} = C_n + dt*(GPP(Qbar) - tau*C_n)."""
    gpp = ("(/ (* alpha (* Qbar Amax)) (+ (* alpha Qbar) Amax))")
    step = (f"(pool (+ C (* {_POOL_DT!r} (- {gpp} (* tau C)))) (- n 1))")
    defn = (f"(define (pool C n) (if (= n 0) C {step}))")
    call = f"(pool 0.0 {_POOL_STEPS})"
    interp_source = (
        EVALUATOR_SOURCE + "\n(scheme-eval-program\n  (list\n    '" +
        defn + "\n    '" + call + ")\n  " +
        _make_env(["Qbar", "alpha", "Amax", "tau"]) + ")\n")
    direct_source = (
        f"(define (pool C n alpha Amax tau Qbar)\n"
        f"  (if (= n 0) C\n"
        f"    (pool (+ C (* {_POOL_DT!r} (- {gpp} (* tau C)))) (- n 1) "
        f"alpha Amax tau Qbar)))\n"
        f"(pool 0.0 {_POOL_STEPS} alpha Amax tau Qbar)\n")

    def gt(**kw) -> float:
        C = 0.0
        Qbar, alpha, Amax, tau = kw["Qbar"], kw["alpha"], kw["Amax"], kw["tau"]
        g = (alpha * Qbar * Amax) / (alpha * Qbar + Amax)
        for _ in range(_POOL_STEPS):
            C = C + _POOL_DT * (g - tau * C)
        return C

    return ModelSpec(
        name="I_carbon_pool",
        interp_source=interp_source,
        direct_source=direct_source,
        expression=defn + " " + call,   # NOT closed-form: lambdify will reject it
        param_names=["alpha", "Amax", "tau"],
        target_values={"alpha": 0.05, "Amax": 15.0, "tau": 0.1},
        input_names=["Qbar"],
        ground_truth=gt,
        driver_ranges={"Qbar": (100.0, 1800.0)},
        param_bounds={"alpha": (0.001, 0.5), "Amax": (1.0, 40.0),
                      "tau": (0.001, 1.0)},
        is_recursive=True,
    )
