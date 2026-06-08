############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# validity.py: The validity funnel and canonical (structural) de-duplication for FluZoo. Because the LLM writes whole...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""The validity funnel and canonical (structural) de-duplication for FluZoo.

Because the LLM writes whole programs, most of the discrete-search effort is spent
deciding which proposals are even runnable. The funnel turns each proposal into a
verdict and, on failure, a targeted repair hint fed back to the LLM:

    parse -> op-prescan -> compile -> free-var match -> finite forward
          -> finite & nonzero gradients -> stable rollout

The pass-rate at each stage is the central "funnel" figure of the experiment. The
canonical hash collapses programs that are structurally identical up to parameter
names and constant values, so 500 proposals reduce to a count of genuinely distinct
model structures (the real measure of the LLM's program-space exploration).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import torch

from neural_compiler.dmci import (
    INTERPRETER_OPS,
    INTERPRETER_SPECIAL_FORMS,
    UnsupportedOperatorError,
)
from neural_compiler.parser.scheme_parser import tokenize, _parse_sexpr

from .config import DEFAULT, GATE
from .paramspec import make_raw
from .programs import (
    FluProgram, parse_program, free_vars, unsupported_ops, run_nll, run_predict,
)

# Macro heads the DMCI front-end lowers before the interpreter sees them.
MACRO_HEADS = {"let*", "when", "unless", "loop", "recur", "vec", "mat"}
_KEYWORDS = INTERPRETER_OPS | INTERPRETER_SPECIAL_FORMS | MACRO_HEADS | {"quote", "obs"}

# Ordered funnel stages (also the columns of the funnel table).
STAGES = (
    "proposed", "parse", "op_prescan", "compile",
    "free_vars", "finite_forward", "finite_gradient", "stable_rollout", "forecastable",
)


def _is_number(tok) -> bool:
    if not isinstance(tok, str):
        return False
    try:
        float(tok)
        return True
    except ValueError:
        return False


def canonical_hash(text_or_prog) -> str:
    """Structural fingerprint: rename all non-keyword symbols positionally and bucket
    every numeric literal, so programs that differ only in parameter names / constants
    collapse to one hash. Operates on the model body (the schema block is ignored)."""
    prog = text_or_prog if isinstance(text_or_prog, FluProgram) else parse_program(text_or_prog)
    datum, _ = _parse_sexpr(tokenize(prog.body), 0)
    mapping: dict[str, str] = {}

    def walk(node):
        if isinstance(node, list):
            return [walk(x) for x in node]
        if _is_number(node):
            return "#"
        if node in _KEYWORDS:
            return node
        if node not in mapping:
            mapping[node] = f"s{len(mapping)}"
        return mapping[node]

    canon = walk(datum)
    return hashlib.sha256(repr(canon).encode()).hexdigest()[:16]


@dataclass
class FunnelResult:
    ok: bool
    stage: str                      # the furthest stage reached
    detail: str = ""
    repair_hint: str = ""
    n_params: int = 0
    canonical: str = ""
    prog: FluProgram | None = None
    grad_norm: float = 0.0
    nll0: float = float("nan")

    @property
    def stage_index(self) -> int:
        return STAGES.index(self.stage) if self.stage in STAGES else -1


def _fail(stage, detail, hint, prog=None) -> FunnelResult:
    return FunnelResult(ok=False, stage=stage, detail=detail, repair_hint=hint, prog=prog)


def screen(text: str, cfg=DEFAULT, gate=GATE, probe_obs: torch.Tensor | None = None,
           name: str = "anon") -> FunnelResult:
    """Run a proposed program text through the full validity funnel."""
    if probe_obs is None:  # small but realistic probe: a few weeks x all regions
        probe_obs = 0.01 * torch.rand(12, cfg.n_regions, dtype=torch.float32)
    Tp = int(probe_obs.shape[0])

    # ---- parse + schema (also parses the body, via canonical_hash) -------
    try:
        prog = parse_program(text, name=name)
        canon = canonical_hash(prog)   # parses the model body; catches unbalanced parens here
    except Exception as exc:  # noqa: BLE001
        return _fail("parse", f"{type(exc).__name__}: {exc}",
                     "Emit EXACTLY two top-level forms: a (params ...) schema then the "
                     "model expression, with BALANCED parentheses. Each param entry is "
                     "(name kind init [scale]) with kind in {positive, unit, signed-unit, free}.")
    if len(prog.specs) > gate.max_params:
        return FunnelResult(ok=False, stage="parse", canonical=canon, prog=prog,
                            n_params=len(prog.specs),
                            detail=f"too many parameters: {len(prog.specs)} > {gate.max_params}",
                            repair_hint=(f"Use at most {gate.max_params} scalar parameters; keep the "
                                         "model compact (fewer compartments/harmonics)."))

    # ---- static op prescan (the silent-evaluate-to-0 footgun) -----------
    try:
        bad = unsupported_ops(prog.body, Tp)
    except Exception as exc:  # noqa: BLE001
        return _fail("parse", f"prescan failed: {type(exc).__name__}: {exc}",
                     "The model body did not parse as a valid S-expression.", prog)
    if bad:
        return FunnelResult(ok=False, stage="op_prescan", canonical=canon, prog=prog,
                            n_params=len(prog.specs),
                            detail=f"unsupported operators: {sorted(bad)}",
                            repair_hint=(f"Operators {sorted(bad)} are NOT supported and would "
                                         "silently compute 0. Use only the allowed op surface."))

    # ---- compile ---------------------------------------------------------
    try:
        from .programs import get_graph
        get_graph(prog.body, Tp)
    except UnsupportedOperatorError as exc:
        return _fail("op_prescan", f"{exc}", "Remove the unsupported operator(s).", prog)
    except Exception as exc:  # noqa: BLE001
        return FunnelResult(ok=False, stage="compile", canonical=canon, prog=prog,
                            n_params=len(prog.specs),
                            detail=f"{type(exc).__name__}: {exc}",
                            repair_hint="The model body failed to compile; check parentheses, "
                                        "binary-only arithmetic, and that every let* body is a "
                                        "single expression.")

    # ---- free-variable contract -----------------------------------------
    declared = set(prog.param_names) | {"obs"}
    detected = free_vars(prog.body, Tp)
    missing = detected - declared        # used but not declared (and not obs)
    unused = set(prog.param_names) - detected
    if missing:
        return FunnelResult(ok=False, stage="free_vars", canonical=canon, prog=prog,
                            n_params=len(prog.specs),
                            detail=f"undeclared free variables: {sorted(missing)}",
                            repair_hint=(f"Every symbol the model uses must be either `obs`, a "
                                         f"loop/let* binding, or a declared parameter. Declare "
                                         f"{sorted(missing)} in (params ...) or remove them."))
    # (unused declared params are tolerated but penalised later via param_count.)

    # ---- finite forward --------------------------------------------------
    raw = make_raw(prog.specs, seed=0)
    try:
        nll0 = run_nll(prog, raw, probe_obs, cfg=cfg, grad=True)
    except Exception as exc:  # noqa: BLE001
        return FunnelResult(ok=False, stage="finite_forward", canonical=canon, prog=prog,
                            n_params=len(prog.specs),
                            detail=f"forward raised {type(exc).__name__}: {exc}",
                            repair_hint=("The rollout failed to evaluate. Ensure the predicted "
                                         "observable is an 11-vector matching obs rows (one entry "
                                         "per region), and that all tensor shapes are consistent."))
    if not torch.isfinite(nll0).all():
        return FunnelResult(ok=False, stage="finite_forward", canonical=canon, prog=prog,
                            n_params=len(prog.specs), nll0=float("nan"),
                            detail="forward NLL is not finite",
                            repair_hint="Keep the rollout bounded: positive rates in (0,1), a "
                                        "floored observation variance, no division by a value "
                                        "that can reach 0.")

    # ---- finite & nonzero gradients -------------------------------------
    nll0.backward()
    gsq = 0.0
    dead = []
    for n in prog.param_names:
        g = raw[n].grad
        if g is None or not torch.isfinite(g).all():
            return FunnelResult(ok=False, stage="finite_gradient", canonical=canon, prog=prog,
                                n_params=len(prog.specs), nll0=float(nll0.detach()),
                                detail=f"parameter {n!r} has a missing/non-finite gradient",
                                repair_hint="Make every parameter actually influence the loss "
                                            "through differentiable operations only.")
        gv = float(g)
        gsq += gv * gv
        if gv == 0.0:
            dead.append(n)
    grad_norm = gsq ** 0.5
    if grad_norm <= gate.grad_floor:
        return FunnelResult(ok=False, stage="finite_gradient", canonical=canon, prog=prog,
                            n_params=len(prog.specs), nll0=float(nll0.detach()), grad_norm=grad_norm,
                            detail=f"vanishing gradient (norm={grad_norm:.2e}); dead params {dead}",
                            repair_hint="The objective does not depend on the parameters; ensure "
                                        "parameters enter the predicted observable.")

    # ---- stable rollout across random parameter draws -------------------
    for s in range(1, gate.rollout_probes + 1):
        raw_s = make_raw(prog.specs, seed=s, jitter=0.5)
        try:
            nll_s = run_nll(prog, raw_s, probe_obs, cfg=cfg, grad=False)
        except Exception as exc:  # noqa: BLE001
            return FunnelResult(ok=False, stage="stable_rollout", canonical=canon, prog=prog,
                                n_params=len(prog.specs), nll0=float(nll0.detach()), grad_norm=grad_norm,
                                detail=f"unstable at draw {s}: {type(exc).__name__}: {exc}",
                                repair_hint="The rollout must stay finite across a range of "
                                            "parameter values, not just the initialisation.")
        if not torch.isfinite(nll_s).all():
            return FunnelResult(ok=False, stage="stable_rollout", canonical=canon, prog=prog,
                                n_params=len(prog.specs), nll0=float(nll0.detach()), grad_norm=grad_norm,
                                detail=f"non-finite NLL at random draw {s}",
                                repair_hint="The rollout diverges for some parameter values; add "
                                            "saturating/bounded dynamics.")

    # ---- forecastable: the PREDICT swap compiles and emits a finite observable ---
    R = int(probe_obs.shape[1])
    try:
        yhat = run_predict(prog, raw, week=min(Tp, 6), n_regions=R, cfg=cfg)
    except Exception as exc:  # noqa: BLE001
        return FunnelResult(ok=False, stage="forecastable", canonical=canon, prog=prog,
                            n_params=len(prog.specs), nll0=float(nll0.detach()), grad_norm=grad_norm,
                            detail=f"prediction failed: {type(exc).__name__}: {exc}",
                            repair_hint="The model must carry a `yhat` loop variable holding the "
                                        "predicted observable (an R-vector), with the loop base "
                                        "case returning the NLL accumulator `L`.")
    if yhat.numel() != R or not torch.isfinite(yhat).all():
        return FunnelResult(ok=False, stage="forecastable", canonical=canon, prog=prog,
                            n_params=len(prog.specs), nll0=float(nll0.detach()), grad_norm=grad_norm,
                            detail=f"prediction shape/finiteness wrong: numel={yhat.numel()} (want {R})",
                            repair_hint="`yhat` must be an R-vector (one predicted %ILI per region) "
                                        "and stay finite.")

    return FunnelResult(ok=True, stage="forecastable", canonical=canon, prog=prog,
                        n_params=len(prog.specs), nll0=float(nll0.detach()), grad_norm=grad_norm,
                        detail="accepted")
