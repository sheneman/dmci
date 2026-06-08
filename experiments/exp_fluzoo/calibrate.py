############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# calibrate.py: The inner-loop calibration: Adam over a program's unconstrained raw parameters. This is the validated fit loop...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""The inner-loop calibration: Adam over a program's unconstrained raw parameters.

This is the validated fit loop (the de-risk pilot's reparametrized descent): O(1)-scale
raw leaves, finite-guarded, grad-clipped, last-finite snapshot, |dNLL|<tol early stop.
It serves both the structural fit (all parameters, on the training window) and the
filter-then-forecast initial-condition refit (only the IC leaves trainable, started from
the structural fit). For robustness across the freely-generated zoo, callers can run it
from several seeds and keep the best -- the structure-agnostic analogue of exp_f's
portfolio, without assuming a fixed parameter layout.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from .config import DEFAULT
from .paramspec import make_raw
from .programs import FluProgram, run_nll, run_nll_batched


def eval_nll(prog: FluProgram, raw, obs, cfg=DEFAULT, grad: bool = True) -> torch.Tensor:
    """NLL over one obs window, or the SUM over a list of windows (multi-season fit).

    Autonomous epidemic models reset at k=0 each season with shared parameters, so the
    multi-season structural fit stacks the seasons into one batched [N, T, R] obs and folds
    them through the interpreter in a SINGLE shared-trajectory walk (run_nll_batched), then
    sums -- an ~N-fold speedup over looping. Seasons are truncated to the common length.
    """
    if isinstance(obs, (list, tuple)):
        if len(obs) == 1:
            return run_nll(prog, raw, obs[0], cfg=cfg, grad=grad)
        Tmin = min(int(w.shape[0]) for w in obs)
        try:  # fast path: all seasons in one shared-trajectory walk
            batch = torch.stack([w[:Tmin] for w in obs], dim=0)   # [N, Tmin, R]
            return run_nll_batched(prog, raw, batch, cfg=cfg, grad=grad).sum()
        except Exception:  # noqa: BLE001  (e.g. a data-dependent branch breaks batching)
            total = None    # correct fallback: loop the seasons
            for w in obs:
                v = run_nll(prog, raw, w[:Tmin], cfg=cfg, grad=grad)
                total = v if total is None else total + v
            return total
    return run_nll(prog, raw, obs, cfg=cfg, grad=grad)


@dataclass
class FitResult:
    raw: dict[str, torch.Tensor]
    nll: float
    iters: int
    converged: bool
    nan_stall: bool = False
    history: list[float] = field(default_factory=list)


def calibrate(prog: FluProgram, obs: torch.Tensor, cfg=DEFAULT, *, seed: int = 0,
              iters: int | None = None, lr: float | None = None,
              frozen: list[str] | None = None,
              init_raw: dict[str, torch.Tensor] | None = None) -> FitResult:
    """Fit `prog` to `obs` ([T,R]) by Adam on its raw leaves; return the best (last-finite)."""
    iters = cfg.adam_iters if iters is None else iters
    lr = cfg.lr if lr is None else lr
    frozen = set(frozen or ())

    if init_raw is None:
        raw = make_raw(prog.specs, seed=seed)
    else:  # start from a prior fit (IC refit freezes the structural leaves)
        raw = {n: init_raw[n].detach().clone().requires_grad_(True) for n in prog.param_names}
    for n in frozen:
        if n in raw:
            raw[n].requires_grad_(False)
    leaves = [raw[n] for n in prog.param_names if raw[n].requires_grad]
    if not leaves:
        return FitResult(raw=raw, nll=float(eval_nll(prog, raw, obs, cfg=cfg, grad=False)),
                         iters=0, converged=True)

    opt = torch.optim.Adam(leaves, lr=lr)
    best_nll, best_snap = math.inf, {n: raw[n].detach().clone() for n in prog.param_names}
    history: list[float] = []
    prev = math.inf
    converged = nan_stall = False
    it = 0
    for it in range(iters):
        opt.zero_grad()
        nll = eval_nll(prog, raw, obs, cfg=cfg, grad=True)
        v = float(nll.detach())
        if not math.isfinite(v):
            nan_stall = True
            break
        nll.backward()
        if any((g.grad is None) or (not torch.isfinite(g.grad).all()) for g in leaves):
            nan_stall = True
            break
        torch.nn.utils.clip_grad_norm_(leaves, cfg.grad_clip)
        opt.step()
        history.append(v)
        if v < best_nll:
            best_nll = v
            best_snap = {n: raw[n].detach().clone() for n in prog.param_names}
        if abs(prev - v) < cfg.conv_tol:
            converged = True
            break
        prev = v

    # restore the best (last finite) parameters
    for n in prog.param_names:
        raw[n] = best_snap[n].clone().requires_grad_(True)
    final = best_nll if math.isfinite(best_nll) else float("inf")
    return FitResult(raw=raw, nll=final, iters=it + 1, converged=converged,
                     nan_stall=nan_stall, history=history)


def calibrate_multistart(prog: FluProgram, obs: torch.Tensor, cfg=DEFAULT,
                         seeds=None, **kw) -> FitResult:
    """Run `calibrate` from several seeds and keep the lowest-NLL fit."""
    seeds = cfg.seeds if seeds is None else seeds
    best = None
    for s in seeds:
        r = calibrate(prog, obs, cfg=cfg, seed=s, **kw)
        if best is None or r.nll < best.nll:
            best = r
    return best
