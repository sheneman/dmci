############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# oe_score.py: Score an ARBITRARY evolved battery program (OpenEvolve candidate) end-to-end through DMCI. The de-risk scorer...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Score an ARBITRARY evolved battery program (OpenEvolve candidate) end-to-end through DMCI.

The de-risk scorer (score.py) read forecasts in closed form keyed on a known structure name; an
evolved program has no such name, so here the held-out forecast goes through the DMCI forward path
(run_predict_batched) -- the same FIT->PREDICT swap FluZoo uses. Protocol = fit-early / forecast-late:
batched per-cell calibration on cycles [0,ksplit) through the interpreter, then roll the fitted
autonomous model to each held-out cycle and read the predicted capacity. combined_score = -mean
held-out RMSE (OpenEvolve maximises).
"""

from __future__ import annotations

import math
import numpy as np
import torch

from experiments.exp_fluzoo.programs import (
    parse_program, run_nll_batched, run_predict_batched, get_graph, free_vars,
    unsupported_ops,
)
from experiments.exp_fluzoo.paramspec import make_raw, constrain

from .config import BCFG, N_SERIES


def screen(src: str, ksplit: int):
    """Validity funnel for an evolved battery program. Returns (ok, stage, detail)."""
    try:
        prog = parse_program(src, name="cand")
    except Exception as e:  # noqa: BLE001
        return False, "parse", f"{type(e).__name__}: {e}", None
    bad = unsupported_ops(prog.body, ksplit)
    if bad:
        return False, "ops", f"unsupported interpreter ops: {sorted(bad)} (silent-0 footgun)", None
    fv = free_vars(prog.body, ksplit) - {s.name for s in prog.specs} - {"obs"}
    if fv:
        return False, "freevars", f"undeclared free vars: {sorted(fv)}", None
    if "yhat" not in prog.body:
        return False, "yhat", "model must carry a `yhat` loop variable to forecast", None
    try:
        get_graph(prog.body, ksplit, mode="fit")
        get_graph(prog.body, ksplit + 1, mode="predict")
    except Exception as e:  # noqa: BLE001
        return False, "compile", f"{type(e).__name__}: {e}", None
    return True, "ok", "", prog


def _make_raw_batched(prog, n, seed=0):
    g = torch.Generator().manual_seed(seed)
    scal = make_raw(prog.specs, seed=seed)
    return {nm: (scal[nm].detach().reshape(1).repeat(n) + 0.02 * torch.randn(n, generator=g)).requires_grad_(True)
            for nm in prog.param_names}


def score_evolved(src: str, obs_batch: torch.Tensor, ksplit: int, cfg=BCFG,
                  iters: int = 60, lr: float = 0.1, hz_stride: int = 10) -> dict:
    """Fit a candidate to early cycles (batched DMCI) and score held-out forecast via run_predict.

    obs_batch is `[N, T, 1]`. Returns mean held-out RMSE + structural features, or an error.
    """
    ok, stage, detail, prog = screen(src, ksplit)
    if not ok:
        return {"ok": False, "stage": stage, "detail": detail}
    N, T = int(obs_batch.shape[0]), int(obs_batch.shape[1])
    early = obs_batch[:, :ksplit, :]
    raw = _make_raw_batched(prog, N)
    leaves = list(raw.values())
    opt = torch.optim.Adam(leaves, lr=lr)
    prev, best, snap = math.inf, math.inf, None
    for _ in range(iters):
        opt.zero_grad()
        nll = run_nll_batched(prog, raw, early, cfg=cfg, grad=True).sum()
        v = float(nll.detach())
        if not math.isfinite(v):
            break
        nll.backward()
        if any((g.grad is None) or (not torch.isfinite(g.grad).all()) for g in leaves):
            break
        torch.nn.utils.clip_grad_norm_(leaves, cfg.grad_clip)
        opt.step()
        if v < best:
            best, snap = v, {n: raw[n].detach().clone() for n in prog.param_names}
        if abs(prev - v) < cfg.conv_tol:
            break
        prev = v
    if snap is None:
        return {"ok": False, "stage": "fit", "detail": "non-finite NLL at init (unstable rollout)"}
    raw = {n: snap[n].requires_grad_(False) for n in prog.param_names}

    # held-out forecast through the DMCI predict path, at strided held-out cycles
    obs = obs_batch[:, :, 0].numpy().astype(np.float64)
    weeks = list(range(ksplit, T, hz_stride))
    if weeks and weeks[-1] != T - 1:
        weeks.append(T - 1)
    sq, cnt = 0.0, 0
    for w in weeks:
        try:
            pred = run_predict_batched(prog, raw, w, N_SERIES, N, cfg=cfg)[:, 0].numpy().astype(np.float64)
        except Exception:  # noqa: BLE001
            return {"ok": False, "stage": "forecast", "detail": f"predict failed at cycle {w}"}
        err = pred - obs[:, w]
        sq += float(np.sum(err ** 2)); cnt += err.size
    if cnt == 0 or not math.isfinite(sq):
        return {"ok": False, "stage": "forecast", "detail": "no finite held-out predictions"}
    holdout_rmse = math.sqrt(sq / cnt)
    feats = _features(prog)
    return {"ok": True, "holdout_rmse": holdout_rmse, "train_nll": best, **feats}


def _features(prog) -> dict:
    """Structural descriptors for MAP-Elites: loop-state size, parameter count, op richness."""
    from neural_compiler.parser.scheme_parser import tokenize, _parse_sexpr
    datum, _ = _parse_sexpr(tokenize(prog.body), 0)
    binds = datum[1] if isinstance(datum, list) and len(datum) > 1 else []
    state = [b[0] for b in binds if isinstance(b, list) and b and b[0] not in ("k", "yhat", "L")]
    has_knee = float(("min" in prog.body) or ("max" in prog.body) or ("exp" in prog.body))
    return {"n_state": float(len(state)), "n_params": float(len(prog.specs)),
            "n_nonlin": float(prog.body.count("(exp") + prog.body.count("(pow")
                              + prog.body.count("(sqrt") + prog.body.count("(min")),
            "knee_capable": has_knee}
