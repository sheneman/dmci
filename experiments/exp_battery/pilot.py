############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# pilot.py: Battery capacity-fade DMCI de-risk pilot: three go/no-go checks. Run (local, CPU; never touches the cluster):...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Battery capacity-fade DMCI de-risk pilot: three go/no-go checks.

Run (local, CPU; never touches the cluster):
    python3 -m experiments.exp_battery.pilot --smoke      # fast plumbing pass
    python3 -m experiments.exp_battery.pilot              # full pilot (n_cells=8, both splits)

Checks
  (1) GRADIENT HEALTH   -- every structure folds; autograd d(NLL)/d(raw) matches central finite
                           differences; and the DMCI forward path (run_predict) reproduces the
                           closed-form forecast. Guards against the silent-0 op footgun.
  (2) LANDSCAPE RUGGED   -- fit all structures on early cycles, score the held-out tail; the
                           inter-structure spread says whether program structure is forecast-
                           decisive (build it) or flat (the FluZoo trap).
  (3) STRUCTURE RECOVERY -- on mechanism-labeled data, does the GENERATING structure win? The
                           confusion-matrix diagonal is the validation FluZoo could not offer.
"""

from __future__ import annotations

import sys
sys.setrecursionlimit(20000)

import argparse
import json
import os
import time

import numpy as np
import torch

from experiments.exp_fluzoo.programs import run_nll
from experiments.exp_fluzoo.paramspec import make_raw

from .config import BCFG, T_CYCLES, KSPLIT_LATE, KSPLIT_EARLY
from .structures import STRUCTURES, CAN_KNEE
from .synth import make_dataset, make_cells
from .score import _prog, score_pooled, dmci_predict_check

NAMES = list(STRUCTURES)
_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(_HERE, "results")


# --------------------------------------------------------------------------- (1)
def gradient_health(t_fit: int = 60) -> dict:
    """Finite-difference vs autograd on every structure + DMCI-forecast-path agreement."""
    print("\n=== (1) GRADIENT HEALTH ===", flush=True)
    rng = np.random.default_rng(0)
    rows = {}
    for name in NAMES:
        prog = _prog(name)
        obs, _ = make_cells(name, 1, rng, t_cycles=t_fit)
        cell = obs[0]                                  # [t_fit, 1]
        raw = make_raw(prog.specs, seed=0)
        # autograd grads
        nll = run_nll(prog, raw, cell, cfg=BCFG, grad=True)
        nll.backward()
        g_auto = {n: float(raw[n].grad) for n in prog.param_names}
        # central finite differences on each raw leaf
        eps = 1e-3
        max_rel = 0.0
        for n in prog.param_names:
            base = float(raw[n].detach())
            r2 = {m: raw[m].detach().clone().requires_grad_(True) for m in prog.param_names}
            r2[n] = torch.tensor(base + eps, requires_grad=True)
            fp = float(run_nll(prog, r2, cell, cfg=BCFG, grad=False))
            r2[n] = torch.tensor(base - eps, requires_grad=True)
            fm = float(run_nll(prog, r2, cell, cfg=BCFG, grad=False))
            g_fd = (fp - fm) / (2 * eps)
            denom = max(abs(g_fd), abs(g_auto[n]), 1.0)
            max_rel = max(max_rel, abs(g_fd - g_auto[n]) / denom)
        # DMCI forward path vs closed form
        hz = [t_fit // 2, t_fit - 1, t_fit + 20]
        chk = dmci_predict_check(name, raw, hz, cfg=BCFG)
        max_pred_diff = max(abs(d - c) for _, d, c in chk)
        finite = all(np.isfinite(list(g_auto.values())))
        rows[name] = {"max_grad_rel_err": max_rel, "max_predict_diff": max_pred_diff,
                      "grads_finite": bool(finite)}
        print(f"  {name:18s} grad_rel_err={max_rel:.2e}  predict_diff={max_pred_diff:.2e}  "
              f"finite={finite}", flush=True)
    ok = all(r["grads_finite"] and r["max_grad_rel_err"] < 5e-2 and r["max_predict_diff"] < 1e-3
             for r in rows.values())
    print(f"  -> gradient health: {'PASS' if ok else 'CHECK'}", flush=True)
    return {"rows": rows, "pass": ok}


# --------------------------------------------------------------------------- (2)+(3)
def confusion(ksplit: int, n_cells: int, seed: int, multistart: bool) -> dict:
    """Fit every structure to every mechanism's cells; held-out RMSE matrix M[gen][fit]."""
    print(f"\n=== (2)+(3) LANDSCAPE @ ksplit={ksplit} (fit [0,{ksplit}), forecast "
          f"[{ksplit},{T_CYCLES})) ===", flush=True)
    data = make_dataset(n_cells=n_cells, seed=seed)
    M = {g: {f: [] for f in NAMES} for g in NAMES}
    insample = {g: {f: [] for f in NAMES} for g in NAMES}
    iters = 60
    t0 = time.time()
    for g in NAMES:
        obs, _ = data[g]                          # [N, T, 1]
        for f in NAMES:
            sc = score_pooled(f, obs, ksplit, cfg=BCFG, iters=iters)
            M[g][f] = sc["holdout_rmse"]
            insample[g][f] = sc["insample_rmse"]
        print(f"  gen={g:18s} done ({time.time()-t0:5.0f}s)", flush=True)

    mean = {g: {f: float(np.mean(M[g][f])) for f in NAMES} for g in NAMES}
    ins = {g: {f: float(np.mean(insample[g][f])) for f in NAMES} for g in NAMES}

    # header
    print("\n  held-out RMSE  rows=GENERATING  cols=FITTED  (diagonal* = self-fit)")
    print("  " + " " * 18 + "".join(f"{f[:9]:>11s}" for f in NAMES))
    recovered = 0
    for g in NAMES:
        best_f = min(NAMES, key=lambda f: mean[g][f])
        ok = best_f == g
        recovered += ok
        cells = []
        for f in NAMES:
            tag = "*" if f == g else " "
            star = "<" if f == best_f else " "
            cells.append(f"{mean[g][f]:.4f}{tag}{star}")
        print(f"  {g:18s}" + "".join(f"{c:>11s}" for c in cells)
              + ("   OK" if ok else f"   -> {best_f}"))

    # ruggedness: in-sample vs held-out spread (FluZoo trap = flat held-out)
    print("\n  ruggedness (per generating structure):")
    rugged_rows = {}
    for g in NAMES:
        vals = np.array([mean[g][f] for f in NAMES])
        ins_vals = np.array([ins[g][f] for f in NAMES])
        ho_spread = float((vals.max() - vals.min()) / max(vals.min(), 1e-9))
        in_spread = float((ins_vals.max() - ins_vals.min()) / max(ins_vals.min(), 1e-9))
        self_rmse = mean[g][g]
        best_wrong = min(mean[g][f] for f in NAMES if f != g)
        margin = float((best_wrong - self_rmse) / max(self_rmse, 1e-9))
        rugged_rows[g] = {"holdout_rel_spread": ho_spread, "insample_rel_spread": in_spread,
                          "self_vs_bestwrong_margin": margin}
        print(f"    {g:18s} holdout_spread={ho_spread*100:6.0f}%  insample_spread="
              f"{in_spread*100:5.0f}%  self-vs-best-wrong margin={margin*100:+6.0f}%")

    mean_ho_spread = float(np.mean([r["holdout_rel_spread"] for r in rugged_rows.values()]))
    return {"ksplit": ksplit, "mean_rmse": mean, "insample_rmse": ins,
            "rugged": rugged_rows, "recovered": recovered, "n_structures": len(NAMES),
            "mean_holdout_rel_spread": mean_ho_spread}


def verdict(grad, late, early) -> None:
    print("\n=== VERDICT ===", flush=True)
    print(f"  gradient health: {'PASS' if grad['pass'] else 'CHECK'}")
    for tag, res in (("knee-visible (ksplit=100)", late), ("pre-knee (ksplit=60)", early)):
        if res is None:
            continue
        print(f"  [{tag}] structure recovery: {res['recovered']}/{res['n_structures']} "
              f"diagonal wins; mean held-out spread {res['mean_holdout_rel_spread']*100:.0f}%")
    rugged = (late and late["mean_holdout_rel_spread"] > 0.5)
    recov = (late and late["recovered"] >= 3)
    go = grad["pass"] and rugged and recov
    print(f"\n  -> {'GO' if go else 'REVIEW'}: substrate {'OK' if grad['pass'] else 'CHECK'}; "
          f"landscape {'RUGGED' if rugged else 'flat-ish'}; "
          f"recovery {'works' if recov else 'weak'}.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="fast plumbing pass (tiny)")
    ap.add_argument("--n-cells", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--multistart", action="store_true")
    ap.add_argument("--splits", default="100,60", help="comma-sep ksplit values")
    args = ap.parse_args()

    if args.smoke:
        args.n_cells = 2
        splits = [KSPLIT_LATE]
    else:
        splits = [int(s) for s in args.splits.split(",")]

    t0 = time.time()
    grad = gradient_health()
    results = {f"ksplit_{s}": confusion(s, args.n_cells, args.seed, args.multistart)
               for s in splits}
    late = results.get(f"ksplit_{KSPLIT_LATE}")
    early = results.get(f"ksplit_{KSPLIT_EARLY}")
    verdict(grad, late, early)

    os.makedirs(RESULTS, exist_ok=True)
    out = {"gradient_health": grad, "confusion": results,
           "config": {"t_cycles": T_CYCLES, "n_cells": args.n_cells, "seed": args.seed,
                      "multistart": args.multistart, "splits": splits},
           "elapsed_s": time.time() - t0}
    path = os.path.join(RESULTS, "pilot_smoke.json" if args.smoke else "pilot.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    print(f"\n[saved] {path}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
