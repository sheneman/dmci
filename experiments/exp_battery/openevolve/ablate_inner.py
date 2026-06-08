############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# ablate_inner.py: Inner-fitter ablation: swap DMCI exact-gradient Adam for a GRADIENT-FREE optimizer (differential evolution, the...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Inner-fitter ablation: swap DMCI exact-gradient Adam for a GRADIENT-FREE optimizer (differential
evolution, the same family as Exp I), holding EVERYTHING ELSE identical -- the OpenEvolve-discovered
structure, the batched interpreter forward, the per-cell parameterization, the data, the held-out
scoring, and a matched wall-clock budget. This isolates DMCI's exact-gradient contribution from
OpenEvolve's structure contribution: if gradient-free cannot reach DMCI's held-out skill on the SAME
structures in matched time, DMCI's gradients were load-bearing, not just OpenEvolve's structures.

Battery's fit is PER-CELL: each of N cells gets its own params, so the joint inner problem is
N*n_params (~585) dimensional -- the high-dim regime where Exp I shows gradient-free erodes. Compare
the DE held-out RMSE printed here against the DMCI iters=300 numbers in battery_rescore_real_ks*.json.

    sbatch --export=ALL,BAT_RESCORE_KS=70 slurm_rescore.sh experiments.exp_battery.openevolve.ablate_inner
"""
from __future__ import annotations

import os
import re
import sys
import json
import math
import time

sys.setrecursionlimit(20000)

import numpy as np
import torch

from experiments.exp_battery.oe_score import screen, _make_raw_batched
from experiments.exp_fluzoo.programs import run_nll_batched, run_predict_batched
from experiments.exp_battery.config import BCFG, N_SERIES, KSPLIT_LATE, KSPLIT_EARLY

ROOT = "experiments/exp_battery"
_KS = os.environ.get("BAT_RESCORE_KS", "")
KSPLITS = [int(x) for x in _KS.split(",") if x] or [KSPLIT_LATE, KSPLIT_EARLY]
BUDGET = float(os.environ.get("BAT_ABL_BUDGET_S", "1000"))   # matched wall-clock per (program, split)
HZ = 10
OUT = f"{ROOT}/results/battery_ablate_inner" + ("" if len(KSPLITS) > 1 else f"_ks{KSPLITS[0]}") + ".json"


def model_body(path: str) -> str:
    s = open(path).read()
    return re.search(r'BATTERY_MODEL\s*=\s*r?"""(.*?)"""', s, re.DOTALL).group(1)


PROGS = {f"island{i}": model_body(f"{ROOT}/results/bat_real_island_{i}/best/best_program.py")
         for i in range(3)}
obs = torch.load(f"{ROOT}/results/target_real.pt")["obs"]
N, T = int(obs.shape[0]), int(obs.shape[1])
OBSN = obs[:, :, 0].numpy().astype(np.float64)


def holdout_rmse(prog, raw, ks):
    weeks = list(range(ks, T, HZ))
    if weeks and weeks[-1] != T - 1:
        weeks.append(T - 1)
    sq, cnt = 0.0, 0
    for w in weeks:
        pred = run_predict_batched(prog, raw, w, N_SERIES, N, cfg=BCFG)[:, 0].numpy().astype(np.float64)
        e = pred - OBSN[:, w]
        sq += float((e ** 2).sum()); cnt += e.size
    return math.sqrt(sq / cnt)


def de_fit(prog, ks, seed=0):
    """Gradient-free differential evolution over the SAME per-cell raw params, matched wall-clock."""
    early = obs[:, :ks, :]
    raw0 = _make_raw_batched(prog, N, seed=seed)
    names = list(raw0.keys())
    x0 = np.concatenate([raw0[n].detach().numpy().astype(np.float64) for n in names])
    D = x0.size

    def to_raw(x):
        out, i = {}, 0
        for n in names:
            out[n] = torch.tensor(x[i:i + N], dtype=torch.float32)
            i += N
        return out

    def f(x):
        with torch.no_grad():
            v = float(run_nll_batched(prog, to_raw(x), early, cfg=BCFG, grad=False).sum())
        return v if math.isfinite(v) else 1e12

    rng = np.random.default_rng(seed)
    POP, F, CR = 24, 0.6, 0.9
    pop = x0[None, :] + 0.3 * rng.standard_normal((POP, D))
    pop[0] = x0
    fit = np.array([f(ind) for ind in pop])
    t0 = time.time(); nev = POP
    while time.time() - t0 < BUDGET:
        for j in range(POP):
            r = rng.choice(POP, 3, replace=False)
            a, b, c = pop[r[0]], pop[r[1]], pop[r[2]]
            cross = rng.random(D) < CR
            trial = np.where(cross, a + F * (b - c), pop[j])
            ft = f(trial); nev += 1
            if ft < fit[j]:
                pop[j], fit[j] = trial, ft
            if time.time() - t0 >= BUDGET:
                break
    best = pop[int(np.argmin(fit))]
    raw_best = {n: t.requires_grad_(False) for n, t in to_raw(best).items()}
    return raw_best, float(fit.min()), D, nev


def main():
    print(f"[ablate_inner] obs={tuple(obs.shape)} ksplits={KSPLITS} budget={BUDGET}s/fit (matched to DMCI)")
    rows = {}
    for ks in KSPLITS:
        tag = f"ksplit{ks}"
        rows[tag] = {}
        for nm, src in PROGS.items():
            ok, stage, detail, prog = screen(src, ks)
            if not ok:
                rows[tag][nm] = {"ok": False, "detail": detail}
                continue
            t = time.time()
            raw, nllbest, D, nev = de_fit(prog, ks)
            try:
                hr = holdout_rmse(prog, raw, ks)
            except Exception as e:  # noqa: BLE001
                hr = None
            rows[tag][nm] = {"de_holdout_rmse": hr, "de_train_nll": nllbest, "dim": D,
                             "de_fevals": nev, "sec": round(time.time() - t, 1)}
            hrs = f"{hr:.5f}" if hr is not None else "None"
            print(f"  {tag} {nm:9s} DIM={D} DE holdout={hrs} nfev={nev} ({time.time()-t:.0f}s)", flush=True)
            json.dump(rows, open(OUT, "w"), indent=2, default=float)
    print(f"WROTE {OUT}", flush=True)


if __name__ == "__main__":
    main()
