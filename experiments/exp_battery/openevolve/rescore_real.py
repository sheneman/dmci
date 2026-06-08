############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# rescore_real.py: Rigorous re-score of the REAL-DATA battery island winners vs baselines (the FluZoo lesson). On real Severson...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Rigorous re-score of the REAL-DATA battery island winners vs baselines (the FluZoo lesson).

On real Severson cells there is no ground-truth mechanism, so the claim is HELD-OUT FORECAST SKILL:
the LLM+DMCI-evolved program must out-forecast (a) the smooth structure families fit through the
SAME DMCI path -- the apples-to-apples internal baselines -- and (b) trivial numerical baselines
(persistence, linear extrapolation of the fit window). We score at adam_iters=300 (rigorous, the
search used 60) at BOTH ksplits (LATE=70, EARLY=45) so any fitness-fidelity flattery is exposed.

    sbatch ... rescore_real (or: python3 -m experiments.exp_battery.openevolve.rescore_real)
"""

from __future__ import annotations

import os
import re
import sys
import time

sys.setrecursionlimit(20000)

import json
import numpy as np
import torch

from experiments.exp_battery.oe_score import score_evolved
from experiments.exp_battery.structures import STRUCTURES
from experiments.exp_battery.config import KSPLIT_LATE, KSPLIT_EARLY

ROOT = "experiments/exp_battery"
ITERS = 300
# Shardable by split: BAT_RESCORE_KS=70 runs one ksplit per node (the iters=300 evals are ~20 min
# each, so splitting the two ksplits across two nodes halves wall-clock). Default = both splits.
_KS = os.environ.get("BAT_RESCORE_KS", "")
KSPLITS = [int(x) for x in _KS.split(",") if x] or [KSPLIT_LATE, KSPLIT_EARLY]
OUT = f"{ROOT}/results/battery_rescore_real" + ("" if len(KSPLITS) > 1 else f"_ks{KSPLITS[0]}") + ".json"


def battery_model(path: str) -> str:
    src = open(path).read()
    m = re.search(r'BATTERY_MODEL\s*=\s*r?"""(.*?)"""', src, re.DOTALL)
    return m.group(1)


def naive_baselines(obs: np.ndarray, ksplit: int) -> dict:
    """Trivial numerical floors on the SAME fit-early/forecast-late split (no DMCI).

    persistence: hold the last fit-window value flat. linear: extrapolate a line fit to the
    fit window. RMSE over the strided held-out tail to match score_evolved's horizon set."""
    N, T = obs.shape
    weeks = list(range(ksplit, T, 10))
    if weeks and weeks[-1] != T - 1:
        weeks.append(T - 1)
    w = np.array(weeks)
    x = np.arange(ksplit)
    out = {}
    # persistence
    pred_p = np.repeat(obs[:, ksplit - 1:ksplit], len(w), axis=1)          # [N, |w|]
    out["persistence"] = float(np.sqrt(np.mean((pred_p - obs[:, w]) ** 2)))
    # linear extrapolation of the fit window
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, obs[:, :ksplit].T, rcond=None)            # [2, N]
    pred_l = (coef[0][:, None] * w[None, :] + coef[1][:, None])            # [N, |w|]
    out["linear_extrap"] = float(np.sqrt(np.mean((pred_l - obs[:, w]) ** 2)))
    return out


def main():
    obs_t = torch.load(f"{ROOT}/results/target_real.pt")["obs"]            # [N, T, 1]
    obs_np = obs_t[:, :, 0].numpy().astype(np.float64)
    print(f"[rescore_real] obs={tuple(obs_t.shape)}  iters={ITERS}")

    # Real-data baselines = the smooth structure families through the SAME interpreter (the
    # synthetic-specific two_reservoir/sigmoidal refs are dropped here; the evolved island winners
    # are the knee structures of interest). Naive persistence/linear floors are added per split.
    progs = {
        "sqrt_seed":     STRUCTURES["sqrt_t_SEI"],
        "power_law_ref": STRUCTURES["power_law_kp"],
        "stretched_ref": STRUCTURES["stretched_exp"],
    }
    for i in range(3):
        p = f"{ROOT}/results/bat_real_island_{i}/best/best_program.py"
        if os.path.exists(p):
            progs[f"evolved_island{i}"] = battery_model(p)
        else:
            print(f"  (island {i} winner not found at {p} -- skipping)")

    rows: dict = {}
    for ks in KSPLITS:
        tag = f"ksplit{ks}"
        rows[tag] = {"naive": naive_baselines(obs_np, ks)}
        print(f"\n===== {tag} =====  naive: {rows[tag]['naive']}")
        for name, src in progs.items():
            t = time.time()
            r = score_evolved(src, obs_t, ks, iters=ITERS)
            hr = r.get("holdout_rmse")
            rows[tag][name] = {"holdout_rmse": hr, "knee_capable": r.get("knee_capable"),
                               "ok": r.get("ok"), "stage": r.get("stage"), "detail": r.get("detail")}
            hrs = f"{hr:.5f}" if hr is not None else f"FAIL@{r.get('stage')}"
            print(f"  {name:18s} holdout={hrs} knee={r.get('knee_capable')} "
                  f"({time.time()-t:.0f}s)", flush=True)
            # incremental dump: a time-limit kill must not lose completed evals
            json.dump(rows, open(OUT, "w"), indent=2, default=float)
    json.dump(rows, open(OUT, "w"), indent=2, default=float)
    print(f"\nWROTE {OUT}", flush=True)


if __name__ == "__main__":
    main()
