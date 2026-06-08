############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# rescore_knee.py: Score the hand KNEE references (two_reservoir_min, sigmoidal_knee) on the REAL Severson target at the rigorous...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Score the hand KNEE references (two_reservoir_min, sigmoidal_knee) on the REAL Severson target at
the rigorous 300-step budget. rescore_real dropped these to save time, but the co-search winner
(island2) is itself a knee structure, so the honest "beats the BEST hand model" claim must include
the hand knee families, not only the smooth ones. Shardable by split via BAT_RESCORE_KS.

    sbatch --export=ALL,BAT_RESCORE_KS=70 slurm_rescore.sh experiments.exp_battery.openevolve.rescore_knee
"""
from __future__ import annotations

import os
import sys
import json
import time

sys.setrecursionlimit(20000)

import torch

from experiments.exp_battery.oe_score import score_evolved
from experiments.exp_battery.structures import STRUCTURES
from experiments.exp_battery.config import KSPLIT_LATE, KSPLIT_EARLY

ROOT = "experiments/exp_battery"
ITERS = 300
_KS = os.environ.get("BAT_RESCORE_KS", "")
KSPLITS = [int(x) for x in _KS.split(",") if x] or [KSPLIT_LATE, KSPLIT_EARLY]
OUT = f"{ROOT}/results/battery_rescore_real_knee" + ("" if len(KSPLITS) > 1 else f"_ks{KSPLITS[0]}") + ".json"

PROGS = {"two_reservoir_ref": STRUCTURES["two_reservoir_min"],
         "sigmoidal_ref": STRUCTURES["sigmoidal_knee"]}


def main():
    obs = torch.load(f"{ROOT}/results/target_real.pt")["obs"]
    print(f"[rescore_knee] obs={tuple(obs.shape)} iters={ITERS} ksplits={KSPLITS}")
    rows = {}
    for ks in KSPLITS:
        tag = f"ksplit{ks}"
        rows[tag] = {}
        for nm, src in PROGS.items():
            t = time.time()
            r = score_evolved(src, obs, ks, iters=ITERS)
            hr = r.get("holdout_rmse")
            rows[tag][nm] = {"holdout_rmse": hr, "knee_capable": r.get("knee_capable"), "ok": r.get("ok")}
            hrs = f"{hr:.5f}" if hr is not None else f"FAIL@{r.get('stage')}"
            print(f"  {tag} {nm:18s} holdout={hrs} ({time.time()-t:.0f}s)", flush=True)
            json.dump(rows, open(OUT, "w"), indent=2, default=float)
    print(f"WROTE {OUT}", flush=True)


if __name__ == "__main__":
    main()
