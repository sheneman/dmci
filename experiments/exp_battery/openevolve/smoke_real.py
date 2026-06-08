############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# smoke_real.py: Quick smoke test of the REAL Severson target.pt before committing the multi-hour island runs. Scores the...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Quick smoke test of the REAL Severson target.pt before committing the multi-hour island runs.

Scores the reference structures (smooth sqrt seed, two_reservoir knee, stretched_exp) through the
SAME DMCI fit-early/forecast-late path the OpenEvolve search uses, at a low iter count, on BOTH
ksplits. Confirms (a) the real obs wires through the scorer with finite NLL/forecasts, and (b) the
real cells DISCRIMINATE structure (a knee-capable family should out-forecast the smooth seed) --
a preview of the real-data forecast-skill claim.

    srun ... python3 -m experiments.exp_battery.openevolve.smoke_real
"""

from __future__ import annotations

import sys
import time

sys.setrecursionlimit(20000)

import torch

torch.set_num_threads(8)

from experiments.exp_battery.oe_score import score_evolved
from experiments.exp_battery.structures import STRUCTURES
from experiments.exp_battery.config import KSPLIT_LATE, KSPLIT_EARLY

ROOT = "experiments/exp_battery"
ITERS = 40
REFS = ["sqrt_t_SEI", "stretched_exp", "two_reservoir_min"]  # smooth seed, soft knee, sharp knee


def main():
    obs = torch.load(f"{ROOT}/results/target_real.pt")["obs"]
    print(f"[smoke_real] obs={tuple(obs.shape)}  iters={ITERS}")
    for ks, nm in [(KSPLIT_EARLY, "EARLY"), (KSPLIT_LATE, "LATE")]:
        print(f"\n===== ksplit={ks} ({nm}) =====")
        for s in REFS:
            t = time.time()
            r = score_evolved(STRUCTURES[s], obs, ks, iters=ITERS)
            if r.get("ok"):
                print(f"  {s:18s} holdout={r['holdout_rmse']:.5f} knee={r['knee_capable']} "
                      f"({time.time()-t:.0f}s)", flush=True)
            else:
                print(f"  {s:18s} FAILED @{r.get('stage')}: {r.get('detail')}", flush=True)


if __name__ == "__main__":
    main()
