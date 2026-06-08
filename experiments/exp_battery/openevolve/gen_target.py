############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# gen_target.py: Generate + save the SEARCH TARGET: mechanism-labeled synthetic capacity-fade cells. The OpenEvolve search fits...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Generate + save the SEARCH TARGET: mechanism-labeled synthetic capacity-fade cells.

The OpenEvolve search fits each candidate program to these cells' early cycles and is scored on
held-out forecast skill. Default target = the two_reservoir (knee) mechanism, so the headline
result is RECOVERY: starting from a smooth sqrt seed, can the LLM+DMCI co-search rediscover a
knee-capable structure and out-forecast the smooth families? (Ground truth FluZoo cannot offer.)

    python3 -m experiments.exp_battery.openevolve.gen_target --mechanism two_reservoir_min --cells 12
"""

from __future__ import annotations

import argparse
import os
import torch

from ..synth import make_cells
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, "..", "results"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mechanism", default="two_reservoir_min")
    ap.add_argument("--cells", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(RESULTS, "target.pt"))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    obs, truths = make_cells(args.mechanism, args.cells, rng)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"obs": obs, "mechanism": args.mechanism, "truths": truths}, args.out)
    print(f"[gen_target] {args.cells} cells of '{args.mechanism}', obs {tuple(obs.shape)} -> {args.out}")


if __name__ == "__main__":
    main()
