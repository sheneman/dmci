############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# rescore.py: Rigorous re-score of the battery island winners + references (the FluZoo lesson applied). Search-time fitness...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Rigorous re-score of the battery island winners + references (the FluZoo lesson applied).

Search-time fitness used adam_iters=60; that proxy can flatter flexible models. Here we score the
3 evolved island winners AND the references (smooth sqrt seed, true two_reservoir, sigmoidal) at
BOTH iters=60 (search-time) and iters=300 (rigorous), on the same target, to (a) report defensible
held-out numbers and (b) confirm the recovered structure genuinely beats the smooth seed under
proper calibration (unlike FluZoo, where the advantage evaporated).
"""
import sys; sys.setrecursionlimit(20000)
import re, json, time
import torch

from experiments.exp_battery.oe_score import score_evolved
from experiments.exp_battery.structures import STRUCTURES
from experiments.exp_battery.config import KSPLIT_LATE

ROOT = "experiments/exp_battery"


def battery_model(path):
    src = open(path).read()
    m = re.search(r'BATTERY_MODEL\s*=\s*r?"""(.*?)"""', src, re.DOTALL)
    return m.group(1)


PROGS = {
    "sqrt_seed":         STRUCTURES["sqrt_t_SEI"],
    "two_reservoir_ref": STRUCTURES["two_reservoir_min"],
    "sigmoidal_ref":     STRUCTURES["sigmoidal_knee"],
    "evolved_island0":   battery_model(f"{ROOT}/results/bat_island_0/best/best_program.py"),
    "evolved_island1":   battery_model(f"{ROOT}/results/bat_island_1/best/best_program.py"),
    "evolved_island2":   battery_model(f"{ROOT}/results/bat_island_2/best/best_program.py"),
}

obs = torch.load(f"{ROOT}/results/target.pt")["obs"]
rows = {}
for name, src in PROGS.items():
    rows[name] = {}
    for iters in (60, 300):
        t = time.time()
        r = score_evolved(src, obs, KSPLIT_LATE, iters=iters)
        hr = r.get("holdout_rmse")
        rows[name][f"iters{iters}"] = hr
        hrs = f"{hr:.5f}" if hr is not None else "None"
        print(f"  {name:18s} iters={iters:3d} holdout={hrs} knee={r.get('knee_capable')} "
              f"({time.time()-t:.0f}s)", flush=True)
        # incremental dump: a time-limit kill must not lose completed evals (job 5150851 lost all)
        json.dump(rows, open(f"{ROOT}/results/battery_rescore.json", "w"), indent=2, default=float)
json.dump(rows, open(f"{ROOT}/results/battery_rescore.json", "w"), indent=2, default=float)
print("WROTE results/battery_rescore.json", flush=True)
