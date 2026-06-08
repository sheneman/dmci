############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# aggregate_table.py: Aggregate Experiment D per-run results into Table 9 and verify manuscript values. Canonical source of truth:...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Aggregate Experiment D per-run results into Table 9 and verify manuscript values.

Canonical source of truth: experiments/exp_d/results/{gp_direct,gp_dmci}_seed0{0..4}.json
Each file holds a genetic-programming run: a `candidates` list of 1,000 evaluated
candidate programs, each with per-candidate `t_compile`, `t_train`, `t_total`
(seconds), plus `total_wall_time`, `best_loss`, `best_source`.

This script reconstructs every cell of Table 9 (tab:exp_d_timing, the per-candidate
cost decomposition) -- mean compile/train/total time, the %-compile fraction, and the
DMCI/direct overhead ratio -- and self-checks them against the values printed in the
manuscript so future drift is caught. It is the Experiment-D counterpart of
exp_b/aggregate_table.py and exp_c/aggregate_table.py.

Usage:
    python3 -m experiments.exp_d.aggregate_table     # print Table 9 + self-check

Notes:
- Five seeds x 1,000 candidates = 5,000 candidate evaluations per method.
- Per-candidate means are over all 5,000 candidates; the GP fitness trajectory is
  identical between gp_direct and gp_dmci (same math), so only timing differs.

If the per-run results are missing locally, on a fresh clone run `git lfs pull`, or
regenerate on the cluster with:
    python3 -m experiments.exp_d.exp_d --method both --seed <0..4>
"""
from __future__ import annotations

import glob
import json
from pathlib import Path
from statistics import mean

RESULTS_DIR = Path(__file__).parent / "results"
METHODS = [("gp_direct", "Direct"), ("gp_dmci", "DMCI")]

# Values printed in paper-dmci/paper2.tex Table 9 (tab:exp_d_timing), for drift detection.
# (t_compile_ms, t_train_ms, t_total_ms, pct_compile, ratio_vs_direct)
PAPER_TABLE9 = {
    "gp_direct": (4.1, 148, 152, 2.7, 1.0),
    "gp_dmci": (22.9, 3824, 3846, 0.6, 25.3),
}


def load_candidates(method):
    files = sorted(RESULTS_DIR.glob(f"{method}_seed*.json"))
    if not files:
        raise SystemExit(
            f"No exp_d results in {RESULTS_DIR}. On a fresh clone run `git lfs pull` first, or "
            "regenerate on the cluster with:\n"
            "  python3 -m experiments.exp_d.exp_d --method both --seed <0..4>")
    cands = []
    for f in files:
        cands.extend(json.loads(f.read_text())["candidates"])
    return cands


def main():
    checks = []
    direct_total = dmci_total = None
    print(f"{'Method':8} {'t_compile':>10} {'t_train':>9} {'t_total':>9} {'%compile':>9} {'Ratio':>7} {'n':>7}")
    print("-" * 64)
    for method, label in METHODS:
        c = load_candidates(method)
        tc = mean(x["t_compile"] for x in c) * 1000.0
        tt = mean(x["t_train"] for x in c) * 1000.0
        to = mean(x["t_total"] for x in c) * 1000.0
        pct = tc / to * 100.0
        if method == "gp_direct":
            direct_total = to
        else:
            dmci_total = to
        ratio = to / direct_total
        print(f"{label:8} {tc:9.1f}m {tt:8.0f}m {to:8.0f}m {pct:8.1f}% {ratio:6.1f}x {len(c):>7}")

        ptc, ptt, pto, ppct, pr = PAPER_TABLE9[method]
        checks.append((f"{label} t_compile", abs(round(tc, 1) - ptc) < 0.05))
        checks.append((f"{label} t_train", abs(round(tt) - ptt) <= 1))
        checks.append((f"{label} t_total", abs(round(to) - pto) <= 1))
        checks.append((f"{label} %compile", abs(round(pct, 1) - ppct) < 0.05))
        checks.append((f"{label} ratio", abs(ratio - pr) < 0.1))

    fails = [n for n, ok in checks if not ok]
    print(f"\nSELF-CHECK vs manuscript: {len(checks) - len(fails)}/{len(checks)} cells match")
    for n in fails:
        print(f"  MISMATCH: {n}")
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
