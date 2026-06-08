############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# aggregate_scaling.py: Aggregate the Experiment I d-scaling job-array outputs. Merges every...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Aggregate the Experiment I d-scaling job-array outputs.

Merges every results/scaling/comparison_<method>_pft*.json the array produced into a
single table (mean held-out MSE and mean per-fit wall-clock by parameter count x method),
writes a combined JSON, and emits a pgfplots-ready .dat (held-out MSE vs d per method)
for the manuscript figure.

    python -m experiments.exp_i.aggregate_scaling
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

RES = Path("experiments/exp_i/results/scaling")


def main():
    files = sorted(RES.glob("comparison_*_pft*.json"))
    rows = []
    for f in files:
        rows.extend(json.loads(f.read_text()))
    if not rows:
        print(f"No per-task result files found in {RES}/ (comparison_*_pft*.json).")
        return

    methods = []
    for r in rows:
        for m in r["methods"]:
            if m not in methods:
                methods.append(m)

    mse = defaultdict(lambda: defaultdict(list))
    tfit = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for m, d in r["methods"].items():
            mse[r["n_params"]][m].append(d["heldout_mse"])
            tfit[r["n_params"]][m].append(d["t_fit_s"])

    def table(title, store, fmt):
        print(f"\n=== {title} ===")
        print(f"{'d':>5s} {'n_pft':>6s} | " + " | ".join(f"{m:>20s}" for m in methods))
        for npar in sorted(store):
            cells = []
            for m in methods:
                vals = store[npar].get(m, [])
                cells.append(format(sum(vals) / len(vals), fmt).rjust(20)
                             if vals else f"{'-':>20s}")
            print(f"{npar:>5d} {npar // 6:>6d} | " + " | ".join(cells))

    table("mean held-out MSE by parameter count x method", mse, ".2e")
    table("mean per-fit wall-clock (s) by parameter count x method", tfit, ".1f")

    # combined JSON
    (RES / "scaling_combined.json").write_text(json.dumps(rows, indent=2, default=str))

    # pgfplots .dat: one row per d, one column per method (mean held-out MSE)
    lines = ["d " + " ".join(methods)]
    for npar in sorted(mse):
        cells = []
        for m in methods:
            vals = mse[npar].get(m, [])
            cells.append(f"{sum(vals) / len(vals):.6e}" if vals else "nan")
        lines.append(f"{npar} " + " ".join(cells))
    (RES / "exp_i_scaling.dat").write_text("\n".join(lines) + "\n")

    print(f"\nWrote {RES}/scaling_combined.json and {RES}/exp_i_scaling.dat "
          f"({len(rows)} rows from {len(files)} task files).")


if __name__ == "__main__":
    main()
