############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# run_comparison.py: Exp I (re-scoped): does exact-gradient calibration through DMCI beat black-box? Compares, at EQUAL WALL-CLOCK,...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Exp I (re-scoped): does exact-gradient calibration through DMCI beat black-box?

Compares, at EQUAL WALL-CLOCK, swept over parameter count (n_pft → 6..30 params):
  - adam_singlestart  — the pilot's method (single-start Adam, raw params): the "before"
  - <method>_lbfgs_ms — reparam(log) + L-BFGS + multi-start: the fixed gradient method
  - diffevo           — differential evolution: the black-box baseline

Headline hypothesis (confirmed on the loss landscape, docs/): the fixed gradient method
matches/beats DE and the margin grows with parameter count, where DE/CMA-ES degrade. The
multi-start is cheap because each restart is a single batched DMCI walk (v1.1.7).

Run on HPC (CPU 'eight' partition):
  python -m experiments.exp_i.run_comparison --method dmci --budget-s 300
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from .config import ExpIConfig
from .models import build_static_model
from .harness import run_dmci, run_diffevo, run_lbfgs_multistart, heldout_mse


def run_comparison(param_counts, seeds, method, budget_s, output_dir, out_tag=""):
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = ExpIConfig()
    results = []
    for n_pft in param_counts:
        model = build_static_model(n_pft)
        npar = len(model.param_names)
        for seed in seeds:
            print(f"\n=== {npar} params (n_pft={n_pft}), seed {seed}, "
                  f"budget {budget_s:.0f}s/method ===")
            row = {"n_pft": n_pft, "n_params": npar, "seed": seed, "methods": {}}
            fits = [
                run_dmci(model, cfg, seed, time_budget=budget_s),             # naive Adam (budgeted)
                run_lbfgs_multistart(model, cfg, seed, method=method,
                                     n_starts=100_000, time_budget=budget_s),  # fixed GD
                run_diffevo(model, cfg, seed, time_budget=budget_s),           # black-box
            ]
            for r in fits:
                hmse = heldout_mse(model, r.fitted_values, cfg, seed)
                row["methods"][r.method] = {
                    "best_mse": r.best_mse, "heldout_mse": hmse,
                    "mean_param_rel_error":
                        sum(r.param_rel_error.values()) / len(r.param_rel_error),
                    "t_fit_s": r.t_fit_s, "n_iters": r.n_epochs,
                }
                print(f"  {r.method:22s} best_mse={r.best_mse:.3e} "
                      f"heldout={hmse:.3e} t={r.t_fit_s:.0f}s iters={r.n_epochs}")
            results.append(row)

    suffix = f"_{out_tag}" if out_tag else ""
    out = output_dir / f"comparison_{method}{suffix}.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nSaved: {out}")
    _summary(results)
    return results


def _summary(results):
    methods = []
    for row in results:
        for m in row["methods"]:
            if m not in methods:
                methods.append(m)
    by_pc = defaultdict(lambda: defaultdict(list))
    for row in results:
        for m, d in row["methods"].items():
            by_pc[row["n_params"]][m].append(d["heldout_mse"])
    print("\n=== mean held-out MSE by method × parameter count ===")
    print(f"{'#params':>8s} | " + " | ".join(f"{m:>20s}" for m in methods))
    for npar in sorted(by_pc):
        cells = []
        for m in methods:
            vals = by_pc[npar].get(m, [])
            cells.append(f"{(sum(vals)/len(vals)):>20.2e}" if vals else f"{'-':>20s}")
        print(f"{npar:>8d} | " + " | ".join(cells))


def main():
    ap = argparse.ArgumentParser(description="Exp I re-scoped comparison")
    ap.add_argument("--param-counts", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--method", default="dmci", choices=["dmci", "direct"],
                    help="gradient path for the L-BFGS multistart fitter")
    ap.add_argument("--budget-s", type=float, default=300.0,
                    help="equal wall-clock budget (s) per method per (model, seed)")
    ap.add_argument("--output-dir", type=Path,
                    default=Path("experiments/exp_i/results"))
    ap.add_argument("--out-tag", default="",
                    help="suffix for the output filename (e.g. pft11), so parallel "
                         "array tasks do not clobber a single comparison_*.json")
    args = ap.parse_args()
    sys.setrecursionlimit(5000)
    run_comparison(args.param_counts, args.seeds, args.method, args.budget_s,
                   args.output_dir, args.out_tag)


if __name__ == "__main__":
    main()
