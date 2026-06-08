############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# analyze.py: Analyze Experiment D results: crossover plot data and summary tables. Usage: python3 -m...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Analyze Experiment D results: crossover plot data and summary tables.

Usage:
    python3 -m experiments.exp_d.analyze --results-dir experiments/exp_d/results
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_results(results_dir: Path) -> dict[str, list[dict]]:
    """Load all result JSONs, grouped by method."""
    by_method: dict[str, list[dict]] = {}
    for path in sorted(results_dir.glob("*.json")):
        with open(path) as f:
            data = json.load(f)
        method = data["method"]
        if method not in by_method:
            by_method[method] = []
        by_method[method].append(data)
    return by_method


def cumulative_times(candidates: list[dict]) -> list[dict]:
    """Compute cumulative compile, train, and total time per candidate."""
    cum_compile = 0.0
    cum_train = 0.0
    cum_total = 0.0
    result = []
    for i, c in enumerate(candidates):
        cum_compile += c["t_compile"]
        cum_train += c["t_train"]
        cum_total += c["t_total"]
        result.append({
            "candidate_idx": i,
            "cum_compile": cum_compile,
            "cum_train": cum_train,
            "cum_total": cum_total,
        })
    return result


def compute_crossover_table(direct_cands: list[dict],
                            dmci_cands: list[dict]) -> list[dict]:
    """Compute Table D1: timing comparison at fixed candidate budgets."""
    budgets = [5, 10, 20, 50, 100, 200, 500, 1000]
    rows = []
    for n in budgets:
        if n > len(direct_cands) or n > len(dmci_cands):
            break

        d_compile = sum(c["t_compile"] for c in direct_cands[:n])
        d_train = sum(c["t_train"] for c in direct_cands[:n])
        m_compile = sum(c["t_compile"] for c in dmci_cands[:n])
        m_train = sum(c["t_train"] for c in dmci_cands[:n])

        # "Cached DMCI": evaluator compiled once, only first candidate
        # pays compile cost
        m_compile_cached = dmci_cands[0]["t_compile"] if n > 0 else 0.0

        rows.append({
            "n_candidates": n,
            "direct_compile": d_compile,
            "direct_train": d_train,
            "direct_total": d_compile + d_train,
            "dmci_compile": m_compile,
            "dmci_train": m_train,
            "dmci_total": m_compile + m_train,
            "dmci_cached_compile": m_compile_cached,
            "dmci_cached_total": m_compile_cached + m_train,
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Analyze Experiment D results")
    parser.add_argument("--results-dir", type=Path,
                        default=Path("experiments/exp_d/results"))
    args = parser.parse_args()

    by_method = load_results(args.results_dir)
    if not by_method:
        print("No results found.", file=sys.stderr)
        return

    for method, runs in by_method.items():
        print(f"\n{'='*60}")
        print(f"Method: {method} ({len(runs)} seeds)")
        print(f"{'='*60}")

        for run in runs:
            cands = run["candidates"]
            compile_times = [c["t_compile"] for c in cands]
            train_times = [c["t_train"] for c in cands]
            losses = [c["final_loss"] for c in cands]

            avg_compile = sum(compile_times) / len(compile_times)
            avg_train = sum(train_times) / len(train_times)
            total = sum(c["t_total"] for c in cands)

            print(f"\n  Seed {run['seed']}:")
            print(f"    Candidates: {run['n_candidates_evaluated']}")
            print(f"    Avg compile: {avg_compile*1000:.1f}ms")
            print(f"    Avg train:   {avg_train*1000:.1f}ms")
            print(f"    Total time:  {total:.1f}s")
            print(f"    Best loss:   {run['best_loss']:.6f}")
            print(f"    Best prog:   {run['best_source']}")

    # Cross-method comparison
    if "gp_direct" in by_method and "gp_dmci" in by_method:
        print(f"\n{'='*60}")
        print("Cross-method comparison (averaged over seeds)")
        print(f"{'='*60}")

        direct_runs = by_method["gp_direct"]
        dmci_runs = by_method["gp_dmci"]

        n_direct = min(len(r["candidates"]) for r in direct_runs)
        n_dmci = min(len(r["candidates"]) for r in dmci_runs)

        avg_direct = _avg_candidates(direct_runs, n_direct)
        avg_dmci = _avg_candidates(dmci_runs, n_dmci)

        table = compute_crossover_table(avg_direct, avg_dmci)

        print(f"\n{'N':>6s} | {'Direct':>12s} | {'DMCI':>12s} | "
              f"{'DMCI(cached)':>12s} | {'D/DMCI':>8s} | {'D/DMCI-C':>8s}")
        print("-" * 72)
        for row in table:
            n = row["n_candidates"]
            d = row["direct_total"]
            m = row["dmci_total"]
            mc = row["dmci_cached_total"]
            ratio_m = d / m if m > 0 else float("inf")
            ratio_mc = d / mc if mc > 0 else float("inf")
            print(f"{n:6d} | {d:10.2f}s | {m:10.2f}s | "
                  f"{mc:10.2f}s | {ratio_m:8.2f} | {ratio_mc:8.2f}")

        # Save crossover data for plotting
        crossover_path = args.results_dir / "crossover_data.json"
        with open(crossover_path, "w") as f:
            json.dump({
                "table": table,
                "direct_cumulative": cumulative_times(avg_direct),
                "dmci_cumulative": cumulative_times(avg_dmci),
            }, f, indent=2)
        print(f"\nCrossover data saved to {crossover_path}")


def _avg_candidates(runs: list[dict], n: int) -> list[dict]:
    """Average candidate stats across seeds."""
    n_runs = len(runs)
    result = []
    for i in range(n):
        avg = {
            "t_compile": sum(r["candidates"][i]["t_compile"] for r in runs) / n_runs,
            "t_train": sum(r["candidates"][i]["t_train"] for r in runs) / n_runs,
            "t_total": sum(r["candidates"][i]["t_total"] for r in runs) / n_runs,
            "final_loss": sum(r["candidates"][i]["final_loss"] for r in runs) / n_runs,
        }
        result.append(avg)
    return result


if __name__ == "__main__":
    main()
