############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# run_expj.py: Exp J: Program-Space Calibration — DMCI vs. compile-each-program (JAX/lambdify). Thesis: when the object of...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Exp J: Program-Space Calibration — DMCI vs. compile-each-program (JAX/lambdify).

Thesis: when the object of optimization is a growing space of distinct, runtime-generated
programs, one compiled interpreter (DMCI) beats compile-each-program workflows on amortized
compile cost, per-structure engineering, and coverage — and reaches recursive/stateful
programs that automatic Scheme->JAX (lambdify) cannot follow.

Arms: DMCI (compile interpreter once), B1 (lambdify->jax.grad, automatic/closed-form-only),
B2 (hand-port to JAX, full coverage at per-structure LOC). Measures four curves vs N (number
of distinct programs), at recursive fractions {0%, 100%}: cumulative compile time, per-structure
engineering, coverage, and matched recovery error (so the comparison is about cost, not accuracy).

HONEST framing baked in: on closed-form programs DMCI has NO per-evaluation advantage over B1
(B1 is fast and automatic) — the advantages are (a) the recursive fraction lambdify can't express,
(b) the amortized one-compile-vs-N-compiles crossover, (c) uniform 100% coverage. DMCI runs the N
programs sequentially through one interpreter (compile amortized); it does NOT execute N distinct
programs in one vectorized walk (that is the batched program-VM, future work).

Run on HPC: python -m experiments.exp_j.run_expj --Ns 1 100 10000 --fractions 0 1
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

from .corpus import generate_corpus
from .arms import dmci_setup, dmci_prepare, b1_prepare, b2_prepare, recover

_ARMS = ["dmci", "b1", "b2"]


def _maxrss_mb() -> float:
    kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return kb / (1024 ** 2 if sys.platform == "darwin" else 1024)  # macOS bytes, Linux KB


def run(Ns, fractions, recover_sample, recover_budget, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    dmci = dmci_setup()
    print(f"DMCI one-time interpreter compile: {dmci['one_time_compile_s']:.2f}s, "
          f".ncg = {dmci['ncg_bytes']/1024:.0f} KB (constant in N)")

    out = {"dmci_one_time_compile_s": dmci["one_time_compile_s"],
           "ncg_bytes": dmci["ncg_bytes"], "cells": [], "recovery": []}

    for frac in fractions:
        Nmax = max(Ns)
        print(f"\n### recursive_fraction = {frac:.0%}  (generating {Nmax} distinct programs) ###")
        corpus = generate_corpus(Nmax, frac, seed=0)
        prep = {a: [] for a in _ARMS}
        prep_wall = {a: 0.0 for a in _ARMS}
        for j, prog in enumerate(corpus):
            for a, fn in (("dmci", lambda p: dmci_prepare(p, dmci["interp"])),
                          ("b1", b1_prepare), ("b2", b2_prepare)):
                t0 = time.perf_counter()
                prep[a].append(fn(prog))
                prep_wall[a] += time.perf_counter() - t0
            if (j + 1) % max(1, Nmax // 10) == 0:
                print(f"  prepared {j+1}/{Nmax}  rss={_maxrss_mb():.0f}MB")

        for N in Ns:
            cell = {"recursive_fraction": frac, "N": N, "arms": {}}
            for a in _ARMS:
                pp = prep[a][:N]
                one_time = dmci["one_time_compile_s"] if a == "dmci" else 0.0
                cell["arms"][a] = {
                    "cumulative_compile_s": one_time + sum(p["compile_s"] for p in pp),
                    "cumulative_eng_loc": sum(p["eng_loc"] for p in pp),
                    "coverage": sum(1 for p in pp if p["covered"]) / N,
                }
            out["cells"].append(cell)
            d = cell["arms"]
            print(f"  N={N:6d} | compile_s  DMCI={d['dmci']['cumulative_compile_s']:.2f} "
                  f"B1={d['b1']['cumulative_compile_s']:.2f} B2={d['b2']['cumulative_compile_s']:.2f}"
                  f" | coverage DMCI={d['dmci']['coverage']:.2f} B1={d['b1']['coverage']:.2f} "
                  f"B2={d['b2']['coverage']:.2f} | eng_LOC DMCI={d['dmci']['cumulative_eng_loc']} "
                  f"B2={d['b2']['cumulative_eng_loc']}")
        out["cells_prep_wall"] = prep_wall
        print(f"  prep wall: {prep_wall}  peak rss={_maxrss_mb():.0f}MB")

        # matched recovery on a subsample (cost-matched accuracy check)
        sample = corpus[:recover_sample]
        for prog in sample:
            rec = {"pid": prog.pid, "kind": prog.kind, "recursive_fraction": frac, "arms": {}}
            for a in _ARMS:
                # skip B1 where it has no coverage
                if a == "b1" and prog.kind == "recursive":
                    rec["arms"][a] = {"covered": False}
                    continue
                mse, rel = recover(a, prog, interp=dmci["interp"],
                                   time_budget=recover_budget, seed=0)
                rec["arms"][a] = {"covered": True, "best_mse": mse, "param_rel_error": rel}
            out["recovery"].append(rec)

    res_path = output_dir / "expj_results.json"
    res_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved: {res_path}")
    _recovery_summary(out["recovery"])
    return out


def _recovery_summary(recovery):
    if not recovery:
        return
    agg = {}
    for r in recovery:
        for a, d in r["arms"].items():
            if d.get("covered"):
                agg.setdefault((r["kind"], a), []).append(d["param_rel_error"])
    print("\n=== matched recovery: mean param-rel-error by (kind, arm) ===")
    for (kind, a), v in sorted(agg.items()):
        print(f"  {kind:11s} {a:5s}: {sum(v)/len(v):.3e}  (n={len(v)})")


def main():
    ap = argparse.ArgumentParser(description="Exp J: program-space calibration")
    ap.add_argument("--Ns", type=int, nargs="+", default=[1, 100, 10000])
    ap.add_argument("--fractions", type=float, nargs="+", default=[0.0, 1.0])
    ap.add_argument("--recover-sample", type=int, default=20)
    ap.add_argument("--recover-budget", type=float, default=20.0)
    ap.add_argument("--output-dir", type=Path, default=Path("experiments/exp_j/results"))
    args = ap.parse_args()
    sys.setrecursionlimit(5000)
    run(args.Ns, args.fractions, args.recover_sample, args.recover_budget, args.output_dir)


if __name__ == "__main__":
    main()
