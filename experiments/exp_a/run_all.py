#!/usr/bin/env python3
############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# run_all.py: Top-level experiment driver for Experiment A. Usage: # Run everything serially python -m...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Top-level experiment driver for Experiment A.

Usage:
    # Run everything serially
    python -m experiments.exp_a.run_all

    # Run a single SLURM array task
    python -m experiments.exp_a.run_all --slurm-task-id $SLURM_ARRAY_TASK_ID

    # Run specific methods/programs
    python -m experiments.exp_a.run_all --methods direct,compiled_interp --programs P1_single_const

    # Run ablations only
    python -m experiments.exp_a.run_all --ablations-only

    # Skip completed runs
    python -m experiments.exp_a.run_all --skip-existing
"""

from __future__ import annotations

import argparse
import sys
import time
from itertools import product
from pathlib import Path

from .config import ExpAConfig, DEFAULT
from .programs import ALL_PROGRAMS, PROGRAMS_BY_NAME
from .runner import run_single, is_complete
from .ablations import run_all_ablations


def build_run_matrix(cfg: ExpAConfig, methods=None, programs=None):
    if methods is None:
        methods = list(cfg.methods)
    if programs is None:
        programs = [p.name for p in ALL_PROGRAMS]
    seeds = list(cfg.seeds)
    return list(product(methods, programs, seeds))


def task_id_to_config(task_id: int, matrix: list) -> tuple[str, str, int]:
    return matrix[task_id]


def main():
    parser = argparse.ArgumentParser(description="Experiment A: full run")
    parser.add_argument("--output-dir", type=str, default=DEFAULT.output_dir)
    parser.add_argument("--methods", type=str, default=None,
                        help="Comma-separated method names")
    parser.add_argument("--programs", type=str, default=None,
                        help="Comma-separated program names")
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated seeds or range (e.g., 0-4)")
    parser.add_argument("--slurm-task-id", type=int, default=None,
                        help="Run a single task by SLURM array index")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip runs whose output files already exist")
    parser.add_argument("--ablations-only", action="store_true")
    parser.add_argument("--no-ablations", action="store_true")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    # Build config
    overrides = {"output_dir": args.output_dir}
    if args.max_epochs is not None:
        overrides["max_epochs"] = args.max_epochs
    if args.lr is not None:
        overrides["lr"] = args.lr
    if args.seeds is not None:
        if "-" in args.seeds:
            lo, hi = args.seeds.split("-")
            overrides["seeds"] = tuple(range(int(lo), int(hi) + 1))
            overrides["n_seeds"] = int(hi) - int(lo) + 1
        else:
            seeds = tuple(int(s) for s in args.seeds.split(","))
            overrides["seeds"] = seeds
            overrides["n_seeds"] = len(seeds)

    cfg = ExpAConfig(**{**DEFAULT.__dict__, **overrides})

    methods = args.methods.split(",") if args.methods else None
    programs = args.programs.split(",") if args.programs else None

    if args.ablations_only:
        run_all_ablations(cfg)
        return

    matrix = build_run_matrix(cfg, methods, programs)

    if args.slurm_task_id is not None:
        if args.slurm_task_id >= len(matrix):
            print(f"Task ID {args.slurm_task_id} out of range "
                  f"(matrix size={len(matrix)})", file=sys.stderr)
            sys.exit(1)
        method, prog, seed = matrix[args.slurm_task_id]
        print(f"[SLURM task {args.slurm_task_id}] "
              f"{method} / {prog} / seed={seed}", file=sys.stderr)
        r = run_single(method, prog, seed, cfg)
        status = "CONV" if r.converged else "----"
        print(f"  {status} loss={r.final_loss:.6f} time={r.total_wall_time:.1f}s",
              file=sys.stderr)
        return

    # Serial execution
    total = len(matrix)
    print(f"Experiment A: {total} runs "
          f"({len(cfg.methods)} methods x {len(ALL_PROGRAMS)} programs "
          f"x {cfg.n_seeds} seeds)", file=sys.stderr)

    t_start = time.perf_counter()
    completed = 0
    skipped = 0

    for i, (method, prog, seed) in enumerate(matrix):
        if args.skip_existing and is_complete(cfg.output_dir, method, prog, seed):
            skipped += 1
            continue

        t0 = time.perf_counter()
        r = run_single(method, prog, seed, cfg)
        dt = time.perf_counter() - t0
        completed += 1
        status = "CONV" if r.converged else "----"
        print(f"[{completed + skipped}/{total}] {method:20s} {prog:20s} "
              f"seed={seed} {status} loss={r.final_loss:.6f} "
              f"({dt:.1f}s)", file=sys.stderr)

    if not args.no_ablations:
        print("\nRunning ablations...", file=sys.stderr)
        run_all_ablations(cfg)

    elapsed = time.perf_counter() - t_start
    print(f"\nDone: {completed} completed, {skipped} skipped "
          f"in {elapsed:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
