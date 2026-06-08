############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# run_all.py: Top-level driver for Experiment C. Usage: python3 -m experiments.exp_c.run_all --output-dir...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Top-level driver for Experiment C.

Usage:
    python3 -m experiments.exp_c.run_all --output-dir experiments/exp_c/results
    python3 -m experiments.exp_c.run_all --output-dir experiments/exp_c/results --skip-existing
    python3 -m experiments.exp_c.run_all --model C01_lotka_volterra --skip-existing
    python3 -m experiments.exp_c.run_all --categories coupled_ode iterative
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTHONUNBUFFERED", "1")
sys.setrecursionlimit(5000)

from .config import DEFAULT, ExpCConfig
from .models import ALL_MODELS


def main():
    parser = argparse.ArgumentParser(description="Run Experiment C")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("experiments/exp_c/results"))
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--methods", nargs="+", default=DEFAULT.methods)
    parser.add_argument("--model", type=str, default=None,
                        help="Single model name (for SLURM array jobs)")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Model names to run (default: all)")
    parser.add_argument("--categories", nargs="+", default=None,
                        choices=["coupled_ode", "iterative",
                                 "recursive_filter"],
                        help="Run only models of specified categories")
    parser.add_argument("--seeds", type=int, default=DEFAULT.n_seeds)
    parser.add_argument("--max-epochs", type=int, default=DEFAULT.max_epochs)
    parser.add_argument("--lr", type=float, default=DEFAULT.lr)

    args = parser.parse_args()

    cfg = ExpCConfig(
        max_epochs=args.max_epochs,
        lr=args.lr,
        n_seeds=args.seeds,
    )

    if args.model:
        models = [m for m in ALL_MODELS if m.name == args.model]
        if not models:
            print(f"ERROR: unknown model '{args.model}'", file=sys.stderr)
            sys.exit(1)
    elif args.models:
        models = [m for m in ALL_MODELS if m.name in args.models]
    elif args.categories:
        models = [m for m in ALL_MODELS if m.category in args.categories]
    else:
        models = ALL_MODELS

    runs = []
    for method in args.methods:
        for model in models:
            for seed in range(cfg.n_seeds):
                runs.append((method, model, seed))

    total = len(runs)
    print(f"Experiment C: {total} runs "
          f"({len(args.methods)} methods x {len(models)} models "
          f"x {cfg.n_seeds} seeds)")
    print(f"Models: {[m.name for m in models]}")
    print(f"Methods: {args.methods}")
    print()

    from .runner import run_single

    for i, (method, model, seed) in enumerate(runs):
        tag = f"{method}_{model.name}_{seed:02d}"
        json_path = args.output_dir / f"{tag}.json"

        if args.skip_existing and json_path.exists():
            print(f"[{i+1}/{total}] {tag} — skipping (exists)",
                  file=sys.stderr)
            continue

        t0 = time.perf_counter()
        try:
            result = run_single(method, model.name, seed, cfg,
                                output_dir=args.output_dir)
            elapsed = time.perf_counter() - t0
            status = "CONV" if result.converged else "NOCONV"
            print(f"[{i+1}/{total}] {method:20s} {model.name:30s} "
                  f"seed={seed} {status} loss={result.final_loss:.6f} "
                  f"({elapsed:.1f}s)", file=sys.stderr)
        except Exception as e:
            elapsed = time.perf_counter() - t0
            print(f"[{i+1}/{total}] {method:20s} {model.name:30s} "
                  f"seed={seed} ERROR: {e} ({elapsed:.1f}s)",
                  file=sys.stderr)


if __name__ == "__main__":
    main()
