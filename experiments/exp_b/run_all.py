############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# run_all.py: Top-level driver for Experiment B. Usage: python3 -m experiments.exp_b.run_all --output-dir...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Top-level driver for Experiment B.

Usage:
    python3 -m experiments.exp_b.run_all --output-dir experiments/exp_b/results
    python3 -m experiments.exp_b.run_all --output-dir experiments/exp_b/results --skip-existing
    python3 -m experiments.exp_b.run_all --generate-only --api anthropic
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# DMCI interprets recursive programs by recursing in the (compiled) evaluator,
# which nests many Python frames per interpreted call; deep recursion (e.g. M09's
# factorial-inside-loop) exceeds Python's default 1000 limit during graph build.
sys.setrecursionlimit(20000)

from .config import DEFAULT, ExpBConfig
from .models import ALL_MODELS, EQUATION_MODELS, PROGRAM_MODELS
from .runner import run_single


def main():
    parser = argparse.ArgumentParser(description="Run Experiment B")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("experiments/exp_b/results"))
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--methods", nargs="+", default=DEFAULT.methods)
    parser.add_argument("--models", nargs="+", default=None,
                        help="Model names to run (default: all)")
    parser.add_argument("--tiers", nargs="+", default=None,
                        choices=["equation", "program"],
                        help="Run only models of specified tiers")
    parser.add_argument("--seeds", type=int, default=DEFAULT.n_seeds)
    parser.add_argument("--max-epochs", type=int, default=DEFAULT.max_epochs)
    parser.add_argument("--lr", type=float, default=DEFAULT.lr)

    parser.add_argument("--generate-only", action="store_true",
                        help="Only generate LLM programs, don't train")
    parser.add_argument("--api", type=str, default="mindrouter",
                        choices=["anthropic", "openai", "mindrouter"])
    parser.add_argument("--use-llm-cache", action="store_true",
                        help="Train the actual LLM-generated programs from llm_cache/ "
                             "(uniform evaluator wrapper, no per-model edits) instead of "
                             "the hand-authored interp_source/direct_source in models.py")

    args = parser.parse_args()

    if args.generate_only:
        from .llm_generate import generate_all
        print("Generating Scheme programs via LLM...")
        results = generate_all(api=args.api, force=True)
        n_compile = sum(1 for r in results if r.compiles)
        n_correct = sum(1 for r in results if r.correct)
        print(f"\nResults: {n_compile}/{len(results)} compile, "
              f"{n_correct}/{len(results)} correct")
        return

    cfg = ExpBConfig(
        max_epochs=args.max_epochs,
        lr=args.lr,
        n_seeds=args.seeds,
    )

    if args.models:
        models = [m for m in ALL_MODELS if m.name in args.models]
    elif args.tiers:
        models = [m for m in ALL_MODELS if m.tier in args.tiers]
    else:
        models = ALL_MODELS

    if args.use_llm_cache:
        from .llm_sources import apply_llm_sources
        overridden = apply_llm_sources(models)
        print(f"LLM-CACHE MODE: training the actual LLM-generated programs for "
              f"{len(overridden)} models (uniform evaluator wrapper, no per-model edits): "
              f"{overridden}")
        print()

    runs = []
    for method in args.methods:
        for model in models:
            for seed in range(cfg.n_seeds):
                runs.append((method, model, seed))

    total = len(runs)
    print(f"Experiment B: {total} runs "
          f"({len(args.methods)} methods x {len(models)} models x {cfg.n_seeds} seeds)")
    print(f"Models: {[m.name for m in models]}")
    print(f"Methods: {args.methods}")
    print()

    for i, (method, model, seed) in enumerate(runs):
        tag = f"{method}_{model.name}_{seed:02d}"
        json_path = args.output_dir / f"{tag}.json"

        if args.skip_existing and json_path.exists():
            print(f"[{i+1}/{total}] {tag} — skipping (exists)", file=sys.stderr)
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
                  f"seed={seed} ERROR: {e} ({elapsed:.1f}s)", file=sys.stderr)


if __name__ == "__main__":
    main()
