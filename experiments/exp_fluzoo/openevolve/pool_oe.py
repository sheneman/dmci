############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# pool_oe.py: Pool the OpenEvolve meta-island runs (n128 + the eight-partition array) into one result. Each OpenEvolve run...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Pool the OpenEvolve meta-island runs (n128 + the eight-partition array) into one result.

Each OpenEvolve run writes <output_dir>/best/best_program_info.json (+ best_program.py). This
collects every run, finds the global best by combined_score (= -val_rmse), reports the per-island
bests, and re-scores the global-best program on the TEST seasons for the final headline number.

    python3 -m experiments.exp_fluzoo.openevolve.pool_oe            # pool + report
    python3 -m experiments.exp_fluzoo.openevolve.pool_oe --no-test  # skip the test re-score
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.setrecursionlimit(20000)

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parents[0] / "results"


def _load_best(run_dir: Path) -> dict | None:
    info = run_dir / "best" / "best_program_info.json"
    prog = run_dir / "best" / "best_program.py"
    if not info.exists():
        return None
    try:
        d = json.loads(info.read_text())
    except Exception:  # noqa: BLE001
        return None
    metrics = d.get("metrics", d)  # OpenEvolve nests under "metrics"; fall back to top level
    return {
        "run": run_dir.name,
        "combined_score": metrics.get("combined_score"),
        "val_rmse": metrics.get("val_rmse"),
        "n_compartments": metrics.get("n_compartments"),
        "n_params": metrics.get("n_params"),
        "program_path": str(prog) if prog.exists() else None,
    }


def _model_source(program_path: str) -> str:
    import importlib.util
    spec = importlib.util.spec_from_file_location("flu_best", program_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.FLU_MODEL


def pool(results_dir: Path = RESULTS, do_test: bool = True) -> dict:
    run_dirs = sorted([p for p in results_dir.glob("oe*") if p.is_dir()
                       and (p / "best" / "best_program_info.json").exists()])
    bests = [b for b in (_load_best(d) for d in run_dirs) if b and b["combined_score"] is not None]
    if not bests:
        raise SystemExit(f"no OpenEvolve run outputs with a best program found under {results_dir}")
    bests.sort(key=lambda b: b["combined_score"], reverse=True)   # higher combined_score = better
    best = bests[0]

    print(f"[pool] {len(bests)} OpenEvolve meta-islands:")
    for b in bests:
        print(f"   {b['run']:20s} val_rmse={b['val_rmse']}  "
              f"compartments={b['n_compartments']} params={b['n_params']}")
    print(f"[pool] GLOBAL BEST: {best['run']} val_rmse={best['val_rmse']}")

    test_block = None
    if do_test and best["program_path"]:
        from experiments.exp_fluzoo.config import DEFAULT
        from experiments.exp_fluzoo.data.build_data import load_processed
        from experiments.exp_fluzoo.evolve import score_source
        src = _model_source(best["program_path"])
        data = load_processed()
        sc = score_source(src, "pool_best", data, DEFAULT, seeds=(0,),
                          refit_iters=5, origin_stride=8, test_seasons=DEFAULT.test_seasons)
        test_block = {"test_mean_rmse": sc["val_rmse"], "test": sc["skill"]}
        print(f"[pool] global-best TEST mean_rmse={sc['val_rmse']:.5f}")

    summary = {"n_islands": len(bests), "global_best": {**best, **(test_block or {})},
               "per_island": bests}
    (results_dir / "oe_pool_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"[pool] wrote {results_dir}/oe_pool_summary.json")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-test", action="store_true", help="skip the DMCI test re-score")
    ap.add_argument("--results-dir", default=str(RESULTS))
    args = ap.parse_args()
    pool(Path(args.results_dir), do_test=not args.no_test)


if __name__ == "__main__":
    main()
