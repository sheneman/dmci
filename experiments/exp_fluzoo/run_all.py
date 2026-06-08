############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# run_all.py: Sweep driver for the FluZoo co-search: calibrate every accepted program, select on held-out validation skill,...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Sweep driver for the FluZoo co-search: calibrate every accepted program, select on
held-out validation skill, report test skill, and emit the skill-vs-programs-searched curve.

    python3 -m experiments.exp_fluzoo.run_all [--programs cache|reference] [--workers N]
        [--train-seasons ...] [--val-seasons ...] [--test-seasons ...]
        [--adam-iters N] [--refit-iters N] [--origin-stride N] [--limit N]

Process-parallel (spawn) over independent programs; each worker pins threads and raises the
recursion limit before importing the interpreter. The selection metric is validation mean
RMSE (never training NLL). The headline figure -- best held-out skill vs. number of programs
searched (LLM order vs. a shuffled control) -- is the direct test that exploring discrete
program structure, not just more parameters, keeps improving the frontier.
"""

from __future__ import annotations

import sys

# Raise BEFORE neural_compiler is imported (here and in every spawned worker).
from .config import DEFAULT, FluZooConfig  # config imports no heavy deps
sys.setrecursionlimit(DEFAULT.recursion_limit)

import argparse
import dataclasses
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
CACHE_DIR = HERE / "llm_cache"
PROCESSED = HERE / "data" / "processed"


# --------------------------------------------------------------------------- #
# Program sources
# --------------------------------------------------------------------------- #

def load_zoo_programs(programs: str, limit: int | None,
                      shard: int = 0, nshards: int = 1) -> list[tuple[str, str]]:
    """Return [(name, source), ...] for the requested program set, in a stable order.

    With nshards>1 this task takes a disjoint stride of the programs (shard, shard+N, ...),
    so a slurm array can fan the sweep across many nodes of the `eight` partition; every shard
    writes per-program records into the SAME output dir and a final --merge pass aggregates them.
    """
    if programs == "reference":
        from .programs import REFERENCE_PROGRAMS
        # the sweep fits the 11-region observation contract, so use the regional references.
        items = [(n, s) for n, s in REFERENCE_PROGRAMS.items() if n.endswith("_regional")]
    else:  # cache: accepted LLM programs
        items = []
        for p in sorted(CACHE_DIR.glob("*.json")):
            rec = json.loads(p.read_text())
            if rec.get("status") == "accepted" and rec.get("source"):
                items.append((rec.get("recipe_id", p.stem), rec["source"]))
    if limit:
        items = items[:limit]
    if nshards > 1:
        items = items[shard::nshards]
    return items


# --------------------------------------------------------------------------- #
# Worker (top-level + picklable; reconstructs everything from a plain payload)
# --------------------------------------------------------------------------- #

def _cell(payload: dict) -> dict:
    import sys as _sys
    _sys.setrecursionlimit(payload["recursion_limit"])
    import torch
    torch.set_num_threads(1)
    from .data.build_data import load_processed
    from .runner import run_single
    cfg = payload["cfg"]
    data = load_processed(Path(payload["processed_dir"]))
    try:
        return run_single(payload["source"], payload["name"], data, cfg=cfg,
                          seeds=cfg.seeds, output_dir=payload["output_dir"],
                          refit_iters=payload["refit_iters"],
                          origin_stride=payload["origin_stride"])
    except Exception as exc:  # noqa: BLE001  (one bad program must not kill the sweep)
        return {"name": payload["name"], "error": f"{type(exc).__name__}: {exc}"}


# --------------------------------------------------------------------------- #
# Skill-vs-programs-searched curve (the headline figure)
# --------------------------------------------------------------------------- #

def skill_curve(records: list[dict]) -> list[dict]:
    """Best-so-far held-out skill as programs are searched in order (lower RMSE = better)."""
    curve, best_val, best_test = [], float("inf"), float("inf")
    for k, r in enumerate(records, start=1):
        v = r.get("val_mean_rmse", float("inf"))
        if v < best_val:
            best_val, best_test = v, r.get("test_mean_rmse", float("inf"))
        curve.append({"k": k, "best_val_rmse": best_val, "selected_test_rmse": best_test})
    return curve


def run_all(cfg: FluZooConfig, programs: str = "cache", workers: int = 8,
            limit: int | None = None, refit_iters: int = 25, origin_stride: int = 3,
            output_dir: Path = RESULTS, shard: int = 0, nshards: int = 1,
            do_finalize: bool = True) -> dict:
    items = load_zoo_programs(programs, limit, shard=shard, nshards=nshards)
    if not items:
        raise SystemExit("no programs to run (build the zoo with llm_generate, or use "
                         "--programs reference)")
    output_dir.mkdir(parents=True, exist_ok=True)
    payloads = [dict(name=name, source=src, cfg=cfg, processed_dir=str(PROCESSED),
                     output_dir=str(output_dir), recursion_limit=cfg.recursion_limit,
                     refit_iters=refit_iters, origin_stride=origin_stride)
                for name, src in items]

    records: list[dict] = []
    if workers <= 1:
        for pl in payloads:
            records.append(_cell(pl))
            print(f"  [{len(records):4d}/{len(payloads)}] {records[-1].get('name')}: "
                  f"val={records[-1].get('val_mean_rmse')} test={records[-1].get('test_mean_rmse')}")
    else:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            for i, rec in enumerate(ex.map(_cell, payloads), start=1):
                records.append(rec)
                print(f"  [{i:4d}/{len(payloads)}] {rec.get('name')}: "
                      f"val={rec.get('val_mean_rmse')} test={rec.get('test_mean_rmse')}")

    if not do_finalize:  # shard mode: per-program JSONs are written; merge happens separately
        print(f"[shard {shard}/{nshards}] scored {len(records)} programs -> {output_dir}")
        return {"shard": shard, "n_run": len(records)}
    return finalize(records, cfg, origin_stride, output_dir, programs, nshards)


def _records_from_dir(output_dir: Path) -> list[dict]:
    """Reload per-program records written by the shards (for the merge pass)."""
    recs = []
    for p in sorted(Path(output_dir).glob("*.json")):
        if p.name in ("run_all_summary.json",):
            continue
        try:
            r = json.loads(p.read_text())
            if isinstance(r, dict) and "name" in r and "val_mean_rmse" in r:
                recs.append(r)
        except Exception:  # noqa: BLE001
            pass
    return recs


def finalize(records: list[dict], cfg: FluZooConfig, origin_stride: int, output_dir: Path,
             programs: str = "cache", nshards: int = 1) -> dict:
    ok = [r for r in records if "error" not in r and r.get("val_mean_rmse") is not None]
    ok_sorted = sorted(ok, key=lambda r: r["val_mean_rmse"])
    selected = ok_sorted[0] if ok_sorted else None

    # baselines on the same val/test seasons
    from .data.build_data import load_processed
    from .baselines import score_baselines
    data = load_processed(PROCESSED)
    baselines = {
        "val": score_baselines(data, cfg=cfg, test_seasons=cfg.val_seasons, stride=origin_stride),
        "test": score_baselines(data, cfg=cfg, test_seasons=cfg.test_seasons, stride=origin_stride),
    }

    summary = {
        "n_run": len(records),
        "n_ok": len(ok),
        "n_error": len(records) - len(ok),
        "selected": None if selected is None else {
            "name": selected["name"],
            "val_mean_rmse": selected["val_mean_rmse"],
            "test_mean_rmse": selected["test_mean_rmse"],
            "n_params": selected["n_params"],
            "test_by_horizon": {h: selected["test"][h] for h in selected["test"]
                                if h.startswith("h")},
        },
        "skill_curve": skill_curve(records),
        "baselines": {k: {b: v["mean_rmse"] for b, v in baselines[k].items()}
                      for k in ("val", "test")},
        "config": {"programs": programs, "train_seasons": list(cfg.train_seasons),
                   "val_seasons": list(cfg.val_seasons), "test_seasons": list(cfg.test_seasons),
                   "adam_iters": cfg.adam_iters, "refit_iters": refit_iters,
                   "origin_stride": origin_stride},
    }
    (output_dir / "run_all_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print("\n[summary]", json.dumps({k: summary[k] for k in ("n_run", "n_ok", "n_error")}))
    if selected:
        print(f"[selected] {selected['name']}  val_rmse={selected['val_mean_rmse']:.4f}  "
              f"test_rmse={selected['test_mean_rmse']:.4f}  (k={selected['n_params']})")
        print(f"[baselines test mean_rmse] "
              + ", ".join(f"{b}={v:.4f}" for b, v in summary['baselines']['test'].items()))

    try:
        from .aggregate import aggregate
        aggregate(output_dir=output_dir)
    except Exception as exc:  # noqa: BLE001
        print(f"[aggregate] skipped: {type(exc).__name__}: {exc}")
    return summary


def merge(cfg: FluZooConfig, output_dir: Path, origin_stride: int) -> dict:
    """Aggregate all shards' per-program records (the final step of a multi-node sweep)."""
    records = _records_from_dir(output_dir)
    if not records:
        raise SystemExit(f"no per-program records found in {output_dir}")
    print(f"[merge] finalizing {len(records)} program records from {output_dir}")
    return finalize(records, cfg, origin_stride, Path(output_dir), "cache")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--programs", choices=["cache", "reference"], default="cache")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--adam-iters", type=int, default=None)
    ap.add_argument("--refit-iters", type=int, default=25)
    ap.add_argument("--origin-stride", type=int, default=3)
    ap.add_argument("--train-seasons", type=int, nargs="*", default=None)
    ap.add_argument("--val-seasons", type=int, nargs="*", default=None)
    ap.add_argument("--test-seasons", type=int, nargs="*", default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--output-dir", default=str(RESULTS))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1,
                    help="multi-node: this task scores the (shard, shard+N, ...) stride")
    ap.add_argument("--merge", action="store_true",
                    help="aggregate all shards' per-program records into the final summary")
    args = ap.parse_args()

    overrides = {}
    if args.adam_iters is not None:
        overrides["adam_iters"] = args.adam_iters
    if args.seeds is not None:
        overrides["seeds"] = tuple(args.seeds)
    if args.train_seasons is not None:
        overrides["train_seasons"] = tuple(args.train_seasons)
    if args.val_seasons is not None:
        overrides["val_seasons"] = tuple(args.val_seasons)
    if args.test_seasons is not None:
        overrides["test_seasons"] = tuple(args.test_seasons)
    cfg = dataclasses.replace(DEFAULT, **overrides) if overrides else DEFAULT
    output_dir = Path(args.output_dir)

    if args.merge:
        merge(cfg, output_dir, args.origin_stride)
        return
    run_all(cfg, programs=args.programs, workers=args.workers, limit=args.limit,
            refit_iters=args.refit_iters, origin_stride=args.origin_stride,
            output_dir=output_dir, shard=args.shard, nshards=args.nshards,
            do_finalize=(args.nshards == 1))


if __name__ == "__main__":
    main()
