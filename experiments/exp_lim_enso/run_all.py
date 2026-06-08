############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# run_all.py: Driver: sweep method x D x structure x seed, one ``runner.run_single`` per cell. This is the full LIM-ENSO...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Driver: sweep method x D x structure x seed, one ``runner.run_single`` per cell.

This is the full LIM-ENSO experiment loop (the gate is the go/no-go that precedes it). It:
  * loads the on-disk ``data/processed/`` PC series + spatial basis ONCE,
  * builds the per-CELL task list (method, D, structure, seed) from the (filtered) config,
  * fits + scores each cell with ``runner.run_single`` (writes ``results/<tag>.json/.csv``),
  * runs the cells with a PROCESS POOL (16-way by default): the DMCI interpreter is single-
    threaded Python (GIL-bound), so a single fit cannot use multiple cores -- the win is
    PROCESS-LEVEL parallelism over INDEPENDENT cells (each optimizes its own parameter vector,
    seeded ONLY from its 'seed', and writes its own JSON/CSV: no gradient sharing, no order-
    dependent state). Parallel results are per-seed-IDENTICAL to the serial loop (THE guardrail).
  * catches per-cell exceptions, logs ``(tag, status)``, and CONTINUES (one bad cell never
    aborts the sweep),
  * selects a portfolio winner per (structure, D) on held-out VALIDATION forecast skill
    (Nino-3.4 ACC at the headline lead) -- NEVER train NLL (the exp_i overfit lesson:
    "converged != recovered"; the winner is the model that FORECASTS best, not the one that
    drives the training likelihood lowest),
  * then aggregates (``aggregate.main`` over ``results/*.json``).

No thread oversubscription: each worker calls ``torch.set_num_threads(1)`` at startup and the
slurm script exports OMP/MKL/OPENBLAS_NUM_THREADS=1 BEFORE python starts (16 workers x 1 thread
= 16 cores, NOT 16 x 64). torch + multiprocessing uses the 'spawn' start method (fork after
importing torch can deadlock); the worker is a TOP-LEVEL picklable function fed a plain dict
(D/method/structure/seed + the processed-dir PATH, never a closure or a live tensor).

``sys.setrecursionlimit(DEFAULT.recursion_limit)`` is raised BEFORE importing neural_compiler
in BOTH the parent (below) and every worker (first thing in ``_run_cell``). DMCI is
interpreter-bound -> run on CPU (the 'sheneman'/n128 node with no GPU requested).

    python3 -u -m experiments.exp_lim_enso.run_all --workers 16 \
        --output-dir experiments/exp_lim_enso/results
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# --- raise recursion limit BEFORE importing neural_compiler (via runner/baselines/models) ---
from .config import DEFAULT, ExpLimConfig  # config has no heavy import

sys.setrecursionlimit(DEFAULT.recursion_limit)

import numpy as np  # noqa: E402

from . import runner  # noqa: E402  (transitively imports neural_compiler)


HERE = Path(__file__).parent
PROCESSED_DIR = HERE / "data" / "processed"

# Default method portfolio. dmci_adam is the PRIMARY (exact DMCI gradients); the others are
# the honest comparison baselines over the SAME shared objective.
DEFAULT_METHODS = ["dmci_adam", "lbfgs_multistart", "diffevo_batched"]

# Validation lead (months) the portfolio winner is selected on (held-out, NEVER train NLL).
HEADLINE_LEAD = 6

# Default number of worker processes (parameterized via --workers).
DEFAULT_WORKERS = 16


# ===========================================================================
# Data loading.
# ===========================================================================

def _required_npy():
    """The processed arrays each cell needs (file stems under ``data/processed/``)."""
    return ["pcs.npy", "eofs.npy", "pc_std.npy", "lat.npy", "lon.npy"]


def _check_processed(processed_dir: Path) -> None:
    """Fail fast (in the PARENT) if the processed arrays are missing."""
    missing = [n for n in _required_npy() if not (processed_dir / n).exists()]
    if missing:
        raise FileNotFoundError(
            f"run_all: missing {missing} in {processed_dir}/. Build/rsync the data first "
            f"(python3 -m experiments.exp_lim_enso.data.build_data; then rsync processed/).")


def load_processed(processed_dir: Path = PROCESSED_DIR):
    """Load pcs / eofs / pc_std / lat / lon from ``data/processed/`` (np.load).

    Returns ``(pcs_torch, eofs, pc_std, lat, lon)``: ``pcs`` is the [T_full, D_max] PC tensor
    (THE input; the solver binds ``as_matrix(pcs[:T_train, :D])`` inside the DMCI program and
    forecast slices its own windows); the rest are numpy spatial-reconstruction arrays. Called
    PER WORKER from the processed-dir path in the payload (the arrays are tiny -- pcs is
    [898,20] -- so per-process re-loading is free and keeps the workers closure-free)."""
    import torch
    p = Path(processed_dir)
    _check_processed(p)
    pcs = np.load(p / "pcs.npy").astype(np.float32)
    eofs = np.load(p / "eofs.npy")
    pc_std = np.load(p / "pc_std.npy")
    lat = np.load(p / "lat.npy")
    lon = np.load(p / "lon.npy")
    return torch.tensor(pcs), eofs, pc_std, lat, lon


# ===========================================================================
# Worker: ONE independent cell (top-level, picklable; runs in a spawned process).
# ===========================================================================

def _run_cell(payload: dict) -> dict:
    """Fit + score ONE (method, D, structure, seed) cell in this worker process.

    The payload is a plain picklable dict (no closures, no live tensors): ``method``,
    ``D``, ``structure``, ``seed``, ``processed_dir`` (path), ``headline_lead``, ``log_every``,
    and a serialized ``cfg`` (see ``main``: the cfg is shipped via ``dataclasses.asdict`` and
    rebuilt here so per-process state is fully determined by the payload).

    Discipline (in THIS order):
      1. ``torch.set_num_threads(1)`` + ``sys.setrecursionlimit(cfg.recursion_limit)`` BEFORE
         importing/using neural_compiler (no thread oversubscription; deep DMCI walk safe).
      2. ``np.load`` pcs/eofs/pc_std/lat/lon from the processed dir.
      3. slice obs = pcs[:, :D] is done INSIDE the solver (it slices [:T, :D]); we pass full pcs.
      4. ``runner.run_single`` (writes ``results/<tag>.json/.csv`` itself).

    Returns a SMALL summary dict ``{tag, status, wall_time, final_nll, heldout_acc}``. A failing
    cell returns ``status='error:<Type>'`` (never raises) so the pool keeps going.
    """
    method = payload["method"]
    D = int(payload["D"])
    structure = payload["structure"]
    seed = int(payload["seed"])
    headline_lead = int(payload["headline_lead"])
    log_every = payload.get("log_every")
    output_dir = Path(payload["output_dir"])

    # --- rebuild the config in-process (fully determined by the payload) ---
    cfg = _cfg_from_dict(payload["cfg"])

    # --- (1) thread + recursion discipline BEFORE touching neural_compiler ---
    sys.setrecursionlimit(int(cfg.recursion_limit))
    import torch
    torch.set_num_threads(1)

    # imported here (in the worker) -- transitively pulls in neural_compiler AFTER the
    # recursion limit is raised. The module-level cache (_GRAPH_CACHE) is per-process.
    from . import runner as _runner

    tag = _runner.tag_for(method, structure, D, seed)
    t0 = time.perf_counter()
    try:
        pcs, eofs, pc_std, lat, lon = load_processed(payload["processed_dir"])
        rec = _runner.run_single(method, D, structure, seed, cfg,
                                 output_dir, pcs, eofs, pc_std, lat, lon,
                                 log_every=log_every)
        wall = time.perf_counter() - t0
        acc = rec.get("heldout_acc") or {}
        hl = acc.get(headline_lead, acc.get(str(headline_lead)))
        return {
            "tag": tag,
            "status": "ok",
            "wall_time": wall,
            "final_nll": float(rec.get("final_nll")) if rec.get("final_nll") is not None
            else None,
            "heldout_acc": _coerce_float(hl),
        }
    except Exception as exc:  # noqa: BLE001  (a failing cell never kills the pool)
        wall = time.perf_counter() - t0
        # print the traceback in the worker's stdout so the slurm .out has the full context.
        print(f"[ERROR {tag}] {type(exc).__name__}: {exc}\n{traceback.format_exc()}",
              file=sys.stderr, flush=True)
        return {
            "tag": tag,
            "status": f"error:{type(exc).__name__}",
            "wall_time": wall,
            "final_nll": None,
            "heldout_acc": None,
        }


def _coerce_float(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


# ===========================================================================
# Config (de)serialization for picklable payloads.
# ===========================================================================

def _cfg_to_dict(cfg: ExpLimConfig) -> dict:
    """Serialize an ExpLimConfig to a plain JSON/pickle-safe dict (for the worker payload)."""
    import dataclasses
    return dataclasses.asdict(cfg)


def _cfg_from_dict(d: dict) -> ExpLimConfig:
    """Rebuild an ExpLimConfig from a plain dict (in the worker)."""
    return ExpLimConfig(**d)


# ===========================================================================
# Portfolio winner: best VALIDATION forecast skill (held-out), never train NLL.
# ===========================================================================

def _val_skill(record: dict, lead: int) -> float:
    """Held-out validation score for the portfolio: fitted-LIM Nino-3.4 ACC at ``lead`` mo.

    Higher is better. Returns -inf when the held-out skill is missing/non-finite so a failed
    or skill-less run can never win the portfolio. This is deliberately the HELD-OUT forecast
    metric, NOT ``final_nll`` -- a model that overfits the train likelihood but forecasts
    poorly must lose (the exp_i overfit lesson)."""
    acc = record.get("heldout_acc") or {}
    v = acc.get(lead, acc.get(str(lead)))
    try:
        v = float(v)
    except (TypeError, ValueError):
        return float("-inf")
    return v if np.isfinite(v) else float("-inf")


# ===========================================================================
# Cell list + skip-existing.
# ===========================================================================

def _build_cells(methods, D_list, structures, seeds, output_dir: Path,
                 skip_existing: bool) -> tuple[list[dict], list[dict]]:
    """Build the per-CELL task list (one seed each) and the skip log.

    Ordering: (structure, D, method, seed) -- the SAME order the serial loop used, so the
    completion order is irrelevant to results but the submission order matches for readability.
    PER-CELL (one seed each), NOT per-(structure, D) chunks, so the seeds of a given D run on
    different cores concurrently (wall time near the single longest fit, not 3x it).

    Returns ``(cells, skip_log)`` where ``cells`` is the list of (method,D,structure,seed)
    coordinate dicts to submit and ``skip_log`` records the already-existing cells skipped.
    """
    cells: list[dict] = []
    skip_log: list[dict] = []
    for structure in structures:
        for D in D_list:
            for method in methods:
                for seed in seeds:
                    tag = runner.tag_for(method, structure, D, seed)
                    jpath = output_dir / f"{tag}.json"
                    if skip_existing and jpath.exists():
                        print(f"[skip] {tag} (exists)", flush=True)
                        skip_log.append({"tag": tag, "status": "skipped", "wall_time": 0.0})
                        continue
                    cells.append({"method": method, "D": int(D),
                                  "structure": structure, "seed": int(seed), "tag": tag})
    return cells, skip_log


def _load_existing_records(output_dir: Path) -> list[dict]:
    """Load every ``results/*.json`` per-run record (skip the summary) for the portfolio.

    The portfolio is computed from the on-disk records (which include the SKIPPED cells'
    artifacts), so a resumed run still selects over the full sweep."""
    recs: list[dict] = []
    for f in sorted(output_dir.glob("*.json")):
        if f.name == "run_all_summary.json":
            continue
        try:
            r = json.loads(f.read_text())
        except Exception:  # noqa: BLE001
            continue
        if isinstance(r, dict) and "method" in r and "D" in r:
            recs.append(r)
    return recs


# ===========================================================================
# Main sweep (process pool).
# ===========================================================================

def run_all(methods, D_list, structures, seeds, output_dir: Path,
            cfg: ExpLimConfig, *, skip_existing: bool, headline_lead: int,
            workers: int = DEFAULT_WORKERS) -> dict:
    """Sweep method x D x structure x seed with a PROCESS POOL; write per-cell + portfolio."""
    output_dir.mkdir(parents=True, exist_ok=True)
    _check_processed(PROCESSED_DIR)   # fail fast in the parent before spawning workers
    print(f"[run_all] processed dir: {PROCESSED_DIR}  "
          f"T_train={cfg.T_train} T_test={cfg.T_test}  workers={workers}", flush=True)

    cells, skip_log = _build_cells(methods, D_list, structures, seeds, output_dir,
                                   skip_existing)
    run_log: list[dict] = list(skip_log)
    N = len(cells)
    print(f"[run_all] {N} cells to run ({len(skip_log)} skipped existing)", flush=True)

    if N:
        cfg_d = _cfg_to_dict(cfg)
        payloads = [{
            "method": c["method"], "D": c["D"], "structure": c["structure"],
            "seed": c["seed"], "tag": c["tag"],
            "processed_dir": str(PROCESSED_DIR),
            "output_dir": str(output_dir),
            "headline_lead": headline_lead,
            "log_every": cfg.log_every,
            "cfg": cfg_d,
        } for c in cells]

        # 'spawn': fork-after-importing-torch can deadlock; spawn is the safe start method.
        ctx = multiprocessing.get_context("spawn")
        done = 0
        with ProcessPoolExecutor(max_workers=max(1, int(workers)), mp_context=ctx) as ex:
            futures = {ex.submit(_run_cell, p): p["tag"] for p in payloads}
            for fut in as_completed(futures):
                tag = futures[fut]
                done += 1
                try:
                    summ = fut.result()
                except Exception as exc:  # noqa: BLE001  (defensive: pool/pickle failure)
                    summ = {"tag": tag, "status": f"error:{type(exc).__name__}",
                            "wall_time": 0.0, "final_nll": None, "heldout_acc": None}
                    print(f"[ERROR] {tag}: {type(exc).__name__}: {exc}",
                          file=sys.stderr, flush=True)
                run_log.append({k: summ.get(k) for k in ("tag", "status", "wall_time")})
                nll = summ.get("final_nll")
                acc = summ.get("heldout_acc")
                nll_s = f"{nll:.4g}" if isinstance(nll, float) else "-"
                acc_s = f"{acc:.3f}" if isinstance(acc, float) else "-"
                wt = summ.get("wall_time") or 0.0
                tagm = "[done]" if summ.get("status") == "ok" else "[FAIL]"
                print(f"{tagm} {summ['tag']} nll={nll_s} acc@{headline_lead}={acc_s} "
                      f"t={wt:.1f}s  ({done}/{N})", flush=True)

    # --- portfolio over ALL on-disk records (includes skipped cells' artifacts) ---
    records = _load_existing_records(output_dir)
    portfolio = _select_portfolio(records, structures, D_list, headline_lead)
    summary = {
        "headline_lead": headline_lead,
        "selection_metric": "heldout_nino34_ACC_fitted_lim",
        "methods": methods, "D_list": D_list, "structures": structures, "seeds": seeds,
        "workers": int(workers),
        "n_cells": N,
        "n_records": len(records),
        "run_log": run_log,
        "portfolio_winner_by_structure_D": portfolio,
    }
    (output_dir / "run_all_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    _print_summary(run_log, portfolio, headline_lead)
    return summary


def _select_portfolio(records, structures, D_list, lead: int) -> dict:
    """Per (structure, D), the method with the best HELD-OUT validation forecast skill.

    Selection is on validation Nino-3.4 ACC at the headline lead (averaged over seeds), NEVER
    on train NLL. Returns ``{ "<structure>_D<D>": {winner, val_acc, by_method} }``."""
    out: dict = {}
    for structure in structures:
        for D in D_list:
            cell = [r for r in records
                    if r.get("structure") == structure and r.get("D") == D]
            if not cell:
                continue
            by_method: dict[str, list[float]] = {}
            for r in cell:
                by_method.setdefault(r["method"], []).append(_val_skill(r, lead))
            mean_acc = {m: float(np.mean(v)) for m, v in by_method.items() if v}
            if not mean_acc:
                continue
            winner = max(mean_acc, key=mean_acc.get)
            out[f"{structure}_D{D}"] = {
                "winner": winner,
                "val_acc": mean_acc[winner],
                "by_method": mean_acc,
            }
    return out


def _print_summary(run_log, portfolio, lead: int) -> None:
    ok = sum(1 for r in run_log if r["status"] == "ok")
    err = sum(1 for r in run_log if str(r["status"]).startswith("error"))
    skip = sum(1 for r in run_log if r["status"] == "skipped")
    print(f"\n=== run_all done: {ok} ok, {err} errors, {skip} skipped ===", flush=True)
    print(f"=== portfolio winner by (structure, D) on held-out Nino-3.4 ACC @ {lead}mo ===")
    for key in sorted(portfolio):
        w = portfolio[key]
        cells = "  ".join(f"{m}={a:.3f}" for m, a in sorted(w["by_method"].items()))
        print(f"  {key:>12s}: WINNER {w['winner']:>18s} (acc={w['val_acc']:.3f})   [{cells}]")


# ===========================================================================
# CLI.
# ===========================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description="LIM-ENSO full sweep (method x D x structure x seed)")
    ap.add_argument("--output-dir", type=Path,
                    default=HERE / "results",
                    help="where to write results/<tag>.json/.csv + run_all_summary.json")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip a cell whose results/<tag>.json already exists (resume)")
    ap.add_argument("--methods", nargs="+", default=DEFAULT_METHODS,
                    choices=DEFAULT_METHODS,
                    help="solver portfolio (dmci_adam is the PRIMARY exact-gradient solver)")
    ap.add_argument("--structures", nargs="+", default=DEFAULT.structures,
                    help="F-assembly variants (Phase 1: S0 dense)")
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT.seeds)
    ap.add_argument("--D", type=int, nargs="+", default=DEFAULT.D_list,
                    help="state dimensions (PC truncations) to sweep")
    ap.add_argument("--headline-D", type=int, default=None,
                    help="restrict the sweep to a single headline dimension (e.g. 10)")
    ap.add_argument("--headline-lead", type=int, default=HEADLINE_LEAD,
                    help="forecast lead (months) the portfolio winner is selected on")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                    help="number of worker PROCESSES (one per core; 16 x 1 thread = 16 cores)")
    ap.add_argument("--no-aggregate", action="store_true",
                    help="skip the aggregate.main() pass at the end of the sweep")
    args = ap.parse_args()

    cfg = DEFAULT
    D_list = [args.headline_D] if args.headline_D is not None else list(args.D)

    print("============================================================", flush=True)
    print("LIM-ENSO run_all (parallel)", flush=True)
    print(f"  methods    = {args.methods}", flush=True)
    print(f"  D_list     = {D_list}", flush=True)
    print(f"  structures = {args.structures}", flush=True)
    print(f"  seeds      = {args.seeds}", flush=True)
    print(f"  workers    = {args.workers}", flush=True)
    print(f"  output-dir = {args.output_dir}", flush=True)
    print(f"  recursionlimit = {sys.getrecursionlimit()}", flush=True)
    print("============================================================", flush=True)

    run_all(args.methods, D_list, args.structures, args.seeds, args.output_dir, cfg,
            skip_existing=args.skip_existing, headline_lead=args.headline_lead,
            workers=args.workers)

    if not args.no_aggregate:
        try:
            from . import aggregate
            _aggregate_results(aggregate, args.output_dir, args.headline_lead)
        except Exception as exc:  # noqa: BLE001  (aggregation must never fail the sweep)
            print(f"[run_all] aggregate skipped ({type(exc).__name__}: {exc})",
                  file=sys.stderr, flush=True)


def _aggregate_results(aggregate, output_dir: Path, headline_lead: int) -> None:
    """Run the aggregator over ``results/*.json`` at the end of the sweep.

    Drives ``aggregate.main`` via its argv (it parses ``--results-dir`` / ``--headline-lead``),
    restoring argv afterward so a programmatic caller is unaffected."""
    saved = sys.argv
    try:
        sys.argv = ["aggregate", "--results-dir", str(output_dir),
                    "--headline-lead", str(headline_lead)]
        aggregate.main()
    finally:
        sys.argv = saved


if __name__ == "__main__":
    main()
