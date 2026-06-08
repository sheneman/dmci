############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# evolve.py: LLM-guided evolutionary search over influenza-model program structure. This is the co-search made closed-loop....
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""LLM-guided evolutionary search over influenza-model program structure.

This is the co-search made closed-loop. Instead of sampling programs i.i.d. and selecting
the best at the end (random search), the LLM EVOLVES structure on feedback: held-out
validation forecast skill (computed by calibrating each program through the frozen DMCI
interpreter) is the fitness, and the language models act as MUTATION and CROSSOVER operators
over programs.

  generation 0 : recipe-seeded proposals -> screen -> calibrate + score val skill
  generation t : take the elite (best val skill) + some novelty, show each parent program
                 and its score back to an LLM, and ask it to mutate / combine them into a
                 better-forecasting variant -> screen -> calibrate + score
                 (elitism keeps winners; a canonical-structure check keeps novelty)

Operators are round-robined over the model zoo (qwen3.6-35b, qwen3.6-27b, GPT-5.5). The
output is the archive, the best program (scored on TEST at the end), and the
best-skill-vs-evaluations curve -- which, against a random-search control, is the direct
evidence that the LLM is EXPLOITING program structure, not merely sampling it.

Run on HPC (n128): generation hits the LLM endpoints, scoring is DMCI-bound on CPU.
"""

from __future__ import annotations

import sys

from .config import DEFAULT, FluZooConfig  # config pulls in no heavy deps
sys.setrecursionlimit(DEFAULT.recursion_limit)

import argparse
import dataclasses
import json
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PROCESSED = HERE / "data" / "processed"


# --------------------------------------------------------------------------- #
# Operator prompts (the LLM as a structure mutation / crossover operator)
# --------------------------------------------------------------------------- #

def mutate_prompt(parent_source: str, val_rmse: float) -> str:
    return (
        "Here is a WORKING influenza forecasting model. On held-out weekly %ILI it achieves a "
        f"validation RMSE of {val_rmse:.5f} (lower is better):\n```scheme\n{parent_source}\n```\n\n"
        "Propose ONE structurally DIFFERENT model that is likely to forecast BETTER. Make a "
        "meaningful epidemiological change -- e.g. add/remove a compartment (exposed, "
        "asymptomatic, hospitalized, waning immunity), change the seasonal forcing (add a "
        "harmonic, shift the phase, modulate amplitude), change the reporting/observation model, "
        "or add regional coupling. Keep EVERY contract rule and the exact two-form structure of "
        "the examples, and keep the model COMPACT (a focused change, not many extra parameters). "
        "Output ONLY the two S-expressions in one ```scheme block.")


def crossover_prompt(a_source: str, a_rmse: float, b_source: str, b_rmse: float) -> str:
    return (
        f"Here are two influenza forecasting models. Model A (validation RMSE {a_rmse:.5f}):\n"
        f"```scheme\n{a_source}\n```\n\nModel B (validation RMSE {b_rmse:.5f}):\n"
        f"```scheme\n{b_source}\n```\n\n"
        "Combine their best structural ideas into a SINGLE new model likely to forecast better "
        "than both. Keep EVERY contract rule and the exact two-form structure. Output ONLY the "
        "two S-expressions in one ```scheme block.")


# --------------------------------------------------------------------------- #
# Fitness: calibrate a program and score held-out validation skill
# --------------------------------------------------------------------------- #

def score_source(source: str, name: str, data: dict, cfg, *, seeds, refit_iters: int,
                 origin_stride: int, test_seasons=None) -> dict:
    """Structural fit + held-out forecast skill for one program source."""
    from .programs import parse_program
    from .paramspec import param_count
    from .calibrate import calibrate
    from .forecast import filter_then_forecast
    from .runner import train_windows

    prog = parse_program(source, name=name)
    windows = train_windows(data, cfg)
    best = None
    for s in seeds:
        fit = calibrate(prog, windows, cfg=cfg, seed=s)
        if best is None or fit.nll < best.nll:
            best = fit
    seasons = test_seasons if test_seasons is not None else cfg.val_seasons
    skill = filter_then_forecast(prog, best.raw, data, cfg=cfg, test_seasons=seasons,
                                 origin_stride=origin_stride, refit_iters=refit_iters)
    return {"val_rmse": float(skill["mean_rmse"]), "train_nll": float(best.nll),
            "n_params": param_count(prog.specs), "skill": skill}


# --------------------------------------------------------------------------- #
# Worker: generate (with repair) one child, then score it
# --------------------------------------------------------------------------- #

def _evolve_cell(payload: dict) -> dict:
    import sys as _sys
    import signal
    _sys.setrecursionlimit(payload["recursion_limit"])
    import torch
    torch.set_num_threads(1)
    from .data.build_data import load_processed
    from .forecast import season_matrix
    from .validity import screen
    from .llm_generate import SPECS, _call_llm, extract_scheme, _repair_suffix

    cfg = payload["cfg"]
    rec = {"name": payload["name"], "gen": payload["gen"], "op": payload["op"],
           "model": payload["spec_key"], "parents": payload["parents"],
           "ok": False, "val_rmse": float("inf"), "stage": "none", "canonical": "", "source": ""}

    # Hard wall-clock cap per child: a complex evolved program can be very slow to score, and
    # one such child would otherwise gate the whole generation's barrier. SIGALRM abandons it.
    def _on_alarm(signum, frame):
        raise TimeoutError("child exceeded wall-clock budget")
    try:
        signal.signal(signal.SIGALRM, _on_alarm)
        signal.alarm(int(payload.get("child_timeout", 1200)))
    except Exception:  # noqa: BLE001  (non-Unix or no main thread)
        pass

    try:
        data = load_processed(Path(payload["processed_dir"]))
        probe = torch.tensor(season_matrix(data, cfg.val_seasons[0])[:12], dtype=torch.float32)
        spec = SPECS[payload["spec_key"]]

        last, res, src = None, None, ""
        for attempt in range(1, cfg.max_repairs + 2):
            prompt = payload["user"] if last is None else payload["user"] + _repair_suffix(*last)
            raw = _call_llm(prompt, spec)
            src = extract_scheme(raw)
            res = screen(src, cfg=cfg, probe_obs=probe, name=payload["name"])
            if res.ok:
                break
            last = (res.stage, res.detail, res.repair_hint)

        rec.update(source=src, stage=(res.stage if res else "none"),
                   canonical=(res.canonical if res else ""), ok=bool(res and res.ok))
        if res and res.ok:
            sc = score_source(src, payload["name"], data, cfg, seeds=cfg.seeds,
                              refit_iters=payload["refit_iters"], origin_stride=payload["origin_stride"])
            rec.update(val_rmse=sc["val_rmse"], train_nll=sc["train_nll"], n_params=sc["n_params"])
    except TimeoutError as exc:
        rec["error"] = f"timeout: {exc}"
    except Exception as exc:  # noqa: BLE001  (a bad child must never kill the run)
        rec["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            signal.alarm(0)
        except Exception:  # noqa: BLE001
            pass
    return rec


# --------------------------------------------------------------------------- #
# The evolutionary loop
# --------------------------------------------------------------------------- #

def _log_child(r: dict, label: str) -> None:
    tag = "ok " if r.get("ok") else "xx "
    extra = (f" val={r['val_rmse']:.5f}" if r.get("ok") else
             f" @{r.get('stage','?')}" + (f" ERR={r['error'][:50]}" if r.get("error") else ""))
    print(f"  [{label}] {tag}{r['name']} {r.get('op')}/{r.get('model')}{extra}", flush=True)


def _run_pool(payloads, workers, label=""):
    if workers <= 1:
        out = []
        for p in payloads:
            r = _evolve_cell(p)
            _log_child(r, label)
            out.append(r)
        return out
    import multiprocessing as mp
    from concurrent.futures import as_completed
    ctx = mp.get_context("spawn")
    out = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        futs = {ex.submit(_evolve_cell, p): p for p in payloads}
        for fut in as_completed(futs):  # report each child as it lands -> a watchable log
            r = fut.result()
            _log_child(r, label)
            out.append(r)
    return out


def _payload(name, gen, op, spec_key, parents, user, cfg, refit_iters, origin_stride):
    return dict(name=name, gen=gen, op=op, spec_key=spec_key, parents=parents, user=user,
                cfg=cfg, processed_dir=str(PROCESSED), recursion_limit=cfg.recursion_limit,
                refit_iters=refit_iters, origin_stride=origin_stride)


def _checkpoint(output_dir: Path, archive: list, curve: list) -> None:
    """Persist progress after every generation so an interrupted overnight run is recoverable."""
    (output_dir / "evolve_archive.json").write_text(json.dumps(archive, indent=2, default=float))
    (output_dir / "evolve_curve.json").write_text(json.dumps(curve, indent=2, default=float))


def evolve(cfg: FluZooConfig, *, generations: int, pop: int, elite: int, models: list[str],
           workers: int, refit_iters: int, origin_stride: int, seed: int = 0,
           child_timeout: int = 1200, output_dir: Path = RESULTS) -> dict:
    from .llm_generate import sample_recipes, _user_prompt
    from .data.build_data import load_processed

    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    archive: list[dict] = []
    curve: list[dict] = []
    evals = 0

    def best_of(recs):
        ok = [r for r in recs if r.get("ok")]
        return min(ok, key=lambda r: r["val_rmse"]) if ok else None

    # ---- generation 0: recipe-seeded proposals ---------------------------
    recipes = sample_recipes(pop, seed=seed)
    payloads = [_payload(f"g0_{i:03d}", 0, "seed", models[i % len(models)], [],
                         _user_prompt(recipes[i]), cfg, refit_iters, origin_stride)
                for i in range(pop)]
    for p in payloads:
        p["child_timeout"] = child_timeout
    gen_recs = _run_pool(payloads, workers, label="gen0")
    archive += gen_recs
    evals += len(gen_recs)
    best = best_of(archive)
    curve.append({"eval": evals, "gen": 0, "best_val_rmse": best["val_rmse"] if best else None,
                  "n_valid": sum(r.get("ok", False) for r in gen_recs)})
    print(f"[gen 0] seed: {sum(r.get('ok',False) for r in gen_recs)}/{pop} valid; "
          f"best val_rmse={best['val_rmse']:.5f}" if best else "[gen 0] no valid", flush=True)
    _checkpoint(output_dir, archive, curve)

    # ---- generations 1..G: mutate / crossover the elite ------------------
    for g in range(1, generations + 1):
        valid = [r for r in archive if r.get("ok")]
        elites = sorted(valid, key=lambda r: r["val_rmse"])[:elite]
        payloads = []
        for i in range(pop):
            spec_key = models[i % len(models)]
            name = f"g{g}_{i:03d}"
            if len(elites) >= 2 and i % 3 == 0:                       # crossover
                a, b = rng.sample(elites, 2)
                user = crossover_prompt(a["source"], a["val_rmse"], b["source"], b["val_rmse"])
                payloads.append(_payload(name, g, "crossover", spec_key,
                                         [a["name"], b["name"]], user, cfg, refit_iters, origin_stride))
            elif elites:                                             # mutate an elite
                p = elites[i % len(elites)]
                user = mutate_prompt(p["source"], p["val_rmse"])
                payloads.append(_payload(name, g, "mutate", spec_key, [p["name"]],
                                         user, cfg, refit_iters, origin_stride))
            else:                                                    # fallback: fresh seed
                r = sample_recipes(pop, seed=seed + g)[i]
                payloads.append(_payload(name, g, "seed", spec_key, [],
                                         _user_prompt(r), cfg, refit_iters, origin_stride))
        for p in payloads:
            p["child_timeout"] = child_timeout
        gen_recs = _run_pool(payloads, workers, label=f"gen{g}")
        archive += gen_recs
        evals += len(gen_recs)
        best = best_of(archive)
        improved = [r for r in gen_recs if r.get("ok") and r["val_rmse"] < (curve[-1]["best_val_rmse"] or 1e9)]
        curve.append({"eval": evals, "gen": g, "best_val_rmse": best["val_rmse"] if best else None,
                      "n_valid": sum(r.get("ok", False) for r in gen_recs),
                      "n_improved": len(improved)})
        print(f"[gen {g}] {sum(r.get('ok',False) for r in gen_recs)}/{pop} valid, "
              f"{len(improved)} improved; best val_rmse="
              f"{best['val_rmse']:.5f}" if best else f"[gen {g}] no valid", flush=True)
        _checkpoint(output_dir, archive, curve)

    # ---- final: score the best program on the TEST seasons ---------------
    test_block = None
    if best is not None:
        data = load_processed(PROCESSED)
        test = score_source(best["source"], best["name"], data, cfg, seeds=cfg.seeds,
                             refit_iters=refit_iters, origin_stride=origin_stride,
                             test_seasons=cfg.test_seasons)
        test_block = {"test_mean_rmse": test["val_rmse"], "test": test["skill"],
                      "n_params": test["n_params"]}

    uniq = len({r["canonical"] for r in archive if r.get("ok") and r.get("canonical")})
    summary = {
        "generations": generations, "pop": pop, "elite": elite, "models": models,
        "evals": evals, "n_valid": sum(r.get("ok", False) for r in archive),
        "unique_structures": uniq,
        "best": None if best is None else {
            "name": best["name"], "gen": best["gen"], "op": best["op"], "model": best["model"],
            "val_rmse": best["val_rmse"], "n_params": best.get("n_params"),
            "source": best["source"], **(test_block or {})},
        "skill_curve": curve,
        "per_model": _per_model(archive),
        "config": {"refit_iters": refit_iters, "origin_stride": origin_stride,
                   "adam_iters": cfg.adam_iters, "seeds": list(cfg.seeds),
                   "val_seasons": list(cfg.val_seasons), "test_seasons": list(cfg.test_seasons)},
    }
    (output_dir / "evolve_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    # full archive (sources) separately so the summary stays readable
    (output_dir / "evolve_archive.json").write_text(json.dumps(archive, indent=2, default=float))
    print("\n[evolve] " + json.dumps({k: summary[k] for k in
          ("evals", "n_valid", "unique_structures")}))
    if best:
        print(f"[best] {best['name']} ({best['op']}, {best['model']}) "
              f"val_rmse={best['val_rmse']:.5f}"
              + (f"  test_rmse={test_block['test_mean_rmse']:.5f}" if test_block else ""))
    return summary


def _per_model(archive) -> dict:
    out = {}
    for r in archive:
        m = r.get("model", "?")
        d = out.setdefault(m, {"n": 0, "valid": 0, "best_val_rmse": None})
        d["n"] += 1
        if r.get("ok"):
            d["valid"] += 1
            if d["best_val_rmse"] is None or r["val_rmse"] < d["best_val_rmse"]:
                d["best_val_rmse"] = r["val_rmse"]
    return out


def merge_islands(dirs: list[str], cfg: FluZooConfig, *, refit_iters: int,
                  origin_stride: int, output_dir: Path = RESULTS) -> dict:
    """Pool K parallel island runs (different seeds, run across nodes) into one result.

    Combines their archives, finds the global best (re-scored on TEST), and builds the
    guided skill curve at the K-fold-higher evaluation rate (islands advance in parallel).
    """
    from .data.build_data import load_processed
    archives, curves = [], []
    for d in dirs:
        d = Path(d)
        try:
            archives.append(json.loads((d / "evolve_archive.json").read_text()))
        except Exception:  # noqa: BLE001
            archives.append([])
        try:
            curves.append(json.loads((d / "evolve_curve.json").read_text()))
        except Exception:  # noqa: BLE001
            curves.append([])
    combined = [r for a in archives for r in a]
    ok = [r for r in combined if r.get("ok")]
    best = min(ok, key=lambda r: r["val_rmse"]) if ok else None

    # combined curve: islands run concurrently, so at generation g the pooled budget is the sum
    # of per-island evals and the frontier is the min of per-island bests.
    G = max((len(c) for c in curves), default=0)
    comb_curve = []
    for g in range(G):
        evals = sum(c[g]["eval"] for c in curves if g < len(c))
        bests = [c[g]["best_val_rmse"] for c in curves
                 if g < len(c) and c[g].get("best_val_rmse") is not None]
        comb_curve.append({"gen": g, "eval": evals,
                           "best_val_rmse": min(bests) if bests else None})

    test_block = None
    if best is not None:
        data = load_processed(PROCESSED)
        test = score_source(best["source"], best["name"], data, cfg, seeds=cfg.seeds,
                            refit_iters=refit_iters, origin_stride=origin_stride,
                            test_seasons=cfg.test_seasons)
        test_block = {"test_mean_rmse": test["val_rmse"], "test": test["skill"],
                      "n_params": test["n_params"]}
    uniq = len({r["canonical"] for r in ok if r.get("canonical")})
    summary = {
        "n_islands": len(dirs), "evals": len(combined), "n_valid": len(ok),
        "unique_structures": uniq,
        "best": None if best is None else {
            **{k: best.get(k) for k in ("name", "op", "model", "val_rmse", "n_params", "source")},
            **(test_block or {})},
        "skill_curve": comb_curve, "per_model": _per_model(combined),
    }
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / "evolve_islands_summary.json").write_text(
        json.dumps(summary, indent=2, default=float))
    print(f"[islands] {len(dirs)} islands, {len(combined)} evals, {len(ok)} valid, "
          f"{uniq} unique; best val_rmse={best['val_rmse']:.5f}" if best else "[islands] no valid",
          flush=True)
    if test_block:
        print(f"[islands] best test_rmse={test_block['test_mean_rmse']:.5f}", flush=True)
    return summary


def main():
    from .llm_generate import SPECS, DEFAULT_MODELS
    ap = argparse.ArgumentParser(description="LLM-guided evolutionary program search")
    ap.add_argument("--generations", type=int, default=6)
    ap.add_argument("--pop", type=int, default=24)
    ap.add_argument("--elite", type=int, default=6)
    ap.add_argument("--models", default="qwen35,qwen27,gpt55")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--refit-iters", type=int, default=5)
    ap.add_argument("--origin-stride", type=int, default=8)
    ap.add_argument("--child-timeout", type=int, default=1200,
                    help="per-child wall-clock cap (s); slow/complex children are abandoned")
    ap.add_argument("--adam-iters", type=int, default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", default=str(RESULTS))
    ap.add_argument("--merge-islands", nargs="*", default=None,
                    help="pool these island output dirs into one combined result")
    args = ap.parse_args()

    overrides = {}
    if args.adam_iters is not None:
        overrides["adam_iters"] = args.adam_iters
    if args.seeds is not None:
        overrides["seeds"] = tuple(args.seeds)
    cfg = dataclasses.replace(DEFAULT, **overrides) if overrides else DEFAULT

    if args.merge_islands:
        merge_islands(args.merge_islands, cfg, refit_iters=args.refit_iters,
                      origin_stride=args.origin_stride, output_dir=Path(args.output_dir))
        return

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    for m in models:
        if m not in SPECS:
            raise SystemExit(f"unknown model {m!r}; choices: {list(SPECS)}")
    evolve(cfg, generations=args.generations, pop=args.pop, elite=args.elite, models=models,
           workers=args.workers, refit_iters=args.refit_iters, origin_stride=args.origin_stride,
           seed=args.seed, child_timeout=args.child_timeout, output_dir=Path(args.output_dir))


if __name__ == "__main__":
    main()
