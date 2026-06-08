############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# run_pilot.py: Exp I de-risking pilot: a <2h go/no-go gate before the full grid. Runs (1 static GPP family, 3 seeds, clean...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Exp I de-risking pilot: a <2h go/no-go gate before the full grid.

Runs (1 static GPP family, 3 seeds, clean synthetic):
  - DMCI fit, direct-compile fit, lambdify->JAX fit, differential-evolution fit
  - predictive (held-out) MSE for each  (PRIMARY metric)
  - DMCI-vs-direct prediction agreement  (interpreter correctness)
  - the lambdify recursion-rejection demo  (pillar 2 engineering-cost delta)
  - the AmeriFlux US-Ha1 loader probe       (pillar 1 obtainability)
then evaluates the six GO criteria from the one-pager and prints a verdict.

Run on HPC (CPU 'eight' partition), NOT locally:  python -m experiments.exp_i.run_pilot
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch

from .config import DEFAULT, PILOT_GATE, ExpIConfig
from .models import build_static_model, build_recursive_pool_model
from .harness import (run_dmci, run_direct, run_diffevo, heldout_mse,
                      load_ameriflux, _predict, _compile)
from .lambdify_baseline import run_lambdify_jax


def _path_agreement(model, cfg, seed) -> float:
    """Max relative disagreement between DMCI and direct predictions at the
    fitted (target) params over a few driver points — interpreter correctness."""
    dmci_g = _compile(model, "dmci")
    direct_g = _compile(model, "direct")
    g = torch.Generator().manual_seed(seed + 777)
    worst = 0.0
    with torch.no_grad():
        for _ in range(8):
            vals = {}
            for d in model.input_names:
                lo, hi = model.driver_ranges[d]
                vals[d] = (lo + (hi - lo) * torch.rand(1, generator=g)).item()
            for n in model.param_names:
                vals[n] = model.target_values[n]
            a = _predict(dmci_g, vals).item()
            b = _predict(direct_g, vals).item()
            denom = max(abs(a), abs(b), 1e-9)
            worst = max(worst, abs(a - b) / denom)
    return worst


def run_pilot(cfg: ExpIConfig = DEFAULT, output_dir: Path | None = None) -> dict:
    output_dir = output_dir or (Path(__file__).parent / "results")
    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_static_model(n_pft=cfg.n_pft)
    print(f"=== Exp I pilot — {model.name} "
          f"({len(model.param_names)} params, {cfg.n_data_points} pts) ===")
    print(f"expression: {model.expression[:90]}...")

    methods = {"dmci": run_dmci, "direct": run_direct, "diffevo": run_diffevo}
    per_seed = []
    for seed in cfg.seeds:
        print(f"\n--- seed {seed} ---")
        row = {"seed": seed, "fits": {}}
        for mname, fn in methods.items():
            r = fn(model, cfg, seed)
            hmse = heldout_mse(model, r.fitted_values, cfg, seed)
            row["fits"][mname] = {
                "best_mse": r.best_mse, "heldout_mse": hmse,
                "converged": r.converged, "n_epochs": r.n_epochs,
                "t_fit_s": r.t_fit_s, "nan_stall": r.nan_stall,
                "mean_param_rel_error":
                    sum(r.param_rel_error.values()) / len(r.param_rel_error),
            }
            print(f"  {mname:8} best_mse={r.best_mse:.3e} "
                  f"heldout={hmse:.3e} conv={r.converged} "
                  f"t={r.t_fit_s/60:.1f}min nan={r.nan_stall}")
        # lambdify->JAX (closed-form parity)
        lr = run_lambdify_jax(model, cfg, seed)
        row["fits"]["lambdify_jax"] = {
            "expressible": lr.expressible, "best_mse": lr.best_mse,
            "heldout_mse": (heldout_mse(model, lr.fitted_values, cfg, seed)
                            if lr.expressible else float("inf")),
            "converged": lr.converged, "t_fit_s": lr.t_fit_s, "reason": lr.reason,
        }
        print(f"  {'lambdify':8} expressible={lr.expressible} "
              f"best_mse={lr.best_mse:.3e} t={lr.t_fit_s:.1f}s")
        per_seed.append(row)

    # interpreter correctness
    agree = _path_agreement(model, cfg, cfg.seeds[0])
    print(f"\nDMCI-vs-direct max rel. disagreement: {agree:.2e}")

    # pillar 2: lambdify must REJECT the recursive carbon pool (nonzero human cost)
    rec = build_recursive_pool_model()
    rec_lr = run_lambdify_jax(rec, cfg, cfg.seeds[0])
    print(f"recursive carbon-pool via lambdify: expressible={rec_lr.expressible} "
          f"-> {rec_lr.reason}")

    # pillar 1: AmeriFlux obtainability probe
    try:
        load_ameriflux()
        ameriflux_status = "loaded"
    except FileNotFoundError as e:
        ameriflux_status = f"absent (expected): {str(e).splitlines()[0]}"
    except NotImplementedError as e:
        ameriflux_status = f"present-needs-mapping: {e}"
    print(f"AmeriFlux US-Ha1 probe: {ameriflux_status}")

    gate = _evaluate_gate(per_seed, agree, rec_lr.expressible, ameriflux_status)
    result = {
        "model": model.name,
        "n_params": len(model.param_names),
        "config": asdict(cfg),
        "per_seed": per_seed,
        "dmci_vs_direct_rel_disagreement": agree,
        "recursive_lambdify_expressible": rec_lr.expressible,
        "recursive_lambdify_reason": rec_lr.reason,
        "ameriflux_status": ameriflux_status,
        "gate": gate,
    }
    out = output_dir / "pilot_result.json"
    out.write_text(json.dumps(result, indent=2, default=str))
    print(f"\nSaved: {out}")
    _print_verdict(gate)
    return result


def _evaluate_gate(per_seed, agree, rec_expressible, ameriflux_status) -> dict:
    g = PILOT_GATE
    dmci = [s["fits"]["dmci"] for s in per_seed]
    n = len(dmci)
    n_conv = sum(1 for d in dmci if d["heldout_mse"] < g.mse_go)
    mean_fit_min = sum(d["t_fit_s"] for d in dmci) / n / 60.0
    any_stall = any(d["nan_stall"] for d in dmci)
    checks = {
        "c1_predictive_converge": (n_conv / n) >= g.frac_seeds_converge,
        "c2_fit_under_45min": mean_fit_min < g.fit_minutes_go,
        "c3_dmci_direct_agree": agree < g.path_agreement_rtol,
        "c4_no_nan_stall": not any_stall,
        "c5_lambdify_cant_recurse": not rec_expressible,
        "c6_ameriflux_obtainable": "loaded" in ameriflux_status
                                   or "needs-mapping" in ameriflux_status,
    }
    checks["_detail"] = {
        "seeds_converged": f"{n_conv}/{n}", "mean_dmci_fit_min": mean_fit_min,
        "dmci_direct_rel_disagreement": agree,
    }
    # c6 is "obtainable" — the stub being absent is expected; flag it as a manual
    # follow-up rather than a hard fail.
    checks["c6_ameriflux_obtainable_note"] = (
        "data file expected absent until downloaded; obtainability is a manual "
        "acquisition step, not a code gate")
    blocking = [k for k in ("c1_predictive_converge", "c2_fit_under_45min",
                            "c3_dmci_direct_agree", "c4_no_nan_stall",
                            "c5_lambdify_cant_recurse") if not checks[k]]
    checks["GO"] = len(blocking) == 0
    checks["blocking_failures"] = blocking
    return checks


def _print_verdict(gate: dict):
    print("\n" + "=" * 56)
    print("PILOT VERDICT:", "GO" if gate["GO"] else "NO-GO")
    for k, v in gate.items():
        if k.startswith("c") and isinstance(v, bool):
            print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    if not gate["GO"]:
        print("  blocking:", gate["blocking_failures"])
        print("  -> if c1/c2 fail: cut params; if c3 fails: interpreter bug;")
        print("     if c5 fails: lambdify reproduces the niche -> fold into F/G.")
    print("=" * 56)


def main():
    ap = argparse.ArgumentParser(description="Exp I pilot")
    ap.add_argument("--n-pft", type=int, default=DEFAULT.n_pft)
    ap.add_argument("--output-dir", type=Path, default=None)
    args = ap.parse_args()
    sys.setrecursionlimit(5000)
    cfg = ExpIConfig(n_pft=args.n_pft)
    run_pilot(cfg, args.output_dir)


if __name__ == "__main__":
    main()
