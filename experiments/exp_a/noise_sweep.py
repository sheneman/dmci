############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# noise_sweep.py: Experiment A noise-robustness sweep (review item R4). The headline experiments recover constants from...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Experiment A noise-robustness sweep (review item R4).

The headline experiments recover constants from noiseless, self-generated data. A fair reviewer
asks whether DMCI's parameter recovery degrades gracefully under measurement noise. This sweep
re-runs the Experiment-A constant-recovery task through the *same compiled self-hosted interpreter*
(`run_compiled_interp` path) at several noise levels and reports the recovered parameters' relative
error versus SNR.

Noise model: additive Gaussian on the targets, scaled to the signal's standard deviation ---
  y_noisy = y + sigma * std(y) * N(0,1),  sigma in {0, 0.02, 0.05, 0.10, 0.20}
so sigma is an interpretable fraction of the signal (sigma=0.10 ~ 10% noise). sigma=0 is the
noiseless control (matching the main experiments).

Metrics, per (program, sigma), averaged over seeds:
  - relative parameter error: mean_n |theta_hat_n - theta*_n| / max(|theta*_n|, eps)  (the robustness metric;
    independent of the noise, since theta* is the TRUE constant)
  - clean-data MSE: MSE of the recovered parameters against the NOISELESS targets (recovery of the
    true function, not the noise realization)

Because the noisy loss plateaus near the noise floor (it cannot reach the 1e-3 convergence
threshold), we use a best-loss patience early stop and report the parameters at the best noisy loss.

Run on HPC:  python -m experiments.exp_a.noise_sweep
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from .config import DEFAULT, ExpAConfig
from .programs import PROGRAMS_BY_NAME, _all_input_names
from .baselines import (_get_graph, _make_params, _generate_data, _compute_loss,
                        _build_tagged_inputs)

OUTPUT_DIR = Path(__file__).parent / "results"


def _compute_loss_batched(graph, spec, params, xs, ys):
    """One batched interpreter walk over all N data points (vs N sequential walks). The compiled
    meta-circular interpreter batches natively for data-independent control flow (Experiment H),
    reproducing the sequential forward bit-for-bit."""
    from neural_compiler.evaluator import evaluate_batched
    from neural_compiler.runtime.tagged_value import make_float, unwrap_number
    tagged = {spec.input_names[0]: make_float(xs)}
    for name, p in params.items():
        tagged[name] = make_float(p)
    preds = unwrap_number(evaluate_batched(graph, tagged))
    return ((preds - ys) ** 2).sum()


def _batched_matches_sequential(spec, cfg, tol=1e-4) -> bool:
    """True iff batched and sequential losses agree at the init params (per-program safety gate)."""
    graph = _get_graph(f"ns_{spec.name}", spec.interp_source, _all_input_names(spec))
    params = _make_params(spec, 0)
    xs, ys = _generate_data(spec, cfg)
    try:
        seq = _compute_loss(graph, spec, params, xs, ys, _build_tagged_inputs).item()
        bat = _compute_loss_batched(graph, spec, params, xs, ys).item()
        return abs(seq - bat) <= tol * (abs(seq) + 1e-9)
    except Exception:
        return False


def fit_noisy(spec, cfg: ExpAConfig, seed: int, sigma: float,
              max_epochs: int, patience: int, use_batched: bool = False) -> dict:
    """Recover spec's constants from noisy data through the compiled interpreter (DMCI)."""
    graph = _get_graph(f"ns_{spec.name}", spec.interp_source, _all_input_names(spec))
    params = _make_params(spec, seed)
    xs, ys_clean = _generate_data(spec, cfg)

    if sigma > 0.0:
        g = torch.Generator().manual_seed(seed * 100003 + int(round(sigma * 1000)) + 7)
        scale = float(ys_clean.std().clamp(min=1e-8))
        ys = ys_clean + torch.randn(ys_clean.shape, generator=g) * (sigma * scale)
    else:
        ys = ys_clean

    opt = torch.optim.Adam(list(params.values()), lr=cfg.lr)
    best_loss = float("inf")                 # true minimum loss (for best params)
    best = {n: p.item() for n, p in params.items()}
    plateau_ref = float("inf")               # last loss at a *significant* relative drop
    wait = 0
    epoch = 0
    for epoch in range(max_epochs):
        loss = (_compute_loss_batched(graph, spec, params, xs, ys) if use_batched
                else _compute_loss(graph, spec, params, xs, ys, _build_tagged_inputs))
        opt.zero_grad()
        if torch.isfinite(loss):
            loss.backward()
            if all(p.grad is None or torch.isfinite(p.grad).all() for p in params.values()):
                torch.nn.utils.clip_grad_norm_(list(params.values()), 10.0)
                opt.step()
        lv = loss.item()
        if lv < best_loss:
            best_loss, best = lv, {n: p.item() for n, p in params.items()}
        # Plateau stop on RELATIVE improvement (NOT the 1e-3 threshold): a threshold stop fires only
        # for sigma=0 and truncates it before convergence, unfairly inflating its error. Every sigma
        # instead trains until its own loss stops dropping by >=0.01%/epoch, i.e. to its own optimum.
        if lv < plateau_ref * (1.0 - 1e-5):
            plateau_ref, wait = lv, 0
        else:
            wait += 1
        if wait > patience:
            break

    n = len(xs)
    rel = [abs(best[name] - spec.target_values[name]) / max(abs(spec.target_values[name]), 1e-8)
           for name in spec.param_names]
    with torch.no_grad():
        for name, p in params.items():
            p.copy_(torch.tensor(best[name]))
        clean_mse = ((_compute_loss_batched(graph, spec, params, xs, ys_clean) if use_batched
                      else _compute_loss(graph, spec, params, xs, ys_clean, _build_tagged_inputs)).item()) / n
    return {"rel_param_err": sum(rel) / len(rel), "clean_mse": clean_mse,
            "noisy_best_mse": best_loss / n, "epochs": epoch + 1, "fitted": best}


def main():
    ap = argparse.ArgumentParser(description="Experiment A noise-robustness sweep (R4)")
    ap.add_argument("--programs", nargs="+",
                    default=["P1_single_const", "P2_multi_const", "P3_recursive",
                             "P4_higher_order", "P5_multi_function"])
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0.0, 0.02, 0.05, 0.10, 0.20])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--max-epochs", type=int, default=1000)
    ap.add_argument("--patience", type=int, default=120)
    ap.add_argument("--output", type=Path, default=OUTPUT_DIR / "noise_sweep.json")
    args = ap.parse_args()

    import sys
    sys.setrecursionlimit(5000)
    cfg = DEFAULT
    cells = []
    t0 = time.perf_counter()
    print(f"Noise sweep: programs={args.programs} sigmas={args.sigmas} seeds={args.seeds}", flush=True)
    for pname in args.programs:
        spec = PROGRAMS_BY_NAME[pname]
        use_batched = _batched_matches_sequential(spec, cfg)
        print(f"  [{pname}] batched eval: {'YES' if use_batched else 'NO (sequential fallback)'}", flush=True)
        for sigma in args.sigmas:
            errs, cmses, eps = [], [], []
            for seed in range(args.seeds):
                r = fit_noisy(spec, cfg, seed, sigma, args.max_epochs, args.patience, use_batched)
                errs.append(r["rel_param_err"]); cmses.append(r["clean_mse"]); eps.append(r["epochs"])
            cell = {"program": pname, "sigma": sigma, "n_seeds": args.seeds,
                    "mean_rel_param_err": statistics.mean(errs),
                    "std_rel_param_err": (statistics.stdev(errs) if len(errs) > 1 else 0.0),
                    "mean_clean_mse": statistics.mean(cmses),
                    "mean_epochs": statistics.mean(eps)}
            cells.append(cell)
            print(f"  {pname:18s} sigma={sigma:4.2f} | rel_param_err={cell['mean_rel_param_err']:.3e}"
                  f" +/- {cell['std_rel_param_err']:.1e} | clean_mse={cell['mean_clean_mse']:.2e}"
                  f" | epochs~{cell['mean_epochs']:.0f}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"cells": cells, "config": {
        "sigmas": args.sigmas, "seeds": args.seeds, "max_epochs": args.max_epochs,
        "patience": args.patience, "lr": cfg.lr, "n_data_points": cfg.n_data_points}}, indent=2))
    print(f"\nSaved: {args.output}  (wall {time.perf_counter()-t0:.0f}s)")

    # aggregate: mean relative param error vs sigma across programs
    print("\n=== mean relative parameter error vs noise (averaged over programs) ===")
    for sigma in args.sigmas:
        vals = [c["mean_rel_param_err"] for c in cells if c["sigma"] == sigma]
        print(f"  sigma={sigma:4.2f}:  {statistics.mean(vals):.3e}")


if __name__ == "__main__":
    main()
