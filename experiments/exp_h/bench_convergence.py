############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# bench_convergence.py: Experiment H Part G: Convergence speedup from DMCI batching. Fits 15 key DiffESM-S climate parameters to...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Experiment H Part G: Convergence speedup from DMCI batching.

Fits 15 key DiffESM-S climate parameters to synthetic observational data.
Three training conditions — all use Adam with identical learning rate:

1. Sequential: N forward passes per epoch (one per data point)
2. Batched:    1 forward pass per epoch (all N data points)
3. Population: M random restarts × N data points, 1 forward pass

Sequential and batched use identical initial parameters, producing the
same gradient trajectory — only wall-clock time differs.  Population
batching explores M independent initializations simultaneously.

Usage:
  python -m experiments.exp_h.bench_convergence [--device cpu]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.setrecursionlimit(5000)

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from neural_compiler.compiler import compile_scheme
from neural_compiler.evaluator import evaluate_batched

OUTPUT_DIR = Path(__file__).parent / "results"
DIFFESM_PATH = PROJECT_ROOT / "large_examples" / "diffesm_s.scm"

DEFAULT_PARAMS = {
    "n_steps": 0.0,
    "C_atm_0": 590.0, "C_upper_0": 900.0, "C_deep_0": 37000.0,
    "C_veg_0": 550.0, "C_soilf_0": 300.0, "C_soils_0": 1200.0,
    "C_perm_0": 1400.0, "CH4_0": 1900.0, "N2O_0": 332.0,
    "Ts_0": 0.0, "Td_0": 0.0, "ice_0": 0.7,
    "sulf_0": 1.0, "bca_0": 0.1,
    "b1_l_0": 5.0, "b1_w_0": 200.0,
    "b2_l_0": 4.0, "b2_w_0": 100.0,
    "b3_l_0": 2.0, "b3_w_0": 30.0,
    "eCO2": 10.0, "eCH4": 100.0, "eN2O": 4.0, "eSO2": 50.0,
    "p_rf_co2": 5.35, "p_rf_co2r": 590.0,
    "p_rf_ch4": 0.036, "p_rf_ch4r": 1900.0,
    "p_rf_n2o": 0.12, "p_rf_n2or": 332.0,
    "p_rf_sulf": 0.1, "p_rf_bc": 0.05, "p_rf_ind": -0.15,
    "p_a_ice": 0.6, "p_a_ocn": 0.06, "p_a_ref": 0.438,
    "p_rf_alb": 1.5, "p_rf_vol": 0.0, "p_rf_fb": 1.0,
    "p_lam": 1.2, "p_kht": 0.73, "p_Cm": 8.0, "p_Cd": 100.0,
    "p_cf": 0.3, "p_npp": 60.0, "p_npp_to": 2.0,
    "p_tr": 0.04, "p_resp": 0.02, "p_lit": 0.035,
    "p_df": 0.1, "p_ds": 0.005, "p_fs": 0.01,
    "p_pts": 2.0, "p_pt": 0.001,
    "p_ao": 0.05, "p_oa": 0.005,
    "p_ots": 0.01, "p_od": 0.001, "p_do": 0.0005,
    "p_md": 0.09, "p_mts": 0.02, "p_mw": 20.0,
    "p_mp": 0.001, "p_mo": 10.0, "p_mnb": 40.0,
    "p_ns": 1.0, "p_nts": 0.02, "p_nd": 0.008, "p_no": 3.0,
    "p_im": 2.0, "p_it": 3.0, "p_ir": 0.05,
    "p_sf": 0.02, "p_sd": 1.0, "p_alt": 0.02,
    "p_be": 0.001, "p_bd": 0.2,
    "p_b1p": 2.0, "p_b1lt": 0.1, "p_b1wt": 0.01, "p_b1ws": 0.05, "p_b1to": 1.0,
    "p_b2p": 1.5, "p_b2lt": 0.08, "p_b2wt": 0.008, "p_b2ws": 0.03,
    "p_b2to": 0.5, "p_b2fs": 0.02,
    "p_b3p": 0.8, "p_b3lt": 0.15, "p_b3wt": 0.02, "p_b3ws": 0.04,
    "p_b3to": 1.5, "p_b3fs": 0.01,
}

OPT_PARAMS = [
    "p_lam", "p_kht", "p_Cm", "p_Cd",
    "p_rf_co2", "p_rf_ch4", "p_rf_n2o", "p_rf_sulf", "p_rf_bc",
    "p_npp", "p_cf", "p_ao",
    "p_md", "p_tr", "p_pts",
]

N_STEPS = 20
N_DATA = 32
EPOCHS = 300
LR = 0.003
PERTURBATION = 0.15
GRAD_CLIP = 1.0
SEED = 42
POPULATION_SIZES = [1, 10, 50, 200]


def _compile():
    source = DIFFESM_PATH.read_text()
    return compile_scheme(source, inputs=DEFAULT_PARAMS)


def _fixed_inputs(device, exclude=None):
    exclude = exclude or set()
    fixed = {}
    for k, v in DEFAULT_PARAMS.items():
        if k not in exclude:
            fixed[k] = torch.tensor(float(v), dtype=torch.float32, device=device)
    fixed["n_steps"] = torch.tensor(float(N_STEPS), device=device)
    return fixed


def generate_targets(graph, device):
    eCO2_values = torch.linspace(5.0, 20.0, N_DATA, device=device)
    fixed = _fixed_inputs(device, exclude={"eCO2", "n_steps"})
    fixed["n_steps"] = torch.tensor(float(N_STEPS), device=device)

    targets = []
    for i in range(N_DATA):
        inp = dict(fixed)
        inp["eCO2"] = eCO2_values[i]
        with torch.no_grad():
            t = evaluate_batched(graph, inp)
        targets.append(t.detach())
    return eCO2_values, torch.stack(targets)


def _make_init_state(seed):
    torch.manual_seed(seed)
    state = {}
    for k in OPT_PARAMS:
        true_val = DEFAULT_PARAMS[k]
        factor = 1.0 + PERTURBATION * torch.randn(1).item()
        state[k] = true_val * factor
    return state


def train_sequential(graph, eCO2_values, targets, init_state, device):
    params = {k: torch.tensor(v, dtype=torch.float32, device=device,
                               requires_grad=True)
              for k, v in init_state.items()}
    param_list = list(params.values())
    optimizer = torch.optim.Adam(param_list, lr=LR)

    exclude = set(params.keys()) | {"eCO2", "n_steps"}
    fixed = _fixed_inputs(device, exclude=exclude)
    fixed["n_steps"] = torch.tensor(float(N_STEPS), device=device)

    history = []
    t0 = time.perf_counter()

    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        epoch_loss = 0.0

        for i in range(N_DATA):
            inp = dict(fixed)
            inp["eCO2"] = eCO2_values[i]
            inp.update(params)

            pred = evaluate_batched(graph, inp)
            loss_i = (pred - targets[i]) ** 2 / N_DATA
            loss_i.backward()
            epoch_loss += loss_i.item()

        torch.nn.utils.clip_grad_norm_(param_list, GRAD_CLIP)
        optimizer.step()

        elapsed = time.perf_counter() - t0
        history.append({"epoch": epoch, "wall_time": elapsed, "loss": epoch_loss})
        if epoch % 25 == 0 or epoch == EPOCHS - 1:
            print(f"    seq  ep={epoch:3d}  loss={epoch_loss:.6f}  t={elapsed:.1f}s")

    return history


def train_batched(graph, eCO2_values, targets, init_state, device):
    params = {k: torch.tensor(v, dtype=torch.float32, device=device,
                               requires_grad=True)
              for k, v in init_state.items()}
    param_list = list(params.values())
    optimizer = torch.optim.Adam(param_list, lr=LR)

    exclude = set(params.keys()) | {"eCO2", "n_steps"}
    fixed = _fixed_inputs(device, exclude=exclude)
    fixed["n_steps"] = torch.tensor(float(N_STEPS), device=device)

    history = []
    t0 = time.perf_counter()

    for epoch in range(EPOCHS):
        optimizer.zero_grad()

        inp = dict(fixed)
        inp["eCO2"] = eCO2_values
        inp.update(params)

        pred = evaluate_batched(graph, inp)
        loss = ((pred - targets) ** 2).mean()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(param_list, GRAD_CLIP)
        optimizer.step()

        elapsed = time.perf_counter() - t0
        history.append({"epoch": epoch, "wall_time": elapsed, "loss": loss.item()})
        if epoch % 25 == 0 or epoch == EPOCHS - 1:
            print(f"    bat  ep={epoch:3d}  loss={loss.item():.6f}  t={elapsed:.1f}s")

    return history


def train_population(graph, eCO2_values, targets, M, device):
    torch.manual_seed(SEED + 1000 + M)

    pop_params = {}
    for k in OPT_PARAMS:
        true_val = DEFAULT_PARAMS[k]
        noise = torch.randn(M, device=device) * PERTURBATION * abs(true_val)
        pop_params[k] = (
            torch.full((M,), true_val, dtype=torch.float32, device=device) + noise
        ).requires_grad_(True)

    param_list = list(pop_params.values())
    optimizer = torch.optim.Adam(param_list, lr=LR)

    eCO2_tiled = eCO2_values.repeat(M)
    targets_2d = targets.unsqueeze(0).expand(M, N_DATA)

    exclude = set(OPT_PARAMS) | {"eCO2", "n_steps"}
    fixed = _fixed_inputs(device, exclude=exclude)
    fixed["n_steps"] = torch.tensor(float(N_STEPS), device=device)

    history = []
    t0 = time.perf_counter()

    for epoch in range(EPOCHS):
        optimizer.zero_grad()

        inp = dict(fixed)
        inp["eCO2"] = eCO2_tiled
        for k in OPT_PARAMS:
            inp[k] = pop_params[k].repeat_interleave(N_DATA)

        pred = evaluate_batched(graph, inp)
        pred_2d = pred.view(M, N_DATA)

        per_restart = ((pred_2d - targets_2d) ** 2).mean(dim=1)
        loss = per_restart.sum()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(param_list, GRAD_CLIP)
        optimizer.step()

        elapsed = time.perf_counter() - t0
        best = per_restart.min().item()
        mean = per_restart.mean().item()

        history.append({
            "epoch": epoch, "wall_time": elapsed,
            "best_loss": best, "mean_loss": mean,
        })
        if epoch % 25 == 0 or epoch == EPOCHS - 1:
            converged = (per_restart < 1e-3).sum().item()
            print(f"    pop M={M:3d}  ep={epoch:3d}  best={best:.6f}  "
                  f"mean={mean:.6f}  conv={converged}/{M}  t={elapsed:.1f}s")

    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    device = torch.device(args.device)
    if args.device == "cpu":
        torch.set_num_threads(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Compiling DiffESM-S...")
    graph = _compile()
    print(f"  {len(graph.nodes)} nodes")

    print(f"\nGenerating targets (N={N_DATA}, steps={N_STEPS})...")
    eCO2_values, targets = generate_targets(graph, device)
    print(f"  Target Ts range: [{targets.min():.4f}, {targets.max():.4f}]")

    init_state = _make_init_state(SEED + 100)
    print(f"  Perturbed {len(OPT_PARAMS)} params by ~{PERTURBATION*100:.0f}%")

    results = {"config": {
        "n_steps": N_STEPS, "n_data": N_DATA, "epochs": EPOCHS,
        "lr": LR, "perturbation": PERTURBATION, "seed": SEED,
        "opt_params": OPT_PARAMS, "device": args.device,
    }}

    # --- Sequential ---
    print(f"\n{'='*60}")
    print(f"Sequential training ({N_DATA} fwd passes / epoch)")
    print(f"{'='*60}")
    results["sequential"] = train_sequential(
        graph, eCO2_values, targets, init_state, device)

    # --- Batched ---
    print(f"\n{'='*60}")
    print(f"Batched training (1 fwd pass / epoch, bs={N_DATA})")
    print(f"{'='*60}")
    results["batched"] = train_batched(
        graph, eCO2_values, targets, init_state, device)

    # --- Population ---
    for M in POPULATION_SIZES:
        print(f"\n{'='*60}")
        print(f"Population training M={M} (bs={M*N_DATA})")
        print(f"{'='*60}")
        results[f"population_M{M}"] = train_population(
            graph, eCO2_values, targets, M, device)

    # --- Save ---
    out_path = OUTPUT_DIR / f"convergence_{args.device}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # --- Summary ---
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    seq = results["sequential"][-1]
    bat = results["batched"][-1]
    speedup = seq["wall_time"] / bat["wall_time"]

    print(f"\n  Sequential:  {seq['wall_time']:7.1f}s  final_loss={seq['loss']:.6f}")
    print(f"  Batched:     {bat['wall_time']:7.1f}s  final_loss={bat['loss']:.6f}")
    print(f"  Speedup:     {speedup:.1f}x  (same optimization trajectory)")

    bat_time = bat["wall_time"]
    seq_at_bat_time = None
    for h in results["sequential"]:
        if h["wall_time"] <= bat_time:
            seq_at_bat_time = h
    if seq_at_bat_time:
        print(f"\n  At t={bat_time:.1f}s (when batched finishes {EPOCHS} epochs):")
        print(f"    Sequential has completed {seq_at_bat_time['epoch']+1} epochs, "
              f"loss={seq_at_bat_time['loss']:.6f}")
        print(f"    Batched has completed {EPOCHS} epochs, "
              f"loss={bat['loss']:.6f}")

    print(f"\n  Population batching:")
    print(f"  {'M':>5}  {'time':>8}  {'best_loss':>12}  {'mean_loss':>12}  "
          f"{'vs_single':>10}")
    for M in POPULATION_SIZES:
        pop = results[f"population_M{M}"][-1]
        ratio = pop["wall_time"] / bat["wall_time"]
        print(f"  {M:5d}  {pop['wall_time']:7.1f}s  {pop['best_loss']:12.6f}  "
              f"{pop['mean_loss']:12.6f}  {ratio:9.1f}x time")

    seq_time_200 = seq["wall_time"] * 200
    pop200 = results["population_M200"][-1]
    print(f"\n  200 sequential restarts would take: {seq_time_200:.0f}s")
    print(f"  200 population-batched restarts took: {pop200['wall_time']:.1f}s")
    print(f"  Exploration speedup: {seq_time_200/pop200['wall_time']:.0f}x")


if __name__ == "__main__":
    main()
