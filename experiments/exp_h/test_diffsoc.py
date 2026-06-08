############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# test_diffsoc.py: Test harness for DiffSoc-S: compile, forward, backward, gradient check. Usage: python -m...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Test harness for DiffSoc-S: compile, forward, backward, gradient check.

Usage:
  python -m experiments.exp_h.test_diffsoc [--steps N] [--device cpu|cuda]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.setrecursionlimit(10000)

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from neural_compiler.compiler import compile_scheme
from neural_compiler.evaluator import evaluate_batched

DIFFSOC_PATH = PROJECT_ROOT / "large_examples" / "diffsoc_s.scm"

# ── Initial state (96 variables) ────────────────────────────────────────

# Household state: 15 groups × 4 vars (pop, inc, wth, skl) = 60
# Downtown (1) has highest density; outer ring (4,5) lowest
_POP = {
    "lo": [200, 180, 160, 120, 100],
    "mi": [100, 120, 130, 150, 140],
    "hi": [30, 40, 50, 70, 80],
}
_INC = {"lo": 25.0, "mi": 55.0, "hi": 120.0}
_WTH = {"lo": 10.0, "mi": 80.0, "hi": 350.0}
_SKL = {"lo": 2.0, "mi": 5.0, "hi": 8.0}

INITIAL_STATE: dict[str, float] = {}
for cls, pops in _POP.items():
    for i, pop in enumerate(pops, 1):
        INITIAL_STATE[f"{cls}{i}_pop_0"] = float(pop)
        INITIAL_STATE[f"{cls}{i}_inc_0"] = _INC[cls]
        INITIAL_STATE[f"{cls}{i}_wth_0"] = _WTH[cls]
        INITIAL_STATE[f"{cls}{i}_skl_0"] = _SKL[cls]

# Neighborhood state: 5 × 5 vars (prc, rnt, sch, amn, crm) = 25
_NBHD = {
    #       prc    rnt   sch   amn   crm
    1: (350.0, 18.0, 6.0, 7.0, 3.0),   # downtown: expensive, good schools
    2: (280.0, 14.0, 5.5, 6.0, 2.5),   # inner ring A
    3: (260.0, 13.0, 5.0, 5.5, 2.0),   # inner ring B
    4: (200.0, 10.0, 4.0, 4.0, 1.5),   # outer ring A
    5: (180.0,  9.0, 3.5, 3.5, 1.0),   # outer ring B
}
for i, (prc, rnt, sch, amn, crm) in _NBHD.items():
    INITIAL_STATE[f"n{i}_prc_0"] = prc
    INITIAL_STATE[f"n{i}_rnt_0"] = rnt
    INITIAL_STATE[f"n{i}_sch_0"] = sch
    INITIAL_STATE[f"n{i}_amn_0"] = amn
    INITIAL_STATE[f"n{i}_crm_0"] = crm

# Labor market: 3 sectors × 2 vars (dem, wge) = 6
INITIAL_STATE.update({
    "s1_dem_0": 500.0, "s1_wge_0": 30.0,    # service
    "s2_dem_0": 300.0, "s2_wge_0": 55.0,    # professional
    "s3_dem_0": 150.0, "s3_wge_0": 115.0,   # finance/tech
})

# Banking: 2
INITIAL_STATE.update({
    "crd_0": 1000.0,
    "irate_0": 0.05,
})

# Macro: 3
INITIAL_STATE.update({
    "gdp_0": 5000.0,
    "infl_0": 0.02,
    "ineq_0": 3.0,
})

# ── Learnable parameters (87) ───────────────────────────────────────────

PARAMS: dict[str, float] = {
    # Employment (Module 1)
    "p_emp_base": 0.5,
    "p_emp_skl": 0.3,
    "p_emp_dem": 0.002,
    "p_emp_cmt": 0.1,
    "p_emp_tran": 0.05,
    "p_emp_net": 0.01,
    "p_emp_disc": 0.3,

    # Wages (Module 1)
    "p_wge_adj": 0.02,
    "p_wge_inf": 0.3,
    "p_wge_prod": 0.002,
    "p_sec_grow": 0.01,

    # Income (Module 2)
    "p_skl_prm": 0.1,
    "p_inc_xfer": 5.0,
    "p_inc_tax": 0.2,
    "p_inc_adj": 0.1,
    "p_pol_mw": 1.0,

    # Credit (Module 3)
    "p_crd_inc": 2.0,
    "p_crd_wth": 1.0,
    "p_crd_crm": 0.5,
    "p_crd_bias": 1.0,
    "p_crd_adj": 0.1,
    "p_crd_rsk": 0.05,
    "p_crd_def": 0.1,

    # Housing costs (Module 4)
    "p_own_wth": 0.005,
    "p_own_dwn": 0.2,
    "p_own_mort": 0.005,
    "p_pol_sub": 0.1,

    # Consumption / wealth (Module 5)
    "p_cns_lo": 0.7,
    "p_cns_mi": 0.5,
    "p_cns_hi": 0.3,
    "p_cns_base": 3.0,
    "p_wth_ret": 0.01,
    "p_ptax": 0.005,

    # Housing market (Module 6)
    "p_prf_sch": 0.3,
    "p_prf_amn": 0.2,
    "p_prf_crm": 0.4,
    "p_prf_prc": 0.5,
    "p_hsg_spec": 0.01,
    "p_hsg_sel": 0.3,
    "p_hsg_padj": 0.01,
    "p_hsg_mom": 0.02,
    "p_hsg_mrev": 0.05,
    "p_hsg_dep": 0.01,
    "p_hsg_radj": 0.05,
    "p_hsg_ryld": 0.05,
    "p_pol_zone": 1.0,
    "p_pol_rent": 0.0,

    # Schools & skill (Module 7)
    "p_sch_tax": 0.002,
    "p_sch_base": 0.2,
    "p_pol_sch": 0.002,
    "p_sch_fund": 0.01,
    "p_sch_peer": 0.02,
    "p_sch_dec": 0.05,
    "p_sch_spill": 0.05,
    "p_skl_sch": 0.01,
    "p_skl_peer": 0.005,
    "p_skl_inc": 0.01,
    "p_skl_base": 0.01,
    "p_skl_dec": 0.1,

    # Migration (Module 8)
    "p_att_sch": 0.3,
    "p_att_amn": 0.2,
    "p_att_crm": 0.4,
    "p_att_prc": 0.5,
    "p_att_job": 0.3,
    "p_mig_att": 0.0,
    "p_mig_lo": 0.02,
    "p_mig_mi": 0.01,
    "p_mig_hi": 0.005,

    # Neighborhood (Module 9)
    "p_nbr_ainc": 0.1,
    "p_nbr_ainv": 0.05,
    "p_nbr_adec": 0.05,
    "p_nbr_acrm": 0.05,
    "p_nbr_spamn": 0.05,
    "p_nbr_cbase": 0.02,
    "p_nbr_cinc": 0.05,
    "p_nbr_cpol": 0.05,
    "p_nbr_cdec": 0.1,
    "p_nbr_spcrm": 0.05,
    "p_soc_seg": 0.05,

    # Macro (Module 10)
    "p_mac_gdp": 0.005,
    "p_mac_lab": 0.1,
    "p_mac_inf": 0.3,
    "p_mac_ihsg": 0.1,
    "p_mac_idem": 0.05,
    "p_mac_ineq": 1.0,
    "p_mac_pop": 0.01,
    "p_pol_cred": 0.5,
}

# ── Exogenous drivers (5) ───────────────────────────────────────────────

EXOGENOUS: dict[str, float] = {
    "e_grow": 0.02,
    "e_irate": 0.05,
    "e_inv": 1.0,
    "e_hsup": 1.0,
    "e_imm": 0.01,
}

# ── Structural inputs (15) ──────────────────────────────────────────────

STRUCTURAL: dict[str, float] = {
    # Housing capacity per neighborhood
    "cap_1": 500.0, "cap_2": 600.0, "cap_3": 550.0, "cap_4": 700.0, "cap_5": 650.0,
    # Distance to CBD
    "dist_1": 1.0, "dist_2": 3.0, "dist_3": 4.0, "dist_4": 7.0, "dist_5": 8.0,
    # Transit accessibility
    "tran_1": 5.0, "tran_2": 3.0, "tran_3": 3.0, "tran_4": 1.5, "tran_5": 1.0,
}

# ── Combine all inputs ──────────────────────────────────────────────────

DEFAULT_PARAMS: dict[str, float] = {"n_steps": 0.0}
DEFAULT_PARAMS.update(INITIAL_STATE)
DEFAULT_PARAMS.update(PARAMS)
DEFAULT_PARAMS.update(EXOGENOUS)
DEFAULT_PARAMS.update(STRUCTURAL)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    device = torch.device(args.device)
    n_steps = args.steps

    print(f"DiffSoc-S Test Harness  (steps={n_steps}, device={args.device})")
    print(f"  Total inputs: {len(DEFAULT_PARAMS)}")
    print(f"  Params (p_*): {sum(1 for k in DEFAULT_PARAMS if k.startswith('p_'))}")
    print(f"  Initial state (*_0): {sum(1 for k in DEFAULT_PARAMS if k.endswith('_0') and k != 'n_steps')}")

    # ── Compile ──────────────────────────────────────────────────────
    print("\nCompiling DiffSoc-S...")
    t0 = time.perf_counter()
    source = DIFFSOC_PATH.read_text()
    graph = compile_scheme(source, inputs=DEFAULT_PARAMS)
    t_compile = time.perf_counter() - t0
    print(f"  {len(graph.nodes)} nodes, compiled in {t_compile:.1f}s")

    # ── Forward pass ─────────────────────────────────────────────────
    print(f"\nForward pass ({n_steps} timesteps)...")
    inputs = {}
    for k, v in DEFAULT_PARAMS.items():
        inputs[k] = torch.tensor(float(v), dtype=torch.float32, device=device)
    inputs["n_steps"] = torch.tensor(float(n_steps), device=device)

    param_tensors = {}
    for k in PARAMS:
        inputs[k] = torch.tensor(float(PARAMS[k]), dtype=torch.float32,
                                  device=device, requires_grad=True)
        param_tensors[k] = inputs[k]

    t0 = time.perf_counter()
    result = evaluate_batched(graph, inputs)
    t_fwd = time.perf_counter() - t0
    print(f"  Result: {result.item():.6f}")
    print(f"  Forward time: {t_fwd:.3f}s")

    # ── Backward pass ────────────────────────────────────────────────
    print("\nBackward pass...")
    t0 = time.perf_counter()
    result.backward()
    t_bwd = time.perf_counter() - t0
    print(f"  Backward time: {t_bwd:.3f}s")

    # ── Gradient check ───────────────────────────────────────────────
    print("\nGradient flow:")
    n_grad = 0
    n_zero = 0
    n_nan = 0
    for k in sorted(param_tensors):
        g = param_tensors[k].grad
        if g is None:
            print(f"  {k:20s}: NO GRAD")
        elif torch.isnan(g):
            n_nan += 1
            print(f"  {k:20s}: NaN!")
        elif abs(g.item()) < 1e-15:
            n_zero += 1
            print(f"  {k:20s}: {g.item():+.2e}  (effectively zero)")
        else:
            n_grad += 1
            print(f"  {k:20s}: {g.item():+.2e}")

    total = len(param_tensors)
    print(f"\n  Summary: {n_grad}/{total} non-zero, {n_zero} zero, {n_nan} NaN")

    # ── Numerical stability ──────────────────────────────────────────
    if n_steps >= 5:
        print(f"\nNumerical stability check (running 1..{n_steps} steps)...")
        results = []
        for s in range(1, n_steps + 1):
            inp = {}
            for k, v in DEFAULT_PARAMS.items():
                inp[k] = torch.tensor(float(v), dtype=torch.float32, device=device)
            inp["n_steps"] = torch.tensor(float(s), device=device)
            with torch.no_grad():
                r = evaluate_batched(graph, inp)
            results.append(r.item())
            if s in [1, 5, 10, 20, 50] or s == n_steps:
                print(f"  steps={s:3d}: result={r.item():.6f}")
        if any(abs(r) > 1e10 or r != r for r in results):
            print("  WARNING: numerical instability detected!")
        else:
            print("  All values finite and bounded.")

    # ── Batching test ────────────────────────────────────────────────
    print(f"\nBatching test (batch=16)...")
    batch_inputs = {}
    for k, v in DEFAULT_PARAMS.items():
        batch_inputs[k] = torch.tensor(float(v), dtype=torch.float32, device=device)
    batch_inputs["n_steps"] = torch.tensor(float(n_steps), device=device)
    batch_inputs["e_grow"] = torch.linspace(0.0, 0.05, 16, device=device)

    for k in PARAMS:
        batch_inputs[k] = torch.tensor(float(PARAMS[k]), dtype=torch.float32,
                                        device=device, requires_grad=True)

    t0 = time.perf_counter()
    batch_result = evaluate_batched(graph, batch_inputs)
    t_batch_fwd = time.perf_counter() - t0
    print(f"  Shape: {batch_result.shape}")
    print(f"  Range: [{batch_result.min().item():.4f}, {batch_result.max().item():.4f}]")
    print(f"  Time: {t_batch_fwd:.3f}s")

    batch_result.sum().backward()
    print(f"  Batched backward: OK")

    print(f"\n{'='*60}")
    print("DONE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
