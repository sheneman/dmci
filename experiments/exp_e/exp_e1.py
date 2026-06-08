############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# exp_e1.py: Experiment E.1: Operator Recovery via Soft-Dispatch. Recovers unknown operators op1, op2 and constant a* in the...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Experiment E.1: Operator Recovery via Soft-Dispatch.

Recovers unknown operators op1, op2 and constant a* in the template
  f(x) = a* · op1(x, op2(x, x))
from 64 noisy samples, using Gumbel-softmax soft-choice through the
compiled meta-circular evaluator.

12 target tasks × 20 random restarts.
Baselines: exhaustive enumeration (64 combos + least-squares),
evolutionary algorithm (GA), and random search.

Usage:
    # Run a single task
    python3 -m experiments.exp_e.exp_e1 --task 0 --output-dir experiments/exp_e/results

    # Run all 12 tasks (SLURM array or loop)
    python3 -m experiments.exp_e.exp_e1 --task all --output-dir experiments/exp_e/results
"""

from __future__ import annotations

import argparse
import json
import math
import random as _random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn

from neural_compiler.compiler import compile_scheme
from neural_compiler.evaluator.engine import (
    _eval_graph,
    set_soft_choice_gumbel,
    set_soft_choice_tau,
    set_soft_choice_hard,
)

from .config import (
    OPERATORS, OP_LABELS, N_OPS, TASKS,
    ExpE1Config, DEFAULT,
)

# ---------------------------------------------------------------------------
# Template: Scheme source with two soft-choice holes
# ---------------------------------------------------------------------------

# Inner soft-choice: each operator applied to x.
# Binary ops (+,-,*,/) use a canonical second argument that varies with x.
# (- 0 x) = -x and (/ 1 x) = 1/x avoid degenerate (- x x)=0 and (/ x x)=1.
_INNER_OPTIONS = [
    "(+ x x)",                       # 2x
    "(- 0 x)",                       # -x
    "(* x x)",                       # x²
    "(/ 1 (+ (abs x) 0.001))",      # ~1/x
    "(sin x)",
    "(cos x)",
    "(exp x)",
    "(log (+ (abs x) 0.001))",       # ~log|x|
]

# Outer soft-choice: binary ops combine (x, inner), unary ops use inner only.
_OUTER_OPTIONS = [
    "(+ x inner)",
    "(- x inner)",
    "(* x inner)",
    "(/ x (+ (abs inner) 0.001))",   # ~x/inner
    "(sin inner)",
    "(cos inner)",
    "(exp inner)",
    "(log (+ (abs inner) 0.001))",   # ~log|inner|
]

TEMPLATE = (
    "(let ((inner (soft-choice ("
    + " ".join(_INNER_OPTIONS)
    + ") w2))) (* alpha (soft-choice ("
    + " ".join(_OUTER_OPTIONS)
    + ") w1)))"
)


def _compile_template():
    """Compile the soft-choice template once (reused across all restarts)."""
    return compile_scheme(
        TEMPLATE,
        inputs={"x": None, "alpha": None, "w1": None, "w2": None},
    )


# ---------------------------------------------------------------------------
# Target function evaluation (pure Python/torch, no compiler)
# ---------------------------------------------------------------------------

def _apply_inner(x: torch.Tensor, op_idx: int) -> torch.Tensor:
    """Evaluate inner operator on x.  Must match _INNER_OPTIONS ordering."""
    if op_idx == 0:
        return x + x            # 2x
    if op_idx == 1:
        return -x               # (- 0 x)
    if op_idx == 2:
        return x * x            # x²
    if op_idx == 3:
        return 1.0 / (torch.abs(x) + 0.001)  # ~1/x
    if op_idx == 4:
        return torch.sin(x)
    if op_idx == 5:
        return torch.cos(x)
    if op_idx == 6:
        return torch.exp(x)
    if op_idx == 7:
        return torch.log(torch.abs(x) + 0.001)
    raise ValueError(f"Unknown op index: {op_idx}")


def _apply_outer(x: torch.Tensor, inner: torch.Tensor, op_idx: int) -> torch.Tensor:
    """Evaluate outer operator on (x, inner).  Must match _OUTER_OPTIONS ordering."""
    if op_idx == 0:
        return x + inner
    if op_idx == 1:
        return x - inner
    if op_idx == 2:
        return x * inner
    if op_idx == 3:
        return x / (torch.abs(inner) + 0.001)
    if op_idx == 4:
        return torch.sin(inner)
    if op_idx == 5:
        return torch.cos(inner)
    if op_idx == 6:
        return torch.exp(inner)
    if op_idx == 7:
        return torch.log(torch.abs(inner) + 0.001)
    raise ValueError(f"Unknown op index: {op_idx}")


def target_fn(x: torch.Tensor, op1_idx: int, op2_idx: int, a_star: float) -> torch.Tensor:
    inner = _apply_inner(x, op2_idx)
    outer = _apply_outer(x, inner, op1_idx)
    return a_star * outer


# ---------------------------------------------------------------------------
# Generate training data
# ---------------------------------------------------------------------------

def generate_data(
    op1_idx: int, op2_idx: int, a_star: float, cfg: ExpE1Config, seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    n_pos = cfg.n_data // 2
    n_neg = cfg.n_data - n_pos
    xs_pos = torch.rand(n_pos) * (cfg.x_hi - cfg.x_lo) + cfg.x_lo
    xs_neg = -torch.rand(n_neg) * (cfg.x_hi - cfg.x_lo) - cfg.x_lo
    xs = torch.cat([xs_pos, xs_neg])
    xs = xs[torch.randperm(len(xs))]
    ys = target_fn(xs, op1_idx, op2_idx, a_star)
    xs_ad = xs.clone().requires_grad_(True)
    ys_ad = target_fn(xs_ad, op1_idx, op2_idx, a_star)
    dys = torch.autograd.grad(ys_ad.sum(), xs_ad)[0].detach()
    if cfg.noise_std > 0:
        ys = ys + torch.randn_like(ys) * cfg.noise_std
    return xs, ys, dys


# ---------------------------------------------------------------------------
# DMCI soft-dispatch restart
# ---------------------------------------------------------------------------

def run_dmci_restart(
    graph, xs: torch.Tensor, ys: torch.Tensor, dys: torch.Tensor,
    cfg: ExpE1Config, seed: int,
) -> dict:
    """One DMCI soft-dispatch restart with straight-through estimation.

    Training uses straight-through softmax: argmax forward (clean signal
    through the nested soft-choices) with softmax gradients backward.
    Temperature anneals from tau_start to tau_end to sharpen selection.
    """
    torch.manual_seed(seed)
    set_soft_choice_gumbel(False)
    set_soft_choice_hard(True)

    alpha = nn.Parameter(torch.tensor(1.0) + 0.3 * torch.randn(1).squeeze())
    w1 = nn.Parameter(torch.randn(N_OPS) * 2.0)
    w2 = nn.Parameter(torch.randn(N_OPS) * 2.0)

    optimizer = torch.optim.Adam([
        {"params": [w1, w2], "lr": cfg.lr * 4},
        {"params": [alpha], "lr": cfg.lr},
    ])

    t0 = time.perf_counter()
    loss_history = []

    for epoch in range(cfg.n_epochs):
        frac = epoch / max(cfg.n_epochs - 1, 1)
        tau = cfg.tau_start + (cfg.tau_end - cfg.tau_start) * frac
        set_soft_choice_tau(tau)

        total_loss = torch.tensor(0.0)
        nan_count = 0
        for i in range(len(xs)):
            x_i = xs[i].detach().clone().requires_grad_(True)
            values = _eval_graph(graph, {
                "x": x_i,
                "alpha": alpha,
                "w1": w1,
                "w2": w2,
            })
            pred = values[graph.root_id]
            sample_loss = (pred - ys[i]) ** 2
            if cfg.deriv_weight > 0 and pred.requires_grad:
                dpred = torch.autograd.grad(
                    pred, x_i, create_graph=True, retain_graph=True,
                )[0]
                sample_loss = sample_loss + cfg.deriv_weight * (dpred - dys[i]) ** 2
            if torch.isnan(sample_loss) or torch.isinf(sample_loss):
                nan_count += 1
                continue
            sample_loss = torch.clamp(sample_loss, max=1e4)
            total_loss = total_loss + sample_loss
        if nan_count >= len(xs):
            break
        total_loss = total_loss / max(len(xs) - nan_count, 1)

        optimizer.zero_grad()
        if total_loss.requires_grad:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_([alpha, w1, w2], max_norm=10.0)
            optimizer.step()

        if epoch % 500 == 0 or epoch == cfg.n_epochs - 1:
            loss_history.append((epoch, total_loss.item(), tau))

    wall_time = time.perf_counter() - t0

    set_soft_choice_hard(False)
    set_soft_choice_tau(0.01)

    with torch.no_grad():
        final_loss = torch.tensor(0.0)
        for i in range(len(xs)):
            values = _eval_graph(graph, {
                "x": xs[i],
                "alpha": alpha,
                "w1": w1,
                "w2": w2,
            })
            pred = values[graph.root_id]
            sample_loss = (pred - ys[i]) ** 2
            if not (torch.isnan(sample_loss) or torch.isinf(sample_loss)):
                final_loss = final_loss + sample_loss
        final_loss = final_loss / len(xs)

    op1_selected = int(torch.argmax(w1).item())
    op2_selected = int(torch.argmax(w2).item())

    op1_probs = torch.softmax(w1 / 0.01, dim=0).detach().tolist()
    op2_probs = torch.softmax(w2 / 0.01, dim=0).detach().tolist()

    set_soft_choice_hard(False)
    set_soft_choice_tau(1.0)

    return {
        "method": "dmci_soft_dispatch",
        "seed": seed,
        "wall_time_s": wall_time,
        "final_loss": final_loss.item(),
        "alpha": alpha.item(),
        "op1_selected": op1_selected,
        "op1_label": OP_LABELS[op1_selected],
        "op2_selected": op2_selected,
        "op2_label": OP_LABELS[op2_selected],
        "op1_probs": op1_probs,
        "op2_probs": op2_probs,
        "w1_logits": w1.detach().tolist(),
        "w2_logits": w2.detach().tolist(),
        "loss_history": loss_history,
    }


# ---------------------------------------------------------------------------
# Random search baseline
# ---------------------------------------------------------------------------

def run_random_restart(
    xs: torch.Tensor, ys: torch.Tensor,
    op1_true: int, op2_true: int, a_true: float,
    cfg: ExpE1Config, seed: int,
) -> dict:
    rng = torch.Generator().manual_seed(seed)

    t0 = time.perf_counter()

    op1_pick = int(torch.randint(N_OPS, (1,), generator=rng).item())
    op2_pick = int(torch.randint(N_OPS, (1,), generator=rng).item())

    preds = torch.stack([
        _apply_outer(xs[i], _apply_inner(xs[i], op2_pick), op1_pick)
        for i in range(len(xs))
    ])

    valid = ~(torch.isnan(preds) | torch.isinf(preds))
    if valid.sum() < 2:
        wall_time = time.perf_counter() - t0
        return {
            "method": "random_search",
            "seed": seed,
            "wall_time_s": wall_time,
            "final_loss": float("inf"),
            "alpha": 0.0,
            "op1_selected": op1_pick,
            "op1_label": OP_LABELS[op1_pick],
            "op2_selected": op2_pick,
            "op2_label": OP_LABELS[op2_pick],
            "op1_correct": op1_pick == op1_true,
            "op2_correct": op2_pick == op2_true,
        }

    pv = preds[valid]
    yv = ys[valid]
    a_fit = (pv * yv).sum() / (pv * pv).sum()

    fitted_preds = a_fit * preds
    mse = ((fitted_preds[valid] - yv) ** 2).mean().item()
    wall_time = time.perf_counter() - t0

    return {
        "method": "random_search",
        "seed": seed,
        "wall_time_s": wall_time,
        "final_loss": mse,
        "alpha": a_fit.item(),
        "op1_selected": op1_pick,
        "op1_label": OP_LABELS[op1_pick],
        "op2_selected": op2_pick,
        "op2_label": OP_LABELS[op2_pick],
        "op1_correct": op1_pick == op1_true,
        "op2_correct": op2_pick == op2_true,
    }


# ---------------------------------------------------------------------------
# Exhaustive enumeration baseline
# ---------------------------------------------------------------------------

def run_exhaustive_restart(
    xs: torch.Tensor, ys: torch.Tensor,
    op1_true: int, op2_true: int, a_true: float,
    cfg: ExpE1Config, seed: int,
) -> dict:
    """Try all 64 (op1, op2) pairs, least-squares fit a, pick lowest MSE."""
    t0 = time.perf_counter()

    best_mse = float("inf")
    best_op1 = 0
    best_op2 = 0
    best_a = 0.0

    with torch.no_grad():
        for op1 in range(N_OPS):
            for op2 in range(N_OPS):
                inner = _apply_inner(xs, op2)
                outer = _apply_outer(xs, inner, op1)
                valid = ~(torch.isnan(outer) | torch.isinf(outer))
                if valid.sum() < 2:
                    continue
                ov = outer[valid]
                yv = ys[valid]
                denom = (ov * ov).sum()
                if denom.abs() < 1e-12:
                    continue
                a_fit = (ov * yv).sum() / denom
                mse = ((a_fit * ov - yv) ** 2).mean().item()
                if mse < best_mse:
                    best_mse = mse
                    best_op1 = op1
                    best_op2 = op2
                    best_a = a_fit.item()

    wall_time = time.perf_counter() - t0

    return {
        "method": "exhaustive",
        "seed": seed,
        "wall_time_s": wall_time,
        "final_loss": best_mse,
        "alpha": best_a,
        "op1_selected": best_op1,
        "op1_label": OP_LABELS[best_op1],
        "op2_selected": best_op2,
        "op2_label": OP_LABELS[best_op2],
        "op1_correct": best_op1 == op1_true,
        "op2_correct": best_op2 == op2_true,
    }


# ---------------------------------------------------------------------------
# Evolutionary algorithm baseline
# ---------------------------------------------------------------------------

def _ea_fitness(op1: int, op2: int, a: float,
                xs: torch.Tensor, ys: torch.Tensor) -> float:
    """MSE for a (op1, op2, a) individual.  Lower is better."""
    with torch.no_grad():
        inner = _apply_inner(xs, op2)
        outer = _apply_outer(xs, inner, op1)
        preds = a * outer
        valid = ~(torch.isnan(preds) | torch.isinf(preds))
        if valid.sum() < 2:
            return float("inf")
        return ((preds[valid] - ys[valid]) ** 2).mean().item()


def run_evolutionary_restart(
    xs: torch.Tensor, ys: torch.Tensor,
    op1_true: int, op2_true: int, a_true: float,
    cfg: ExpE1Config, seed: int,
) -> dict:
    """GA with mixed discrete (ops) + continuous (a) representation."""
    _random.seed(seed)
    t0 = time.perf_counter()

    pop = []
    for _ in range(cfg.ea_pop_size):
        ind = [_random.randint(0, N_OPS - 1),
               _random.randint(0, N_OPS - 1),
               _random.gauss(1.0, 1.0)]
        pop.append(ind)

    fits = [_ea_fitness(ind[0], ind[1], ind[2], xs, ys) for ind in pop]

    for _gen in range(cfg.ea_generations):
        new_pop = []
        best_idx = min(range(len(fits)), key=lambda i: fits[i])
        new_pop.append(list(pop[best_idx]))

        while len(new_pop) < cfg.ea_pop_size:
            # tournament selection
            def _tournament():
                idxs = _random.sample(range(len(pop)), cfg.ea_tournament_k)
                return pop[min(idxs, key=lambda i: fits[i])]

            p1 = _tournament()
            p2 = _tournament()

            if _random.random() < cfg.ea_crossover_rate:
                child = [
                    p1[0] if _random.random() < 0.5 else p2[0],
                    p1[1] if _random.random() < 0.5 else p2[1],
                    p1[2] if _random.random() < 0.5 else p2[2],
                ]
            else:
                child = list(p1)

            if _random.random() < cfg.ea_mutation_op_prob:
                child[0] = _random.randint(0, N_OPS - 1)
            if _random.random() < cfg.ea_mutation_op_prob:
                child[1] = _random.randint(0, N_OPS - 1)
            child[2] += _random.gauss(0, cfg.ea_mutation_a_std)

            new_pop.append(child)

        pop = new_pop
        fits = [_ea_fitness(ind[0], ind[1], ind[2], xs, ys) for ind in pop]

    # Refine a with least-squares on best individual's operators
    best_idx = min(range(len(fits)), key=lambda i: fits[i])
    best = pop[best_idx]
    with torch.no_grad():
        inner = _apply_inner(xs, best[1])
        outer = _apply_outer(xs, inner, best[0])
        valid = ~(torch.isnan(outer) | torch.isinf(outer))
        if valid.sum() >= 2:
            ov = outer[valid]
            yv = ys[valid]
            denom = (ov * ov).sum()
            if denom.abs() > 1e-12:
                best[2] = ((ov * yv).sum() / denom).item()
                fits[best_idx] = _ea_fitness(best[0], best[1], best[2], xs, ys)

    wall_time = time.perf_counter() - t0
    op1_sel = best[0]
    op2_sel = best[1]

    return {
        "method": "evolutionary",
        "seed": seed,
        "wall_time_s": wall_time,
        "final_loss": fits[best_idx],
        "alpha": best[2],
        "op1_selected": op1_sel,
        "op1_label": OP_LABELS[op1_sel],
        "op2_selected": op2_sel,
        "op2_label": OP_LABELS[op2_sel],
        "op1_correct": op1_sel == op1_true,
        "op2_correct": op2_sel == op2_true,
    }


# ---------------------------------------------------------------------------
# Run one task (all restarts)
# ---------------------------------------------------------------------------

def run_task(task_idx: int, cfg: ExpE1Config, output_dir: Path) -> dict:
    op1_idx, op2_idx, a_star, desc = TASKS[task_idx]
    print(f"\n{'='*60}")
    print(f"Task T{task_idx+1:02d}: op1={OP_LABELS[op1_idx]}, op2={OP_LABELS[op2_idx]}, "
          f"a*={a_star} — {desc}")
    print(f"{'='*60}")

    graph = _compile_template()

    dmci_results = []
    random_results = []
    exhaustive_results = []
    evolutionary_results = []

    for restart in range(cfg.n_restarts):
        data_seed = cfg.seed_offset + task_idx * 1000 + restart
        xs, ys, dys = generate_data(op1_idx, op2_idx, a_star, cfg, data_seed)

        # DMCI soft-dispatch
        dmci_seed = cfg.seed_offset + task_idx * 10000 + restart
        res = run_dmci_restart(graph, xs, ys, dys, cfg, dmci_seed)
        res["restart"] = restart
        res["op1_correct"] = res["op1_selected"] == op1_idx
        res["op2_correct"] = res["op2_selected"] == op2_idx
        res["both_correct"] = res["op1_correct"] and res["op2_correct"]
        res["const_error"] = abs(res["alpha"] - a_star)
        dmci_results.append(res)

        status = "✓" if res["both_correct"] else "✗"
        print(
            f"  DMCI restart {restart:2d}: "
            f"op1={res['op1_label']:>3s} op2={res['op2_label']:>3s} "
            f"a={res['alpha']:.3f} loss={res['final_loss']:.2e} "
            f"t={res['wall_time_s']:.1f}s {status}"
        )

        # Random search
        rng_seed = cfg.seed_offset + task_idx * 20000 + restart
        rres = run_random_restart(xs, ys, op1_idx, op2_idx, a_star, cfg, rng_seed)
        rres["restart"] = restart
        rres["both_correct"] = rres["op1_correct"] and rres["op2_correct"]
        rres["const_error"] = abs(rres["alpha"] - a_star)
        random_results.append(rres)

        # Exhaustive enumeration
        exh_seed = cfg.seed_offset + task_idx * 30000 + restart
        eres = run_exhaustive_restart(xs, ys, op1_idx, op2_idx, a_star, cfg, exh_seed)
        eres["restart"] = restart
        eres["both_correct"] = eres["op1_correct"] and eres["op2_correct"]
        eres["const_error"] = abs(eres["alpha"] - a_star)
        exhaustive_results.append(eres)

        # Evolutionary algorithm
        evo_seed = cfg.seed_offset + task_idx * 40000 + restart
        evres = run_evolutionary_restart(xs, ys, op1_idx, op2_idx, a_star, cfg, evo_seed)
        evres["restart"] = restart
        evres["both_correct"] = evres["op1_correct"] and evres["op2_correct"]
        evres["const_error"] = abs(evres["alpha"] - a_star)
        evolutionary_results.append(evres)

    dmci_success = sum(1 for r in dmci_results if r["both_correct"])
    random_success = sum(1 for r in random_results if r["both_correct"])
    exh_success = sum(1 for r in exhaustive_results if r["both_correct"])
    evo_success = sum(1 for r in evolutionary_results if r["both_correct"])

    def _mean_err(results):
        errs = [r["const_error"] for r in results if r["both_correct"]]
        return sum(errs) / len(errs) if errs else float("nan")

    dmci_mean_time = sum(r["wall_time_s"] for r in dmci_results) / len(dmci_results)
    evo_mean_time = sum(r["wall_time_s"] for r in evolutionary_results) / len(evolutionary_results)

    summary = {
        "task_idx": task_idx,
        "task_id": f"T{task_idx+1:02d}",
        "op1_true": op1_idx,
        "op1_label": OP_LABELS[op1_idx],
        "op2_true": op2_idx,
        "op2_label": OP_LABELS[op2_idx],
        "a_star": a_star,
        "description": desc,
        "n_restarts": cfg.n_restarts,
        "dmci_success_rate": dmci_success / cfg.n_restarts,
        "dmci_success_count": dmci_success,
        "dmci_mean_const_error": _mean_err(dmci_results),
        "dmci_mean_wall_time_s": dmci_mean_time,
        "exhaustive_success_rate": exh_success / cfg.n_restarts,
        "exhaustive_success_count": exh_success,
        "exhaustive_mean_const_error": _mean_err(exhaustive_results),
        "evolutionary_success_rate": evo_success / cfg.n_restarts,
        "evolutionary_success_count": evo_success,
        "evolutionary_mean_const_error": _mean_err(evolutionary_results),
        "evolutionary_mean_wall_time_s": evo_mean_time,
        "random_success_rate": random_success / cfg.n_restarts,
        "random_success_count": random_success,
        "random_mean_const_error": _mean_err(random_results),
    }

    print(f"\n  Summary:")
    for name, cnt in [("DMCI", dmci_success), ("Exhaustive", exh_success),
                       ("Evolutionary", evo_success), ("Random", random_success)]:
        pct = 100 * cnt / cfg.n_restarts
        print(f"    {name:<14s} {cnt}/{cfg.n_restarts} ({pct:.0f}%)")

    dmci_cerr = _mean_err(dmci_results)
    if not math.isnan(dmci_cerr):
        print(f"  DMCI mean |a-a*| on correct: {dmci_cerr:.4f}")
    print(f"  Mean wall-clock per DMCI restart: {dmci_mean_time:.1f}s")

    task_result = {
        "summary": summary,
        "config": asdict(cfg),
        "dmci_restarts": dmci_results,
        "random_restarts": random_results,
        "exhaustive_restarts": exhaustive_results,
        "evolutionary_restarts": evolutionary_results,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"exp_e1_T{task_idx+1:02d}.json"
    with open(out_path, "w") as f:
        json.dump(task_result, f, indent=2, default=str)
    print(f"  Saved: {out_path}")

    return task_result


# ---------------------------------------------------------------------------
# Aggregate results across all tasks
# ---------------------------------------------------------------------------

def aggregate(output_dir: Path) -> None:
    """Read all per-task results and print an aggregate table."""
    summaries = []
    for i in range(12):
        path = output_dir / f"exp_e1_T{i+1:02d}.json"
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            summaries.append(data["summary"])

    if not summaries:
        print("No results found.")
        return

    has_exh = "exhaustive_success_count" in summaries[0]

    print(f"\n{'='*100}")
    print("Experiment E.1 Aggregate Results — Operator Recovery")
    print(f"{'='*100}")
    hdr = f"{'Task':<6} {'Target':<16} {'a*':>4}  {'DMCI':>8}"
    if has_exh:
        hdr += f"  {'Exhaust':>8}  {'Evol':>8}"
    hdr += f"  {'Random':>8}  {'DMCI|a-a*|':>10}  {'Time(s)':>8}"
    print(hdr)
    print("-" * 100)

    totals = {"dmci": 0, "exhaustive": 0, "evolutionary": 0, "random": 0}
    all_errs = {"dmci": [], "exhaustive": [], "evolutionary": [], "random": []}
    total_restarts = 0

    for s in summaries:
        n = s["n_restarts"]
        total_restarts += n

        d_cnt = s["dmci_success_count"]
        r_cnt = s["random_success_count"]
        totals["dmci"] += d_cnt
        totals["random"] += r_cnt

        d_str = f"{d_cnt}/{n}"
        r_str = f"{r_cnt}/{n}"
        cerr = s.get("dmci_mean_const_error", float("nan"))
        cerr_str = f"{cerr:.4f}" if not math.isnan(cerr) else "—"

        row = f"{s['task_id']:<6} {s['description']:<16} {s['a_star']:>4.1f}  {d_str:>8}"

        if has_exh:
            e_cnt = s.get("exhaustive_success_count", 0)
            v_cnt = s.get("evolutionary_success_count", 0)
            totals["exhaustive"] += e_cnt
            totals["evolutionary"] += v_cnt
            row += f"  {e_cnt}/{n:>5}  {v_cnt}/{n:>5}"

        row += f"  {r_str:>8}  {cerr_str:>10}  {s['dmci_mean_wall_time_s']:>7.1f}"
        print(row)

        for key in ["dmci", "exhaustive", "evolutionary", "random"]:
            v = s.get(f"{key}_mean_const_error", float("nan"))
            if not math.isnan(v):
                all_errs[key].append(v)

    print("-" * 100)

    tot_row = f"{'Total':<6} {'':16} {'':>4}  {totals['dmci']}/{total_restarts:>5}"
    if has_exh:
        tot_row += f"  {totals['exhaustive']}/{total_restarts:>5}  {totals['evolutionary']}/{total_restarts:>5}"
    tot_row += f"  {totals['random']}/{total_restarts:>5}"
    print(tot_row)

    rate_row = f"{'Rate':<6} {'':16} {'':>4}  {100*totals['dmci']/total_restarts:>7.1f}%"
    if has_exh:
        rate_row += f"  {100*totals['exhaustive']/total_restarts:>7.1f}%  {100*totals['evolutionary']/total_restarts:>7.1f}%"
    rate_row += f"  {100*totals['random']/total_restarts:>7.1f}%"
    print(rate_row)

    print(f"\n  Continuous Parameter Recovery (mean |a-a*| on correct ops):")
    for name, key in [("DMCI", "dmci"), ("Exhaustive", "exhaustive"),
                       ("Evolutionary", "evolutionary"), ("Random", "random")]:
        errs = all_errs[key]
        if errs:
            print(f"    {name:<14s} {sum(errs)/len(errs):.6f}  ({len(errs)} tasks)")
        else:
            print(f"    {name:<14s} —")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Experiment E.1: Operator Recovery")
    parser.add_argument("--task", default="all",
                        help="Task index (0-11) or 'all'")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("experiments/exp_e/results"))
    parser.add_argument("--n-restarts", type=int, default=None)
    parser.add_argument("--n-epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--aggregate-only", action="store_true",
                        help="Just print aggregate table from existing results")
    args = parser.parse_args()

    if args.aggregate_only:
        aggregate(args.output_dir)
        return

    cfg = ExpE1Config()
    if args.n_restarts is not None:
        cfg.n_restarts = args.n_restarts
    if args.n_epochs is not None:
        cfg.n_epochs = args.n_epochs
    if args.lr is not None:
        cfg.lr = args.lr

    if args.task == "all":
        task_indices = list(range(12))
    else:
        task_indices = [int(args.task)]

    t0 = time.perf_counter()
    for idx in task_indices:
        run_task(idx, cfg, args.output_dir)
    total = time.perf_counter() - t0

    print(f"\nTotal wall-clock: {total:.1f}s")

    if args.task == "all":
        aggregate(args.output_dir)


if __name__ == "__main__":
    main()
