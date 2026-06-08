############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# exp_d.py: Experiment D: GP + DMCI vs GP + direct-recompile crossover study. Measures per-candidate compile time and train...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Experiment D: GP + DMCI vs GP + direct-recompile crossover study.

Measures per-candidate compile time and train time for both methods
across a GP-driven symbolic regression task.

Target function: f(x) = a*sin(b*x) + c*x^2

Usage:
    python3 -m experiments.exp_d.exp_d --method gp_direct --seed 0
    python3 -m experiments.exp_d.exp_d --method gp_dmci --seed 0
    python3 -m experiments.exp_d.exp_d --method both --seed 0
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import torch.nn as nn

from neural_compiler.compiler import compile_program
from neural_compiler.evaluator import evaluate
from neural_compiler.runtime.tagged_value import make_float, unwrap_number

from .config import ExpDConfig, DEFAULT
from .gp import (
    GPNode, ConstCounter, collect_consts, to_scheme,
    make_initial_population, subtree_mutation, subtree_crossover,
    tournament_select, random_tree,
)


EVALUATOR_SOURCE = (
    Path(__file__).parent.parent.parent / "bootstrap" / "compiler.scm"
).read_text()


@dataclass
class CandidateResult:
    generation: int
    candidate_id: int
    scheme_source: str
    n_consts: int
    tree_size: int
    tree_depth: int
    t_compile: float
    t_train: float
    t_total: float
    final_loss: float
    const_values: dict[str, float]


@dataclass
class GPRunResult:
    method: str
    seed: int
    config: dict
    candidates: list[dict]
    total_wall_time: float
    best_loss: float
    best_source: str
    n_candidates_evaluated: int


def _target_fn(x: float, a: float = 2.0, b: float = 3.0, c: float = 0.5) -> float:
    return a * math.sin(b * x) + c * x * x


def _generate_data(cfg: ExpDConfig):
    xs = torch.linspace(cfg.x_range[0], cfg.x_range[1], cfg.n_data_points)
    ys = torch.tensor([
        _target_fn(x.item(), cfg.target_a, cfg.target_b, cfg.target_c)
        for x in xs
    ])
    return xs, ys


def _make_direct_source(tree: GPNode) -> str:
    """Convert GP tree to a directly compilable Scheme program."""
    consts = collect_consts(tree)
    expr = to_scheme(tree)
    if not consts:
        return expr
    bindings = " ".join(f"({c} {c})" for c in consts)
    return f"(let ({bindings}) {expr})"


def _make_dmci_source(tree: GPNode) -> str:
    """Convert GP tree to an evaluator+scheme-eval source."""
    consts = collect_consts(tree)
    expr = to_scheme(tree)
    all_names = ["x"] + consts
    env_pairs = " ".join(f"(cons '{n} {n})" for n in all_names)
    return (
        EVALUATOR_SOURCE
        + f"\n(scheme-eval '{expr} (list {env_pairs}))\n"
    )


def _input_names(tree: GPNode) -> dict[str, None]:
    consts = collect_consts(tree)
    return {n: None for n in ["x"] + consts}


def _fit_constants(graph, tree: GPNode, xs, ys, cfg: ExpDConfig,
                   seed: int) -> tuple[float, dict[str, float], float]:
    """Run inner constant-fitting loop. Returns (final_loss, const_values, wall_time)."""
    consts = collect_consts(tree)
    if not consts:
        total_loss = torch.tensor(0.0)
        for x_val, y_val in zip(xs, ys):
            inp = {"x": make_float(torch.tensor(x_val.item()))}
            result = evaluate(graph, inp)
            pred = unwrap_number(result)
            total_loss = total_loss + (pred - y_val) ** 2
        return total_loss.item(), {}, 0.0

    torch.manual_seed(seed)
    params = {
        name: nn.Parameter(torch.tensor(1.0) + 0.5 * torch.randn(1).squeeze())
        for name in consts
    }
    optimizer = torch.optim.Adam(list(params.values()), lr=cfg.inner_lr)

    t0 = time.perf_counter()
    final_loss = float("inf")
    for epoch in range(cfg.inner_epochs):
        total_loss = torch.tensor(0.0)
        for x_val, y_val in zip(xs, ys):
            inp = {"x": make_float(torch.tensor(x_val.item()))}
            for n, p in params.items():
                inp[n] = make_float(p)
            try:
                result = evaluate(graph, inp)
                pred = unwrap_number(result)
                total_loss = total_loss + (pred - y_val) ** 2
            except Exception:
                total_loss = total_loss + torch.tensor(1e6)

        optimizer.zero_grad()
        try:
            total_loss.backward()
            optimizer.step()
        except Exception:
            pass
        final_loss = total_loss.item()

    wall_time = time.perf_counter() - t0
    const_values = {n: p.item() for n, p in params.items()}
    return final_loss, const_values, wall_time


def evaluate_candidate(tree: GPNode, method: str, xs, ys, cfg: ExpDConfig,
                       seed: int, generation: int, candidate_id: int,
                       dmci_graph_cache: dict | None = None,
                       ) -> CandidateResult:
    """Evaluate a single GP candidate with either method."""
    consts = collect_consts(tree)
    inputs = _input_names(tree)

    if method == "gp_direct":
        source = _make_direct_source(tree)
        t0 = time.perf_counter()
        try:
            graph = compile_program(source, inputs=inputs, prelude=True)
        except Exception:
            return CandidateResult(
                generation=generation, candidate_id=candidate_id,
                scheme_source=to_scheme(tree), n_consts=len(consts),
                tree_size=tree.size(), tree_depth=tree.depth(),
                t_compile=0.0, t_train=0.0, t_total=0.0,
                final_loss=1e10, const_values={},
            )
        t_compile = time.perf_counter() - t0

    elif method == "gp_dmci":
        source = _make_dmci_source(tree)
        t0 = time.perf_counter()
        try:
            graph = compile_program(source, inputs=inputs, prelude=True)
        except Exception:
            return CandidateResult(
                generation=generation, candidate_id=candidate_id,
                scheme_source=to_scheme(tree), n_consts=len(consts),
                tree_size=tree.size(), tree_depth=tree.depth(),
                t_compile=0.0, t_train=0.0, t_total=0.0,
                final_loss=1e10, const_values={},
            )
        t_compile = time.perf_counter() - t0

    else:
        raise ValueError(f"Unknown method: {method}")

    final_loss, const_values, t_train = _fit_constants(
        graph, tree, xs, ys, cfg, seed + candidate_id)

    return CandidateResult(
        generation=generation, candidate_id=candidate_id,
        scheme_source=to_scheme(tree), n_consts=len(consts),
        tree_size=tree.size(), tree_depth=tree.depth(),
        t_compile=t_compile, t_train=t_train,
        t_total=t_compile + t_train,
        final_loss=final_loss, const_values=const_values,
    )


def run_gp(method: str, seed: int, cfg: ExpDConfig) -> GPRunResult:
    """Run the full GP search with one method."""
    random.seed(seed)
    torch.manual_seed(seed)

    xs, ys = _generate_data(cfg)
    population = make_initial_population(cfg.pop_size, cfg.min_tree_depth,
                                         cfg.max_tree_depth)

    all_candidates: list[CandidateResult] = []
    candidate_counter = 0
    t_start = time.perf_counter()

    best_loss = float("inf")
    best_source = ""

    for gen in range(cfg.n_generations):
        gen_results: list[tuple[GPNode, float]] = []

        for i, tree in enumerate(population):
            result = evaluate_candidate(
                tree, method, xs, ys, cfg, seed,
                generation=gen, candidate_id=candidate_counter,
            )
            all_candidates.append(result)
            gen_results.append((tree, result.final_loss))
            candidate_counter += 1

            if result.final_loss < best_loss:
                best_loss = result.final_loss
                best_source = result.scheme_source

        gen_losses = [loss for _, loss in gen_results]
        avg_loss = sum(gen_losses) / len(gen_losses)
        min_loss = min(gen_losses)
        print(f"  Gen {gen:3d}: avg_loss={avg_loss:.4f} "
              f"min_loss={min_loss:.4f} best_ever={best_loss:.4f}",
              file=sys.stderr)

        new_pop = []
        elite = min(gen_results, key=lambda x: x[1])[0]
        new_pop.append(elite.copy())

        while len(new_pop) < cfg.pop_size:
            if random.random() < cfg.crossover_rate:
                p1 = tournament_select(gen_results, cfg.tournament_size)
                p2 = tournament_select(gen_results, cfg.tournament_size)
                child = subtree_crossover(p1, p2)
            else:
                parent = tournament_select(gen_results, cfg.tournament_size)
                child = subtree_mutation(parent, cfg.max_tree_depth)

            if child.depth() > cfg.max_tree_depth:
                counter = ConstCounter()
                child = random_tree(cfg.max_tree_depth, counter)
            new_pop.append(child)

        population = new_pop

    total_wall_time = time.perf_counter() - t_start

    return GPRunResult(
        method=method, seed=seed,
        config=asdict(cfg),
        candidates=[asdict(c) for c in all_candidates],
        total_wall_time=total_wall_time,
        best_loss=best_loss,
        best_source=best_source,
        n_candidates_evaluated=candidate_counter,
    )


def save_result(result: GPRunResult, output_dir: Path):
    """Save results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{result.method}_seed{result.seed:02d}"
    path = output_dir / f"{tag}.json"
    with open(path, "w") as f:
        json.dump(asdict(result) if hasattr(result, '__dataclass_fields__')
                  else {
                      "method": result.method,
                      "seed": result.seed,
                      "config": result.config,
                      "candidates": result.candidates,
                      "total_wall_time": result.total_wall_time,
                      "best_loss": result.best_loss,
                      "best_source": result.best_source,
                      "n_candidates_evaluated": result.n_candidates_evaluated,
                  },
                  f, indent=2, default=str)
    print(f"Results saved to {path}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Experiment D: GP crossover study")
    parser.add_argument("--method", type=str, default="both",
                        choices=["gp_direct", "gp_dmci", "both"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("experiments/exp_d/results"))
    parser.add_argument("--pop-size", type=int, default=DEFAULT.pop_size)
    parser.add_argument("--generations", type=int, default=DEFAULT.n_generations)
    parser.add_argument("--inner-epochs", type=int, default=DEFAULT.inner_epochs)

    args = parser.parse_args()

    cfg = ExpDConfig(
        pop_size=args.pop_size,
        n_generations=args.generations,
        inner_epochs=args.inner_epochs,
    )

    methods = [args.method] if args.method != "both" else ["gp_direct", "gp_dmci"]

    for method in methods:
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"Experiment D: {method}, seed={args.seed}", file=sys.stderr)
        print(f"  pop={cfg.pop_size}, gens={cfg.n_generations}, "
              f"inner_epochs={cfg.inner_epochs}", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)

        result = run_gp(method, args.seed, cfg)

        # Summary
        compile_times = [c["t_compile"] for c in result.candidates]
        train_times = [c["t_train"] for c in result.candidates]
        total_compile = sum(compile_times)
        total_train = sum(train_times)
        avg_compile = total_compile / len(compile_times) if compile_times else 0
        avg_train = total_train / len(train_times) if train_times else 0

        print(f"\n--- {method} Summary ---", file=sys.stderr)
        print(f"  Candidates evaluated: {result.n_candidates_evaluated}",
              file=sys.stderr)
        print(f"  Total wall time: {result.total_wall_time:.1f}s", file=sys.stderr)
        print(f"  Total compile time: {total_compile:.3f}s "
              f"(avg {avg_compile*1000:.1f}ms/candidate)", file=sys.stderr)
        print(f"  Total train time: {total_train:.1f}s "
              f"(avg {avg_train:.3f}s/candidate)", file=sys.stderr)
        print(f"  Best loss: {result.best_loss:.6f}", file=sys.stderr)
        print(f"  Best program: {result.best_source}", file=sys.stderr)

        save_result(result, args.output_dir)


if __name__ == "__main__":
    main()
