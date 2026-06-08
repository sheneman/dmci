############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# ablations.py: Ablation studies for Experiment A. 1. Dict-backed vs tensor-backed heap (gradient preservation) 2. Lazy vs...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Ablation studies for Experiment A.

1. Dict-backed vs tensor-backed heap (gradient preservation)
2. Lazy vs eager evaluation (recursive program handling)
3. Gradient path length analysis (autograd graph size per program)
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from neural_compiler.compiler import compile_program
from neural_compiler.evaluator import evaluate
from neural_compiler.runtime.tagged_value import make_float, unwrap_number, VALUE_DIM
from neural_compiler.runtime.heap import TensorHeap

from .config import ExpAConfig, DEFAULT
from .programs import ProgramSpec, P3, ALL_PROGRAMS, _all_input_names
from .baselines import (
    _get_graph, _make_params, _build_tagged_inputs, _generate_data,
    _grad_norm, _param_errors, _record, TrainResult,
)


# ---------------------------------------------------------------------------
# Ablation 1: Dict-backed vs tensor-backed heap
# ---------------------------------------------------------------------------

class OldTensorHeap:
    """Original tensor-backed heap (pre-allocated storage tensor).

    Reproduced from git history. In-place tensor mutations break autograd
    version tracking across many cons calls.
    """

    def __init__(self, max_size: int = 65536,
                 device: torch.device | None = None):
        self.max_size = max_size
        self.device = device
        self.storage = torch.zeros(max_size, VALUE_DIM, dtype=torch.float32,
                                   device=device)
        self.alloc_ptr = 0

    def reset(self) -> None:
        self.storage = torch.zeros(self.max_size, VALUE_DIM, dtype=torch.float32,
                                   device=self.device)
        self.alloc_ptr = 0

    def cons(self, car, cdr):
        from neural_compiler.runtime.tagged_value import make_pair
        if self.alloc_ptr + 2 > self.max_size:
            raise RuntimeError("Heap overflow")
        car_addr = self.alloc_ptr
        cdr_addr = self.alloc_ptr + 1
        self.alloc_ptr += 2
        self.storage = self.storage.clone()
        self.storage[car_addr] = car
        self.storage[cdr_addr] = cdr
        return make_pair(float(car_addr), float(cdr_addr), device=self.device)

    def read(self, addr):
        idx = int(addr.item()) if isinstance(addr, torch.Tensor) else int(addr)
        if idx < 0 or idx >= self.alloc_ptr:
            raise IndexError(f"Heap read OOB: {idx}")
        return self.storage[idx]

    def write(self, addr, val):
        idx = int(addr.item()) if isinstance(addr, torch.Tensor) else int(addr)
        self.storage = self.storage.clone()
        self.storage[idx] = val

    def car(self, pair_val):
        from neural_compiler.runtime.tagged_value import extract_payload
        payload = extract_payload(pair_val)
        return self.read(payload[0])

    def cdr(self, pair_val):
        from neural_compiler.runtime.tagged_value import extract_payload
        payload = extract_payload(pair_val)
        return self.read(payload[1])

    def to(self, device):
        self.device = device
        self.storage = self.storage.to(device)
        return self

    def allocated(self):
        return self.alloc_ptr


@dataclass
class HeapAblationResult:
    heap_type: str
    converged: bool
    convergence_epoch: int | None
    final_loss: float
    error_message: str | None
    loss_history: list[float]


def ablation_heap(spec: ProgramSpec = P3, cfg: ExpAConfig = DEFAULT,
                  seed: int = 0) -> dict[str, HeapAblationResult]:
    """Compare dict-backed heap (working) vs tensor-backed heap (broken)."""
    results = {}

    # Dict-backed (current implementation) — should work
    r = _run_with_heap_type("dict", spec, cfg, seed)
    results["dict"] = r

    # Tensor-backed (old implementation) — should fail or produce bad gradients
    r = _run_with_heap_type("tensor", spec, cfg, seed)
    results["tensor"] = r

    return results


def _run_with_heap_type(heap_type: str, spec: ProgramSpec,
                        cfg: ExpAConfig, seed: int) -> HeapAblationResult:
    import neural_compiler.runtime.heap as heap_mod
    original_class = heap_mod.TensorHeap

    try:
        if heap_type == "tensor":
            heap_mod.TensorHeap = OldTensorHeap

        graph = compile_program(
            spec.interp_source,
            inputs={n: None for n in _all_input_names(spec)},
            prelude=True,
        )
        params = _make_params(spec, seed)
        optimizer = torch.optim.Adam(list(params.values()), lr=cfg.lr)
        xs, ys = _generate_data(spec, cfg)

        loss_history = []
        conv_epoch = None
        max_ep = min(cfg.max_epochs, 500)  # cap for ablation

        for epoch in range(max_ep):
            total_loss = torch.tensor(0.0)
            for x_val, y_val in zip(xs, ys):
                inputs = _build_tagged_inputs(spec, params, x_val)
                result = evaluate(graph, inputs)
                pred = unwrap_number(result)
                total_loss = total_loss + (pred - y_val) ** 2

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            loss_val = total_loss.item()
            loss_history.append(loss_val)
            if conv_epoch is None and loss_val < cfg.convergence_threshold:
                conv_epoch = epoch

        return HeapAblationResult(
            heap_type=heap_type, converged=conv_epoch is not None,
            convergence_epoch=conv_epoch, final_loss=loss_history[-1],
            error_message=None, loss_history=loss_history,
        )

    except Exception as e:
        return HeapAblationResult(
            heap_type=heap_type, converged=False,
            convergence_epoch=None, final_loss=float("inf"),
            error_message=str(e), loss_history=[],
        )
    finally:
        heap_mod.TensorHeap = original_class


# ---------------------------------------------------------------------------
# Ablation 2: Lazy vs eager evaluation
# ---------------------------------------------------------------------------

@dataclass
class EvalAblationResult:
    eval_mode: str
    program: str
    success: bool
    error_message: str | None
    final_loss: float


def ablation_eval_mode(programs: list[ProgramSpec] | None = None,
                       cfg: ExpAConfig = DEFAULT,
                       seed: int = 0) -> list[EvalAblationResult]:
    """Test lazy (default) vs eager evaluation on each program."""
    from neural_compiler.evaluator.engine import _evaluate_tagged
    if programs is None:
        programs = ALL_PROGRAMS

    results = []
    for spec in programs:
        for mode in ["lazy", "eager"]:
            try:
                graph = compile_program(
                    spec.interp_source,
                    inputs={n: None for n in _all_input_names(spec)},
                    prelude=True,
                )
                params = _make_params(spec, seed)
                xs, ys = _generate_data(spec, cfg)

                x_val = xs[0]
                inputs = _build_tagged_inputs(spec, params, x_val)

                if mode == "eager":
                    result = evaluate(graph, inputs, max_depth=0)
                else:
                    result = evaluate(graph, inputs)

                pred = unwrap_number(result)
                loss = (pred - ys[0]) ** 2
                results.append(EvalAblationResult(
                    eval_mode=mode, program=spec.name,
                    success=True, error_message=None,
                    final_loss=loss.item(),
                ))
            except Exception as e:
                results.append(EvalAblationResult(
                    eval_mode=mode, program=spec.name,
                    success=False, error_message=str(e),
                    final_loss=float("inf"),
                ))
    return results


# ---------------------------------------------------------------------------
# Ablation 3: Gradient path length
# ---------------------------------------------------------------------------

@dataclass
class GradPathResult:
    program: str
    method: str
    autograd_nodes: int
    graph_nodes: int
    graph_depth: int


def ablation_grad_path(programs: list[ProgramSpec] | None = None,
                       cfg: ExpAConfig = DEFAULT) -> list[GradPathResult]:
    """Count autograd graph nodes between parameter and loss for each program."""
    if programs is None:
        programs = ALL_PROGRAMS

    results = []
    for spec in programs:
        for method, source_attr in [("direct", "direct_source"),
                                     ("compiled_interp", "interp_source")]:
            source = getattr(spec, source_attr)
            graph = compile_program(
                source,
                inputs={n: None for n in _all_input_names(spec)},
                prelude=True,
            )

            params = _make_params(spec, seed=0)
            xs, ys = _generate_data(spec, cfg)

            total_loss = torch.tensor(0.0)
            for x_val, y_val in zip(xs[:1], ys[:1]):  # single data point
                inputs = _build_tagged_inputs(spec, params, x_val)
                result = evaluate(graph, inputs)
                pred = unwrap_number(result)
                total_loss = total_loss + (pred - y_val) ** 2

            # Walk autograd graph
            n_nodes = _count_autograd_nodes(total_loss)

            results.append(GradPathResult(
                program=spec.name, method=method,
                autograd_nodes=n_nodes,
                graph_nodes=len(graph.nodes),
                graph_depth=graph.depth(),
            ))

    return results


def _count_autograd_nodes(tensor: torch.Tensor) -> int:
    visited = set()

    def walk(grad_fn):
        if grad_fn is None or id(grad_fn) in visited:
            return
        visited.add(id(grad_fn))
        for child, _ in grad_fn.next_functions:
            walk(child)

    walk(tensor.grad_fn)
    return len(visited)


# ---------------------------------------------------------------------------
# Save all ablation results
# ---------------------------------------------------------------------------

def run_all_ablations(cfg: ExpAConfig = DEFAULT,
                      output_dir: str | None = None) -> dict:
    if output_dir is None:
        output_dir = str(Path(cfg.output_dir) / "ablations")
    os.makedirs(output_dir, exist_ok=True)

    all_results = {}

    # Heap ablation
    print("Running heap ablation (P3)...")
    heap_res = ablation_heap(P3, cfg, seed=0)
    all_results["heap"] = {k: {
        "heap_type": v.heap_type,
        "converged": v.converged,
        "convergence_epoch": v.convergence_epoch,
        "final_loss": v.final_loss,
        "error_message": v.error_message,
        "n_epochs": len(v.loss_history),
    } for k, v in heap_res.items()}

    # Eval mode ablation
    print("Running eval mode ablation...")
    eval_res = ablation_eval_mode(cfg=cfg, seed=0)
    all_results["eval_mode"] = [{
        "eval_mode": r.eval_mode,
        "program": r.program,
        "success": r.success,
        "error_message": r.error_message,
    } for r in eval_res]

    # Gradient path length
    print("Running gradient path analysis...")
    grad_res = ablation_grad_path(cfg=cfg)
    all_results["grad_path"] = [{
        "program": r.program,
        "method": r.method,
        "autograd_nodes": r.autograd_nodes,
        "graph_nodes": r.graph_nodes,
        "graph_depth": r.graph_depth,
    } for r in grad_res]

    with open(os.path.join(output_dir, "ablations.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"Ablation results saved to {output_dir}/ablations.json")
    return all_results
