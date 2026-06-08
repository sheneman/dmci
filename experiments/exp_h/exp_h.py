############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# exp_h.py: Experiment H: Batched GPU Parallelization of DMCI. Demonstrates that DMCI's interpretation overhead is...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Experiment H: Batched GPU Parallelization of DMCI.

Demonstrates that DMCI's interpretation overhead is amortized at scale:
  Part A — Forward throughput scaling with batch size
  Part B — Training epoch speedup (batched vs sequential)
  Part C — Correctness verification (batched = sequential, same gradients)
  Part D — Population batching (M random restarts × N data points)
  Part E — torch.compile speedup over plain batched evaluation

Usage:
  python -m experiments.exp_h.exp_h [--part A|B|C|D|E|all] [--device cpu|cuda]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.exp_b.models import ALL_MODELS, MODEL_BY_NAME, ModelSpec
from experiments.exp_h.config import DEFAULT, ExpHConfig
from neural_compiler.compiler import compile_program
from neural_compiler.evaluator import evaluate, evaluate_batched, compile_batched
from neural_compiler.runtime.tagged_value import make_float, unwrap_number, VALUE_DIM


OUTPUT_DIR = Path(__file__).parent / "results"


def _compile_direct(spec: ModelSpec):
    return compile_program(spec.direct_source, inputs=spec.input_names + spec.param_names)


def _generate_data(spec: ModelSpec, n: int, device: torch.device):
    if len(spec.input_names) == 1:
        xs_raw = torch.linspace(spec.x_range[0], spec.x_range[1], n)
        ys = torch.tensor([spec.ground_truth(x.item()) for x in xs_raw],
                          dtype=torch.float32, device=device)
        xs = {spec.input_names[0]: xs_raw.to(device)}
    else:
        xs = {}
        for name in spec.input_names:
            xs[name] = torch.linspace(spec.x_range[0], spec.x_range[1], n).to(device)
        call_args = [xs[name].tolist() for name in spec.input_names]
        ys = torch.tensor(
            [spec.ground_truth(*args) for args in zip(*call_args)],
            dtype=torch.float32, device=device,
        )
    return xs, ys


def _time_fn(fn, warmup: int, iters: int):
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return elapsed / iters


# ===================================================================
# Part A: Forward Throughput
# ===================================================================

def run_part_a(cfg: ExpHConfig, device: torch.device):
    print("\n" + "=" * 60)
    print("Part A: Forward Throughput Scaling")
    print("=" * 60)

    results = []
    for model_name in cfg.batchable_models:
        spec = MODEL_BY_NAME[model_name]
        graph = _compile_direct(spec)

        for bs in cfg.batch_sizes:
            xs_data, _ = _generate_data(spec, bs, device)
            params = {
                name: torch.tensor(spec.target_values.get(name, spec.init_values.get(name, 1.0)),
                                   dtype=torch.float32, device=device)
                for name in spec.param_names
            }
            batch_inputs = {**xs_data, **params}

            def forward_batched():
                return evaluate_batched(graph, batch_inputs)

            # Sequential baseline
            def forward_sequential():
                for i in range(bs):
                    single = {name: v[i:i+1] if v.dim() > 0 and v.shape[0] >= bs else v
                              for name, v in batch_inputs.items()}
                    evaluate(graph, {
                        name: (v[i].item() if v.dim() > 0 and v.shape[0] >= bs else v.item())
                        for name, v in batch_inputs.items()
                    })

            t_batch = _time_fn(forward_batched, cfg.warmup_iters, cfg.timing_iters)
            t_seq = _time_fn(forward_sequential, min(cfg.warmup_iters, 2),
                             min(cfg.timing_iters, 5)) if bs <= 512 else float('nan')

            throughput_batch = bs / t_batch
            throughput_seq = bs / t_seq if not math.isnan(t_seq) else float('nan')
            speedup = throughput_batch / throughput_seq if not math.isnan(throughput_seq) else float('nan')

            row = {
                "model": model_name, "batch_size": bs, "device": str(device),
                "time_batched_s": t_batch, "time_sequential_s": t_seq,
                "throughput_batched": throughput_batch,
                "throughput_sequential": throughput_seq,
                "speedup": speedup,
            }
            results.append(row)
            sp_str = f"{speedup:.1f}x" if not math.isnan(speedup) else "N/A"
            seq_str = f"{t_seq:.4f}s" if not math.isnan(t_seq) else "N/A"
            print(f"  {model_name:30s} bs={bs:5d}  batch={t_batch:.4f}s  "
                  f"seq={seq_str:>9s}  speedup={sp_str}")

    return results


# ===================================================================
# Part B: Training Epoch Speedup
# ===================================================================

def _train_sequential(graph, spec, xs_data, ys, device, epochs, lr):
    """Train with one evaluate_batched call per data point (simulates sequential)."""
    params = {
        name: torch.tensor(spec.init_values[name], dtype=torch.float32,
                           device=device, requires_grad=True)
        for name in spec.param_names
    }
    opt = torch.optim.Adam(list(params.values()), lr=lr)
    n = ys.shape[0]

    t0 = time.perf_counter()
    final_loss = float('nan')
    for epoch in range(epochs):
        opt.zero_grad()
        preds = []
        for i in range(n):
            inp = {name: (v[i:i+1] if v.dim() > 0 and v.shape[0] == n else v)
                   for name, v in xs_data.items()}
            inp.update({name: p for name, p in params.items()})
            r = evaluate_batched(graph, inp)
            preds.append(r.squeeze(0) if r.dim() > 0 else r)
        pred_tensor = torch.stack(preds)
        loss = (pred_tensor - ys).pow(2).mean()
        loss.backward()
        opt.step()
        final_loss = loss.item()
    elapsed = time.perf_counter() - t0
    return elapsed, final_loss, {name: p.item() for name, p in params.items()}


def _train_batched(graph, spec, xs_data, ys, device, epochs, lr):
    params = {
        name: torch.tensor(spec.init_values[name], dtype=torch.float32,
                           device=device, requires_grad=True)
        for name in spec.param_names
    }
    opt = torch.optim.Adam(list(params.values()), lr=lr)

    t0 = time.perf_counter()
    final_loss = float('nan')
    for epoch in range(epochs):
        opt.zero_grad()
        batch_inputs = {**xs_data, **params}
        result = evaluate_batched(graph, batch_inputs)
        loss = (result - ys).pow(2).mean()
        loss.backward()
        opt.step()
        final_loss = loss.item()
    elapsed = time.perf_counter() - t0
    return elapsed, final_loss, {name: p.item() for name, p in params.items()}


def run_part_b(cfg: ExpHConfig, device: torch.device):
    print("\n" + "=" * 60)
    print("Part B: Training Epoch Speedup")
    print("=" * 60)

    results = []
    for model_name in cfg.batchable_models:
        spec = MODEL_BY_NAME[model_name]
        graph = _compile_direct(spec)
        xs_data, ys = _generate_data(spec, cfg.n_data_points, device)

        t_seq, loss_seq, params_seq = _train_sequential(
            graph, spec, xs_data, ys, device, cfg.training_epochs, cfg.training_lr)
        t_batch, loss_batch, params_batch = _train_batched(
            graph, spec, xs_data, ys, device, cfg.training_epochs, cfg.training_lr)

        speedup = t_seq / t_batch
        row = {
            "model": model_name, "device": str(device),
            "epochs": cfg.training_epochs,
            "time_sequential_s": t_seq, "time_batched_s": t_batch,
            "speedup": speedup,
            "loss_sequential": loss_seq, "loss_batched": loss_batch,
            "params_sequential": params_seq, "params_batched": params_batch,
        }
        results.append(row)
        print(f"  {model_name:30s}  seq={t_seq:.2f}s  batch={t_batch:.2f}s  "
              f"speedup={speedup:.1f}x  loss_s={loss_seq:.6f}  loss_b={loss_batch:.6f}")

    return results


# ===================================================================
# Part C: Correctness Verification
# ===================================================================

def run_part_c(cfg: ExpHConfig, device: torch.device):
    print("\n" + "=" * 60)
    print("Part C: Correctness Verification")
    print("=" * 60)

    results = []
    for model_name in cfg.batchable_models:
        spec = MODEL_BY_NAME[model_name]
        graph = _compile_direct(spec)
        n = cfg.n_data_points
        xs_data, ys = _generate_data(spec, n, device)

        true_params = {
            name: spec.target_values.get(name, 1.0)
            for name in spec.param_names
        }

        # Sequential forward (scalar, no grad)
        seq_preds = []
        for i in range(n):
            inp = {name: (v[i].item() if v.dim() > 0 else v.item())
                   for name, v in xs_data.items()}
            inp.update(true_params)
            r = evaluate(graph, inp)
            seq_preds.append(r if isinstance(r, float) else r.item())
        seq_tensor = torch.tensor(seq_preds, dtype=torch.float32, device=device)

        # Batched forward + grads
        params_t = {
            name: torch.tensor(v, dtype=torch.float32, device=device, requires_grad=True)
            for name, v in true_params.items()
        }
        batch_inputs = {**xs_data, **params_t}
        batch_result = evaluate_batched(graph, batch_inputs)
        batch_loss = (batch_result - ys).pow(2).mean()
        batch_loss.backward()

        pred_diff = (seq_tensor - batch_result.detach()).abs().max().item()
        grads_finite = all(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in params_t.values()
        )
        loss_near_zero = batch_loss.item() < 1e-6

        passed = pred_diff < 1e-5 and grads_finite and loss_near_zero
        status = "PASS" if passed else "FAIL"

        grad_vals = {name: p.grad.item() for name, p in params_t.items()
                     if p.grad is not None}

        row = {
            "model": model_name, "device": str(device),
            "pred_max_diff": pred_diff,
            "loss_at_true_params": batch_loss.item(),
            "grads_finite": grads_finite,
            "grad_values": grad_vals,
            "passed": passed,
        }
        results.append(row)
        print(f"  {model_name:30s}  pred_diff={pred_diff:.2e}  "
              f"loss={batch_loss.item():.2e}  grads_ok={grads_finite}  [{status}]")

    n_pass = sum(1 for r in results if r["passed"])
    print(f"\n  {n_pass}/{len(results)} models passed correctness check")
    return results


# ===================================================================
# Part D: Population Batching
# ===================================================================

def run_part_d(cfg: ExpHConfig, device: torch.device):
    print("\n" + "=" * 60)
    print("Part D: Population Batching")
    print("=" * 60)

    results = []
    subset = cfg.batchable_models[:5]

    for model_name in subset:
        spec = MODEL_BY_NAME[model_name]
        graph = _compile_direct(spec)
        n = cfg.n_data_points

        for pop_size in cfg.population_sizes:
            xs_data_1d, ys = _generate_data(spec, n, device)

            # Population: M random param inits × N data points
            # Each param: shape (M, 1) to broadcast with x shape (N,)
            pop_params = {}
            for name in spec.param_names:
                target = spec.target_values.get(name, 1.0)
                pop_params[name] = (
                    torch.randn(pop_size, 1, device=device) * 0.5 + target
                ).requires_grad_(True)

            # xs: shape (N,) broadcast with params (M, 1) → results (M, N)
            pop_inputs = {**xs_data_1d, **pop_params}

            def pop_forward():
                return evaluate_batched(graph, pop_inputs)

            # Sequential population: M × N individual evaluations
            def pop_sequential():
                for m in range(pop_size):
                    for i in range(n):
                        inp = {name: (v[i].item() if v.dim() > 0 and v.shape[0] == n else v.item())
                               for name, v in xs_data_1d.items()}
                        inp.update({name: pop_params[name][m, 0].item()
                                    for name in spec.param_names})
                        evaluate(graph, inp)

            t_batch = _time_fn(pop_forward, cfg.warmup_iters, cfg.timing_iters)

            total_evals = pop_size * n
            if total_evals <= 6400:
                t_seq = _time_fn(pop_sequential, 1, 3)
            else:
                t_seq = float('nan')

            throughput_batch = total_evals / t_batch
            throughput_seq = total_evals / t_seq if not math.isnan(t_seq) else float('nan')
            speedup = throughput_batch / throughput_seq if not math.isnan(throughput_seq) else float('nan')

            # Verify gradient flows through population
            r = pop_forward()
            if r.dim() == 2:
                loss = (r - ys.unsqueeze(0)).pow(2).mean()
            else:
                loss = (r - ys).pow(2).mean()
            loss.backward()
            has_grad = all(pop_params[name].grad is not None for name in spec.param_names)

            for p in pop_params.values():
                p.grad = None

            row = {
                "model": model_name, "pop_size": pop_size,
                "n_data": n, "total_evals": total_evals,
                "device": str(device),
                "time_batched_s": t_batch, "time_sequential_s": t_seq,
                "throughput_batched": throughput_batch,
                "throughput_sequential": throughput_seq,
                "speedup": speedup, "grad_flows": has_grad,
            }
            results.append(row)
            sp_str = f"{speedup:.1f}x" if not math.isnan(speedup) else "N/A"
            print(f"  {model_name:30s} pop={pop_size:5d} evals={total_evals:7d}  "
                  f"batch={t_batch:.4f}s  speedup={sp_str}  grad={has_grad}")

    return results


# ===================================================================
# Part E: torch.compile Speedup
# ===================================================================

def run_part_e(cfg: ExpHConfig, device: torch.device):
    print("\n" + "=" * 60)
    print("Part E: torch.compile Speedup")
    print("=" * 60)

    results = []
    for model_name in cfg.batchable_models:
        spec = MODEL_BY_NAME[model_name]
        graph = _compile_direct(spec)
        all_names = spec.input_names + spec.param_names

        compiled_fn = compile_batched(graph, input_names=all_names)

        for bs in [64, 512, 4096]:
            xs_data, _ = _generate_data(spec, bs, device)
            params = {
                name: torch.tensor(
                    spec.target_values.get(name, spec.init_values.get(name, 1.0)),
                    dtype=torch.float32, device=device)
                for name in spec.param_names
            }
            batch_inputs = {**xs_data, **params}
            ordered_args = tuple(batch_inputs[n] for n in all_names)

            def fwd_plain():
                return evaluate_batched(graph, batch_inputs)

            def fwd_compiled():
                return compiled_fn(*ordered_args)

            # Forward only
            t_plain = _time_fn(fwd_plain, cfg.warmup_iters, cfg.timing_iters)
            t_compiled = _time_fn(fwd_compiled,
                                  cfg.warmup_iters + 5, cfg.timing_iters)
            fwd_speedup = t_plain / t_compiled

            # Forward + backward
            grad_params = {
                name: torch.tensor(
                    spec.target_values.get(name, spec.init_values.get(name, 1.0)),
                    dtype=torch.float32, device=device, requires_grad=True)
                for name in spec.param_names
            }
            grad_inputs_plain = {**xs_data, **grad_params}
            grad_inputs_ordered = {**xs_data, **grad_params}
            grad_ordered = tuple(grad_inputs_ordered[n] for n in all_names)

            def fwd_bwd_plain():
                r = evaluate_batched(graph, grad_inputs_plain)
                r.sum().backward()
                for p in grad_params.values():
                    p.grad = None

            def fwd_bwd_compiled():
                r = compiled_fn(*grad_ordered)
                r.sum().backward()
                for p in grad_params.values():
                    p.grad = None

            t_plain_gb = _time_fn(fwd_bwd_plain, cfg.warmup_iters,
                                  cfg.timing_iters)
            t_compiled_gb = _time_fn(fwd_bwd_compiled,
                                     cfg.warmup_iters + 5, cfg.timing_iters)
            gb_speedup = t_plain_gb / t_compiled_gb

            # Correctness
            r1 = fwd_plain()
            r2 = fwd_compiled()
            max_diff = (r1 - r2).abs().max().item()

            row = {
                "model": model_name, "batch_size": bs, "device": str(device),
                "time_plain_fwd_s": t_plain,
                "time_compiled_fwd_s": t_compiled,
                "fwd_speedup": fwd_speedup,
                "time_plain_fwdbwd_s": t_plain_gb,
                "time_compiled_fwdbwd_s": t_compiled_gb,
                "fwdbwd_speedup": gb_speedup,
                "max_diff": max_diff,
            }
            results.append(row)
            print(f"  {model_name:30s} bs={bs:5d}  "
                  f"fwd={fwd_speedup:.2f}x  fwd+bwd={gb_speedup:.2f}x  "
                  f"diff={max_diff:.2e}")

    return results


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="Experiment H: Batched GPU Parallelization")
    parser.add_argument("--part", choices=["A", "B", "C", "D", "E", "all"],
                        default="all")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    cfg = DEFAULT
    print(f"Experiment H: device={device}, models={len(cfg.batchable_models)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}

    if args.part in ("A", "all"):
        all_results["part_a"] = run_part_a(cfg, device)
    if args.part in ("B", "all"):
        all_results["part_b"] = run_part_b(cfg, device)
    if args.part in ("C", "all"):
        all_results["part_c"] = run_part_c(cfg, device)
    if args.part in ("D", "all"):
        all_results["part_d"] = run_part_d(cfg, device)
    if args.part in ("E", "all"):
        all_results["part_e"] = run_part_e(cfg, device)

    outfile = OUTPUT_DIR / f"exp_h_{args.device}.json"
    with open(outfile, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {outfile}")


if __name__ == "__main__":
    main()
