############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# bench_diffsoc.py: Benchmark DiffSoc-S: single-thread CPU vs batched CPU vs batched GPU. Measures forward-only and...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Benchmark DiffSoc-S: single-thread CPU vs batched CPU vs batched GPU.

Measures forward-only and forward+backward wall time for the DiffSoc-S
Urban Political Economy Simulator at various batch sizes and timestep counts.
Apple-to-apple comparison with bench_diffesm.py.

Usage:
  python -m experiments.exp_h.bench_diffsoc [--device cpu|cuda|all]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.setrecursionlimit(10000)

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from neural_compiler.compiler import compile_scheme
from neural_compiler.evaluator import evaluate_batched

from experiments.exp_h.test_diffsoc import DEFAULT_PARAMS, PARAMS

OUTPUT_DIR = Path(__file__).parent / "results"
DIFFSOC_PATH = PROJECT_ROOT / "large_examples" / "diffsoc_s.scm"

PARAM_NAMES = list(PARAMS.keys())
BATCH_SIZES = [1, 4, 16, 64, 256, 1024]
TIMESTEPS = [10, 50, 100]
WARMUP = 3
TIMING_ITERS = 10


def _compile():
    source = DIFFSOC_PATH.read_text()
    return compile_scheme(source, inputs=DEFAULT_PARAMS)


def _make_inputs(n_steps: int, device: torch.device, requires_grad=False):
    inputs = {}
    for k, v in DEFAULT_PARAMS.items():
        t = torch.tensor(float(v), dtype=torch.float32, device=device)
        if requires_grad and k in PARAM_NAMES:
            t = t.requires_grad_(True)
        inputs[k] = t
    inputs["n_steps"] = torch.tensor(float(n_steps), device=device)
    return inputs


def _make_batched_inputs(n_steps: int, batch_size: int, device: torch.device,
                         requires_grad=False):
    inputs = {}
    for k, v in DEFAULT_PARAMS.items():
        if k in PARAM_NAMES and requires_grad:
            base = torch.tensor(float(v), dtype=torch.float32, device=device)
            noise = torch.randn(batch_size, device=device) * abs(float(v)) * 0.01
            t = (base + noise).requires_grad_(True)
        elif k == "e_grow":
            t = torch.linspace(0.0, 0.05, batch_size, device=device)
        else:
            t = torch.tensor(float(v), dtype=torch.float32, device=device)
        inputs[k] = t
    inputs["n_steps"] = torch.tensor(float(n_steps), device=device)
    return inputs


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
    return (time.perf_counter() - t0) / iters


def bench_sequential(graph, n_steps: int, device: torch.device):
    results = []

    for bs in BATCH_SIZES:
        if bs > 64:
            results.append({
                "method": "sequential", "batch_size": bs, "n_steps": n_steps,
                "device": str(device), "fwd_time": float("nan"),
                "fwd_bwd_time": float("nan"), "note": "skipped (too slow)",
            })
            continue

        inputs_list = [_make_inputs(n_steps, device) for _ in range(bs)]

        def fwd():
            for inp in inputs_list:
                evaluate_batched(graph, inp)

        t_fwd = _time_fn(fwd, min(WARMUP, 1), min(TIMING_ITERS, 3))

        inputs_grad = [_make_inputs(n_steps, device, requires_grad=True) for _ in range(bs)]

        def fwd_bwd():
            for inp in inputs_grad:
                r = evaluate_batched(graph, inp)
                r.backward()
                for k in PARAM_NAMES:
                    if inp[k].grad is not None:
                        inp[k].grad.zero_()

        t_fwd_bwd = _time_fn(fwd_bwd, 1, min(TIMING_ITERS, 3))

        row = {
            "method": "sequential", "batch_size": bs, "n_steps": n_steps,
            "device": str(device),
            "fwd_time": t_fwd, "fwd_bwd_time": t_fwd_bwd,
            "fwd_per_eval": t_fwd / bs,
            "fwd_bwd_per_eval": t_fwd_bwd / bs,
        }
        results.append(row)
        print(f"  seq  bs={bs:4d}  fwd={t_fwd:.4f}s  fwd+bwd={t_fwd_bwd:.4f}s  "
              f"({t_fwd/bs*1000:.2f} ms/eval)")

    return results


def bench_batched(graph, n_steps: int, device: torch.device):
    results = []

    for bs in BATCH_SIZES:
        inputs = _make_batched_inputs(n_steps, bs, device)

        def fwd():
            return evaluate_batched(graph, inputs)

        t_fwd = _time_fn(fwd, WARMUP, TIMING_ITERS)

        inputs_grad = _make_batched_inputs(n_steps, bs, device, requires_grad=True)

        def fwd_bwd():
            r = evaluate_batched(graph, inputs_grad)
            loss = r.sum()
            loss.backward()
            for k in PARAM_NAMES:
                if inputs_grad[k].grad is not None:
                    inputs_grad[k].grad.zero_()

        t_fwd_bwd = _time_fn(fwd_bwd, WARMUP, TIMING_ITERS)

        mem_mb = float("nan")
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            evaluate_batched(graph, inputs_grad).sum().backward()
            mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

        row = {
            "method": "batched", "batch_size": bs, "n_steps": n_steps,
            "device": str(device),
            "fwd_time": t_fwd, "fwd_bwd_time": t_fwd_bwd,
            "fwd_per_eval": t_fwd / bs,
            "fwd_bwd_per_eval": t_fwd_bwd / bs,
            "peak_gpu_mb": mem_mb,
        }
        results.append(row)
        mem_str = f"  mem={mem_mb:.0f}MB" if not math.isnan(mem_mb) else ""
        print(f"  bat  bs={bs:4d}  fwd={t_fwd:.4f}s  fwd+bwd={t_fwd_bwd:.4f}s  "
              f"({t_fwd/bs*1000:.2f} ms/eval){mem_str}")

    return results


def run_benchmark(device: torch.device):
    print(f"\n{'='*70}")
    print(f"DiffSoc-S Benchmark — device={device}")
    print(f"{'='*70}")

    print("\nCompiling DiffSoc-S...")
    graph = _compile()
    print(f"  {len(graph.nodes)} outer nodes, "
          f"uses_tagged_values={graph.uses_tagged_values}")

    all_results = []

    for n_steps in TIMESTEPS:
        print(f"\n--- n_steps={n_steps} ---")

        print(f"\n  [Sequential, {device}]")
        seq_results = bench_sequential(graph, n_steps, device)
        all_results.extend(seq_results)

        print(f"\n  [Batched, {device}]")
        bat_results = bench_batched(graph, n_steps, device)
        all_results.extend(bat_results)

        print(f"\n  [Speedup: batched/sequential]")
        for br in bat_results:
            bs = br["batch_size"]
            sr = next((s for s in seq_results if s["batch_size"] == bs), None)
            if sr and not math.isnan(sr.get("fwd_time", float("nan"))):
                fwd_speedup = sr["fwd_time"] / br["fwd_time"]
                bwd_speedup = sr["fwd_bwd_time"] / br["fwd_bwd_time"]
                print(f"    bs={bs:4d}  fwd={fwd_speedup:.1f}x  fwd+bwd={bwd_speedup:.1f}x")
                br["fwd_speedup"] = fwd_speedup
                br["fwd_bwd_speedup"] = bwd_speedup

    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="all", choices=["cpu", "cuda", "all"])
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []

    if args.device in ("cpu", "all"):
        torch.set_num_threads(1)
        results = run_benchmark(torch.device("cpu"))
        all_results.extend(results)

    if args.device in ("cuda", "all"):
        if not torch.cuda.is_available():
            print("\nCUDA not available, skipping GPU benchmark")
        else:
            results = run_benchmark(torch.device("cuda"))
            all_results.extend(results)

    out_path = OUTPUT_DIR / f"bench_diffsoc_{args.device}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    print(f"\n{'='*70}")
    print("SUMMARY TABLE")
    print(f"{'='*70}")
    print(f"{'method':<12} {'device':<6} {'steps':>5} {'batch':>5} "
          f"{'fwd(s)':>10} {'fwd+bwd(s)':>12} {'ms/eval':>10} {'speedup':>8}")
    print("-" * 80)
    for r in all_results:
        if math.isnan(r.get("fwd_time", float("nan"))):
            continue
        speedup = r.get("fwd_speedup", "")
        if speedup:
            speedup = f"{speedup:.1f}x"
        print(f"{r['method']:<12} {r['device']:<6} {r['n_steps']:>5} {r['batch_size']:>5} "
              f"{r['fwd_time']:>10.4f} {r['fwd_bwd_time']:>12.4f} "
              f"{r['fwd_per_eval']*1000:>10.2f} {speedup:>8}")


if __name__ == "__main__":
    main()
