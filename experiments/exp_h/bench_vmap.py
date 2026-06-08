############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# bench_vmap.py: Exp H — vmap baseline: DMCI batched throughput vs. jax.vmap of the directly-compiled model. Answers the ICLR...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Exp H — vmap baseline: DMCI batched throughput vs. jax.vmap of the directly-compiled model.

Answers the ICLR review's #1 ask (REVIEW_MEDIATION.md, R1/Part B): are DMCI's batched and
population speedups competitive with standard vectorization, or do they merely recover
self-inflicted interpreter overhead? Every other Exp H speedup is measured against DMCI's OWN
sequential Python loop; this benchmark adds the missing cross-system reference.

Method (apples-to-apples on the SAME compiled graph): for each heap-free batchable model we run the
identical directly-compiled `ComputeGraph` two ways on identical hardware/data ---
  - DMCI:     `evaluate_batched(graph, inputs)` (Python graph-walk with broadcasting), and
  - baseline: `jax.jit(jax.vmap(...))` of `emit_jax_module(graph)` (XLA-compiled vectorization) ---
and report DMCI throughput as a FRACTION of the jax.vmap baseline for forward, population (M x N),
and one training step. jit/trace (warm-up) time is reported SEPARATELY from steady-state (XLA's
one-time compile is a cost DMCI does not pay; conflating them would flatter DMCI). A numerical
equivalence check (max abs diff) guards that the two paths compute the same function.

Honest expectation: jax.vmap+XLA is faster, so the fractions are < 1. The contribution is NOT raw
speed; it is that batched DMCI obtains the same single-graph/single-optimizer (M,N) sweep
AUTOMATICALLY for any program supplied as data, with zero per-program vectorization.

Run on HPC (jax[cpu] installed):
  python -m experiments.exp_h.bench_vmap --device cpu
"""

from __future__ import annotations

import argparse
import json
import time
import importlib.util
import tempfile
from pathlib import Path

import numpy as np
import torch

from experiments.exp_b.models import MODEL_BY_NAME, ModelSpec
from experiments.exp_h.config import DEFAULT
from experiments.exp_h.exp_h import _compile_direct, _generate_data, _time_fn
from neural_compiler.evaluator import evaluate_batched
# NOTE: the directly-compiled (heap-free) graph returns a RAW tensor from evaluate_batched
# (not a tagged value), exactly as Exp H's own Part B/C consume it — so we do NOT unwrap it.

OUTPUT_DIR = Path(__file__).parent / "results"


def _load_jax_apply(spec: ModelSpec, graph):
    """Emit the directly-compiled graph as a standalone JAX module and import its `apply`."""
    from neural_compiler.emit import emit_jax_module           # lazy: pulls in jax
    src = emit_jax_module(graph, spec.direct_source, spec.param_names, spec.input_names)
    path = Path(tempfile.mkdtemp()) / f"jaxmod_{spec.name}.py"
    path.write_text(src)
    s = importlib.util.spec_from_file_location(f"jaxmod_{spec.name}", path)
    mod = importlib.util.module_from_spec(s)
    s.loader.exec_module(mod)
    return mod.apply, mod.init_params


def _time_jax(fn, iters: int):
    import jax
    jax.block_until_ready(fn())                                # ensure compiled before timing
    t0 = time.perf_counter()
    for _ in range(iters):
        jax.block_until_ready(fn())
    return (time.perf_counter() - t0) / iters


def bench_model(spec: ModelSpec, N: int, pop_sizes, warmup: int, iters: int) -> dict:
    import jax
    import jax.numpy as jnp

    graph = _compile_direct(spec)
    if graph.uses_tagged_values:
        return {"model": spec.name, "skipped": "uses heap/tagged values; JAX backend N/A"}

    device = torch.device("cpu")                               # CPU parity with jax[cpu]
    xs_data, ys = _generate_data(spec, N, device)
    target = {n: float(spec.target_values.get(n, spec.init_values.get(n, 1.0)))
              for n in spec.param_names}

    # ---- JAX baseline: vmap+jit of the directly-compiled model ----
    apply, init_params = _load_jax_apply(spec, graph)
    data_jax = {n: jnp.asarray(xs_data[n].cpu().numpy(), jnp.float32) for n in spec.input_names}
    p0 = init_params(**target)
    fwd = jax.jit(lambda p: jax.vmap(lambda d: apply(p, **d))(data_jax))
    t_c = time.perf_counter(); jax.block_until_ready(fwd(p0)); jax_jit_s = time.perf_counter() - t_c
    t_jax_fwd = _time_jax(lambda: fwd(p0), iters)
    jax_fwd_out = np.asarray(fwd(p0)).reshape(-1)

    # ---- DMCI batched forward: same graph, evaluate_batched ----
    params_t = {n: torch.tensor(target[n], dtype=torch.float32, device=device)
                for n in spec.param_names}
    bin_fwd = {**xs_data, **params_t}
    dmci_fwd_out = evaluate_batched(graph, bin_fwd).detach().cpu().numpy().reshape(-1)
    t_dmci_fwd = _time_fn(lambda: evaluate_batched(graph, bin_fwd), warmup, iters)

    max_abs_diff = float(np.max(np.abs(jax_fwd_out - dmci_fwd_out)))
    forward = {
        "N": N, "jax_jit_s": jax_jit_s, "jax_time_s": t_jax_fwd, "dmci_time_s": t_dmci_fwd,
        "jax_throughput": N / t_jax_fwd, "dmci_throughput": N / t_dmci_fwd,
        "dmci_frac_of_vmap": t_jax_fwd / t_dmci_fwd,            # <1 means DMCI slower than vmap
        "max_abs_diff": max_abs_diff,
    }

    # ---- population (M x N) ----
    rng = np.random.default_rng(0)
    population = []
    for M in pop_sizes:
        pop_np = {n: rng.normal(target[n], 0.5, size=M).astype("float32") for n in spec.param_names}
        pop_p = {n: jnp.asarray(pop_np[n]) for n in spec.param_names}
        popfwd = jax.jit(lambda pp: jax.vmap(
            lambda one: jax.vmap(lambda d: apply(one, **d))(data_jax))(pp))
        t_c = time.perf_counter(); jax.block_until_ready(popfwd(pop_p))
        jax_pop_jit = time.perf_counter() - t_c
        t_jax_pop = _time_jax(lambda: popfwd(pop_p), max(3, iters // 2))

        pop_pt = {n: torch.tensor(pop_np[n], dtype=torch.float32, device=device).reshape(M, 1)
                  for n in spec.param_names}
        bin_pop = {**xs_data, **pop_pt}
        t_dmci_pop = _time_fn(lambda: evaluate_batched(graph, bin_pop),
                              max(2, warmup // 2), max(3, iters // 2))
        tot = M * N
        population.append({
            "M": M, "total_evals": tot, "jax_jit_s": jax_pop_jit,
            "jax_time_s": t_jax_pop, "dmci_time_s": t_dmci_pop,
            "jax_throughput": tot / t_jax_pop, "dmci_throughput": tot / t_dmci_pop,
            "dmci_frac_of_vmap": t_jax_pop / t_dmci_pop,
        })

    # ---- one training step (value+grad over the batch) ----
    yj = jnp.asarray(ys.cpu().numpy(), jnp.float32)

    def jax_loss(p):
        pred = jax.vmap(lambda d: apply(p, **d))(data_jax)
        return jnp.mean((pred - yj) ** 2)

    gstep = jax.jit(jax.grad(jax_loss))
    t_c = time.perf_counter(); jax.block_until_ready(gstep(p0)); jax_grad_jit = time.perf_counter() - t_c
    t_jax_step = _time_jax(lambda: gstep(p0), iters)

    def dmci_step():
        ps = {n: torch.tensor(target[n], dtype=torch.float32, device=device, requires_grad=True)
              for n in spec.param_names}
        pred = evaluate_batched(graph, {**xs_data, **ps})
        loss = ((pred - ys) ** 2).mean()
        loss.backward()

    t_dmci_step = _time_fn(dmci_step, warmup, iters)
    training = {
        "jax_jit_s": jax_grad_jit, "jax_time_s": t_jax_step, "dmci_time_s": t_dmci_step,
        "dmci_frac_of_vmap": t_jax_step / t_dmci_step,
    }

    return {"model": spec.name, "forward": forward, "population": population, "training": training}


def main():
    ap = argparse.ArgumentParser(description="Exp H vmap baseline (DMCI batched vs jax.vmap)")
    ap.add_argument("--models", nargs="+", default=DEFAULT.batchable_models)
    ap.add_argument("--n-data", type=int, default=DEFAULT.n_data_points)
    ap.add_argument("--pop-sizes", type=int, nargs="+", default=[1, 10, 100])
    ap.add_argument("--warmup", type=int, default=DEFAULT.warmup_iters)
    ap.add_argument("--iters", type=int, default=DEFAULT.timing_iters)
    ap.add_argument("--output", type=Path, default=OUTPUT_DIR / "vmap_results.json")
    args = ap.parse_args()

    try:
        import jax
        print(f"jax {jax.__version__}; devices={jax.devices()}")
    except Exception as e:
        raise SystemExit(f"jax is required for the vmap baseline: {e}")

    results = []
    for name in args.models:
        spec = MODEL_BY_NAME[name]
        print(f"\n=== {name} ===", flush=True)
        try:
            rec = bench_model(spec, args.n_data, args.pop_sizes, args.warmup, args.iters)
        except Exception as e:
            print(f"  FAILED: {e}", flush=True)
            rec = {"model": name, "error": str(e)}
        results.append(rec)
        if "forward" in rec:
            f = rec["forward"]
            print(f"  forward  N={f['N']}: dmci={f['dmci_time_s']*1e3:.2f}ms "
                  f"vmap={f['jax_time_s']*1e3:.2f}ms (jit {f['jax_jit_s']*1e3:.0f}ms) "
                  f"-> DMCI is {f['dmci_frac_of_vmap']:.2f}x vmap | maxdiff={f['max_abs_diff']:.1e}")
            for p in rec["population"]:
                print(f"  pop M={p['M']:5d} (MxN={p['total_evals']}): "
                      f"DMCI is {p['dmci_frac_of_vmap']:.2f}x vmap "
                      f"(dmci={p['dmci_time_s']*1e3:.1f}ms vmap={p['jax_time_s']*1e3:.1f}ms)")
            t = rec["training"]
            print(f"  train step: DMCI is {t['dmci_frac_of_vmap']:.2f}x vmap "
                  f"(dmci={t['dmci_time_s']*1e3:.1f}ms vmap={t['jax_time_s']*1e3:.1f}ms)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {args.output}")

    # headline: geometric-mean DMCI fraction of vmap across covered models
    covered = [r for r in results if "forward" in r]
    if covered:
        import math
        gm = lambda xs: math.exp(sum(math.log(max(x, 1e-12)) for x in xs) / len(xs))
        print(f"\nCovered {len(covered)}/{len(results)} models.")
        print(f"  geomean DMCI/vmap forward  : {gm([r['forward']['dmci_frac_of_vmap'] for r in covered]):.3f}")
        print(f"  geomean DMCI/vmap train    : {gm([r['training']['dmci_frac_of_vmap'] for r in covered]):.3f}")


if __name__ == "__main__":
    main()
