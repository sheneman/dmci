############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# bench_dmci_batched.py: Batched-DMCI throughput benchmark (Exp-H-style, but the meta-circular-interpreter path). The original...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Batched-DMCI throughput benchmark (Exp-H-style, but the meta-circular-interpreter path).

The original Experiment H batched only heap-FREE / directly-compiled graphs
("Programs using heap operations are excluded", paper2 Exp H). This benchmark measures
the case that exclusion ruled out: the heap-using compiled interpreter (DMCI itself),
which now batches natively (v1.1.7) because structural values stay scalar/data-independent
while only numeric leaves carry the batch dimension.

For a set of DMCI programs (compiler.scm + (scheme-eval 'P env)) it reports:
  - correctness: batched == sequential (max |diff|), all gradients finite
  - forward throughput: sequential (N interpreter walks) vs batched (1 walk) across batch sizes
  - training speedup: batched vs sequential fit at a fixed epoch budget

Run on the CPU 'eight' partition (DMCI is scalar/Python-bound; the paper shows CPU > GPU here).
Usage:  python -m experiments.exp_h.bench_dmci_batched [--max-batch 1024] [--train-epochs 20]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from neural_compiler.compiler import compile_program
from neural_compiler.evaluator import evaluate
from neural_compiler.runtime.tagged_value import make_float, unwrap_number

EVAL_SRC = (Path(__file__).parent.parent.parent / "bootstrap" / "compiler.scm").read_text()


def _env(names):
    return "(list " + " ".join(f"(cons '{n} {n})" for n in names) + ")"


# --- DMCI benchmark programs (each: interp_source + schema + driver ranges) ---

def _equation_model():
    expr = "(* a (exp (* (- 0 b) x)))"
    return dict(
        name="eq_beer_lambert", kind="arithmetic",
        interp=EVAL_SRC + f"\n(scheme-eval '{expr} {_env(['x', 'a', 'b'])})\n",
        inputs=["x"], params=["a", "b"], targets={"a": 2.5, "b": 0.8},
        ranges={"x": (0.1, 5.0)})


def _decay_chain_model():
    from experiments.exp_c.models import C03_DECAY_CHAIN as c
    return dict(
        name=c.name, kind="recursive_ode",
        interp=c.interp_source, inputs=list(c.input_names), params=list(c.param_names),
        targets=dict(c.target_values), ranges={c.input_names[0]: c.x_range})


def _gpp_model():
    from experiments.exp_i.models import build_static_model
    m = build_static_model(2)
    return dict(
        name=m.name, kind="multi_driver_composite",
        interp=m.interp_source, inputs=list(m.input_names), params=list(m.param_names),
        targets=dict(m.target_values), ranges=dict(m.driver_ranges))


def _compile(m):
    return compile_program(
        m["interp"], inputs={n: None for n in m["inputs"] + m["params"]}, prelude=True)


def _drivers(m, B, seed=0):
    g = torch.Generator().manual_seed(seed)
    cols = {}
    for d in m["inputs"]:
        lo, hi = m["ranges"][d]
        cols[d] = lo + (hi - lo) * torch.rand(B, generator=g)
    return cols


def _seq_eval(graph, m, cols, B):
    pv = {n: torch.tensor(float(m["targets"][n])) for n in m["params"]}
    out = []
    for i in range(B):
        tv = {d: make_float(cols[d][i]) for d in m["inputs"]}
        for n, v in pv.items():
            tv[n] = make_float(v)
        out.append(unwrap_number(evaluate(graph, tv)))
    return torch.stack(out)


def _bat_eval(graph, m, cols):
    binp = {d: make_float(cols[d]) for d in m["inputs"]}
    for n in m["params"]:
        binp[n] = make_float(torch.tensor(float(m["targets"][n])))
    return unwrap_number(evaluate(graph, binp))


def bench_model(m, batch_sizes, train_epochs):
    graph = _compile(m)

    # correctness @ B=64
    cols = _drivers(m, 64)
    s = _seq_eval(graph, m, cols, 64)
    b = _bat_eval(graph, m, cols)
    max_diff = float((s - b).abs().max())
    p = {n: torch.tensor(float(m["targets"][n]), requires_grad=True) for n in m["params"]}
    binp = {d: make_float(cols[d]) for d in m["inputs"]}
    for n in m["params"]:
        binp[n] = make_float(p[n])
    (unwrap_number(evaluate(graph, binp)) ** 2).mean().backward()
    grads_finite = all(p[n].grad is not None and torch.isfinite(p[n].grad).all()
                       for n in m["params"])

    rows = []
    for B in batch_sizes:
        cols = _drivers(m, B)
        _ = _bat_eval(graph, m, cols)                         # warmup
        t0 = time.perf_counter()
        for _ in range(3):
            _ = _bat_eval(graph, m, cols)
        t_bat = (time.perf_counter() - t0) / 3
        t0 = time.perf_counter()
        _ = _seq_eval(graph, m, cols, B)                      # B interpreter walks
        t_seq = time.perf_counter() - t0
        rows.append({"B": B, "t_seq_s": t_seq, "t_bat_s": t_bat,
                     "fwd_speedup": t_seq / t_bat if t_bat > 0 else float("nan")})
        print(f"    B={B:5d}  seq={t_seq:8.3f}s  bat={t_bat:8.4f}s  "
              f"speedup={t_seq / t_bat:8.1f}x")

    # training speedup @ N=64
    train = _train_speedup(graph, m, n_data=64, epochs=train_epochs)
    return {"model": m["name"], "kind": m["kind"], "n_params": len(m["params"]),
            "max_diff": max_diff, "grads_finite": grads_finite,
            "throughput": rows, "training": train}


def _train_speedup(graph, m, n_data, epochs):
    cols = _drivers(m, n_data, seed=7)
    pv = {n: torch.tensor(float(m["targets"][n])) for n in m["params"]}
    ys = (_seq_eval(graph, m, cols, n_data)).detach()

    def init_params():
        torch.manual_seed(0)
        return {n: torch.nn.Parameter(torch.tensor(float(m["targets"][n])) + 0.1)
                for n in m["params"]}

    # sequential epoch: n_data walks
    pars = init_params()
    opt = torch.optim.Adam(list(pars.values()), lr=0.01)
    t0 = time.perf_counter()
    for _ in range(epochs):
        loss = torch.tensor(0.0)
        for i in range(n_data):
            tv = {d: make_float(cols[d][i]) for d in m["inputs"]}
            for n, pp in pars.items():
                tv[n] = make_float(pp)
            loss = loss + (unwrap_number(evaluate(graph, tv)) - ys[i]) ** 2
        opt.zero_grad(); loss.backward(); opt.step()
    t_seq = time.perf_counter() - t0

    # batched epoch: 1 walk
    pars = init_params()
    opt = torch.optim.Adam(list(pars.values()), lr=0.01)
    t0 = time.perf_counter()
    for _ in range(epochs):
        binp = {d: make_float(cols[d]) for d in m["inputs"]}
        for n, pp in pars.items():
            binp[n] = make_float(pp)
        loss = ((unwrap_number(evaluate(graph, binp)) - ys) ** 2).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    t_bat = time.perf_counter() - t0

    return {"n_data": n_data, "epochs": epochs, "t_seq_s": t_seq, "t_bat_s": t_bat,
            "train_speedup": t_seq / t_bat if t_bat > 0 else float("nan")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-batch", type=int, default=1024)
    ap.add_argument("--train-epochs", type=int, default=20)
    ap.add_argument("--output-dir", type=Path,
                    default=Path("experiments/exp_h/results"))
    args = ap.parse_args()
    sys.setrecursionlimit(5000)

    batch_sizes = [b for b in (1, 8, 64, 256, 1024) if b <= args.max_batch]
    models = [_equation_model(), _decay_chain_model(), _gpp_model()]

    results = []
    for m in models:
        print(f"\n=== {m['name']} ({m['kind']}, {len(m['params'])} params) ===")
        r = bench_model(m, batch_sizes, args.train_epochs)
        print(f"  correctness: max|seq-bat|={r['max_diff']:.2e}  grads_finite={r['grads_finite']}")
        print(f"  training (N=64, {args.train_epochs} ep): seq={r['training']['t_seq_s']:.1f}s "
              f"bat={r['training']['t_bat_s']:.1f}s  speedup={r['training']['train_speedup']:.1f}x")
        results.append(r)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "dmci_batched_bench.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
