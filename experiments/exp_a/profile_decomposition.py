############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# profile_decomposition.py: Profile the 10x DMCI overhead decomposition. Compiles a Scheme program in both direct and DMCI...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Profile the 10x DMCI overhead decomposition.

Compiles a Scheme program in both direct and DMCI (compiled-interpreter)
modes, runs forward+backward passes, and uses cProfile to decompose
wall-clock time into the major cost categories:

  - Tagged-value wrap/unwrap  (make_float, unwrap_number, ...)
  - Heap operations           (TensorHeap.cons, .car, .cdr, .read, .write)
  - Dispatch / soft-select    (tagged_if, soft_select)
  - Arithmetic primitives     (evaluate_op on raw tensors)
  - Autograd backward pass

Produces a formatted table suitable for the paper.
"""

from __future__ import annotations

import argparse
import cProfile
import pstats
import sys
import time
from io import StringIO
from pathlib import Path

import torch
import torch.nn as nn

from neural_compiler.compiler import compile_program
from neural_compiler.evaluator import evaluate
from neural_compiler.runtime.tagged_value import make_float, unwrap_number


# ---------------------------------------------------------------------------
# Source file paths (relative to repo root)
# ---------------------------------------------------------------------------

BOOTSTRAP_DIR = Path(__file__).parent.parent.parent / "bootstrap"
EVALUATOR_SOURCE = (BOOTSTRAP_DIR / "compiler.scm").read_text()


# ---------------------------------------------------------------------------
# Program definitions
# ---------------------------------------------------------------------------

PROGRAMS = {
    "multi_const": {
        "expr": "(+ (* alpha x) beta)",
        "params": {"alpha": 2.0, "beta": 1.0},
        "inputs": ["x"],
    },
    "single_const": {
        "expr": "(* alpha (* x x))",
        "params": {"alpha": 1.0},
        "inputs": ["x"],
    },
}


def _build_direct_source(prog: dict) -> str:
    """Build direct-compilation source for a program."""
    return prog["expr"]


def _build_dmci_source(prog: dict) -> str:
    """Build DMCI source: evaluator + scheme-eval call."""
    expr = prog["expr"]
    all_vars = prog["inputs"] + list(prog["params"].keys())
    env_entries = " ".join(f"(cons '{v} {v})" for v in all_vars)
    return (
        EVALUATOR_SOURCE
        + f"\n(scheme-eval '{expr} (list {env_entries}))\n"
    )


def _input_names(prog: dict) -> list[str]:
    return prog["inputs"] + list(prog["params"].keys())


# ---------------------------------------------------------------------------
# Category classification for cProfile stats
# ---------------------------------------------------------------------------

# Map (filename_substring, function_name_pattern) -> category
# Order matters: first match wins.
CATEGORY_RULES = [
    # Tagged value wrap/unwrap
    ("tagged_value.py", "make_float",       "tagged_value"),
    ("tagged_value.py", "make_int",         "tagged_value"),
    ("tagged_value.py", "make_bool",        "tagged_value"),
    ("tagged_value.py", "make_nil",         "tagged_value"),
    ("tagged_value.py", "make_pair",        "tagged_value"),
    ("tagged_value.py", "make_symbol",      "tagged_value"),
    ("tagged_value.py", "make_char",        "tagged_value"),
    ("tagged_value.py", "make_closure",     "tagged_value"),
    ("tagged_value.py", "make_vector",      "tagged_value"),
    ("tagged_value.py", "_make",            "tagged_value"),
    ("tagged_value.py", "from_scalar",      "tagged_value"),
    ("tagged_value.py", "unwrap_number",    "tagged_value"),
    ("tagged_value.py", "unwrap_bool",      "tagged_value"),
    ("tagged_value.py", "unwrap_char",      "tagged_value"),
    ("tagged_value.py", "unwrap_symbol_id", "tagged_value"),
    ("tagged_value.py", "unwrap_pair_addrs","tagged_value"),
    ("tagged_value.py", "unwrap_closure",   "tagged_value"),
    ("tagged_value.py", "extract_tag",      "tagged_value"),
    ("tagged_value.py", "extract_payload",  "tagged_value"),
    ("tagged_value.py", "type_index",       "tagged_value"),
    ("tagged_value.py", "is_nil",           "tagged_value"),
    ("tagged_value.py", "is_pair",          "tagged_value"),
    ("tagged_value.py", "is_number",        "tagged_value"),
    ("tagged_value.py", "is_symbol",        "tagged_value"),
    ("tagged_value.py", "is_closure",       "tagged_value"),
    ("tagged_value.py", "is_bool",          "tagged_value"),
    ("tagged_value.py", "is_type",          "tagged_value"),
    ("tagged_value.py", "to_scalar",        "tagged_value"),

    # Dispatch (soft branching)
    ("tagged_value.py", "tagged_if",        "dispatch"),
    ("tagged_value.py", "soft_select",      "dispatch"),

    # Heap operations
    ("heap.py", "cons",                     "heap"),
    ("heap.py", "car",                      "heap"),
    ("heap.py", "cdr",                      "heap"),
    ("heap.py", "read",                     "heap"),
    ("heap.py", "write",                    "heap"),
    ("heap.py", "build_list",               "heap"),
    ("heap.py", "reset",                    "heap"),
    ("heap.py", "allocated",                "heap"),

    # Tagged ops (cons/car/cdr wrappers, list ops, type predicates via tagged_ops)
    ("tagged_ops.py", "evaluate_tagged_op", "tagged_ops"),
    ("tagged_ops.py", "_tagged_arith",      "tagged_ops"),
    ("tagged_ops.py", "_tagged_compare",    "tagged_ops"),
    ("tagged_ops.py", "_tagged_logic",      "tagged_ops"),
    ("tagged_ops.py", "_op_cons",           "heap"),
    ("tagged_ops.py", "_op_car",            "heap"),
    ("tagged_ops.py", "_op_cdr",            "heap"),
    ("tagged_ops.py", "_op_list",           "heap"),
    ("tagged_ops.py", "_op_length",         "heap"),
    ("tagged_ops.py", "_op_append",         "heap"),
    ("tagged_ops.py", "_op_reverse",        "heap"),
    ("tagged_ops.py", "_op_null_p",         "tagged_value"),
    ("tagged_ops.py", "_op_pair_p",         "tagged_value"),
    ("tagged_ops.py", "_op_number_p",       "tagged_value"),
    ("tagged_ops.py", "_op_boolean_p",      "tagged_value"),
    ("tagged_ops.py", "_op_symbol_p",       "tagged_value"),
    ("tagged_ops.py", "_op_char_p",         "tagged_value"),
    ("tagged_ops.py", "_op_procedure_p",    "tagged_value"),
    ("tagged_ops.py", "_op_eq",             "dispatch"),
    ("tagged_ops.py", "_op_eqv",            "dispatch"),
    ("tagged_ops.py", "_op_equal",          "dispatch"),
    ("tagged_ops.py", "_deep_equal",        "dispatch"),
    ("tagged_ops.py", "materialize_quote",  "tagged_value"),

    # Raw arithmetic primitives
    ("primitives.py", "evaluate_op",        "arithmetic"),
    ("primitives.py", "_op_add",            "arithmetic"),
    ("primitives.py", "_op_sub",            "arithmetic"),
    ("primitives.py", "_op_mul",            "arithmetic"),
    ("primitives.py", "_op_div",            "arithmetic"),
    ("primitives.py", "_op_if",             "arithmetic"),
    ("primitives.py", "_op_eq",             "arithmetic"),
    ("primitives.py", "_op_lt",             "arithmetic"),
    ("primitives.py", "_op_gt",             "arithmetic"),
    ("primitives.py", "_op_le",             "arithmetic"),
    ("primitives.py", "_op_ge",             "arithmetic"),
    ("primitives.py", "_op_not",            "arithmetic"),

    # Evaluator engine (graph walking)
    ("engine.py", "_eval_lazy_tagged",      "evaluator"),
    ("engine.py", "_eval_graph_tagged",     "evaluator"),
    ("engine.py", "_evaluate_tagged",       "evaluator"),
    ("engine.py", "_eval_call_tagged",      "evaluator"),
    ("engine.py", "_eval_call_lazy_tagged", "evaluator"),
    ("engine.py", "_eval_dynamic_call",     "evaluator"),
    ("engine.py", "_eval_loop_tagged",      "evaluator"),
    ("engine.py", "_eval_loop_lazy_tagged", "evaluator"),
    ("engine.py", "_trace_loop_root_lazy",  "evaluator"),
    ("engine.py", "_pack_env",              "evaluator"),
    ("engine.py", "_unpack_env",            "evaluator"),
    ("engine.py", "_list_to_vec",           "evaluator"),
    ("engine.py", "_func_name_to_id",       "evaluator"),
    ("engine.py", "_func_id_to_name",       "evaluator"),
    ("engine.py", "evaluate",               "evaluator"),
    ("engine.py", "_eval_graph",            "evaluator"),
    ("engine.py", "_eval_lazy",             "evaluator"),
    ("engine.py", "_eval_call",             "evaluator"),
    ("engine.py", "_to_tensor",             "evaluator"),

    # Symbols
    ("symbols.py", None,                    "tagged_value"),
]


def classify_function(filename: str, funcname: str) -> str | None:
    """Classify a profiled function into a cost category."""
    for file_pat, func_pat, category in CATEGORY_RULES:
        if file_pat in filename:
            if func_pat is None or funcname == func_pat:
                return category
    # Torch autograd / tensor ops
    if "torch" in filename or "autograd" in filename:
        return "torch_runtime"
    return None


# ---------------------------------------------------------------------------
# Profiling harness
# ---------------------------------------------------------------------------

def profile_mode(
    mode_name: str,
    graph,
    input_names: list[str],
    param_values: dict[str, float],
    n_iters: int,
) -> tuple[float, float, float, pstats.Stats]:
    """Profile forward+backward for one mode.

    Returns:
        (total_wall_s, fwd_wall_s, bwd_wall_s, cprofile_stats)
    """
    # Create learnable parameters
    params = {
        name: nn.Parameter(torch.tensor(val))
        for name, val in param_values.items()
    }

    x_val = torch.tensor(1.5)

    # Warmup (5 iterations, uncounted)
    for _ in range(5):
        inputs = {"x": make_float(x_val)}
        for name, param in params.items():
            inputs[name] = make_float(param)
        result = evaluate(graph, inputs)
        pred = unwrap_number(result)
        loss = pred ** 2
        loss.backward()
        for p in params.values():
            if p.grad is not None:
                p.grad.zero_()

    # Timed run with cProfile
    profiler = cProfile.Profile()

    total_fwd = 0.0
    total_bwd = 0.0

    profiler.enable()
    t_total_start = time.perf_counter()

    for _ in range(n_iters):
        # Forward pass
        t_fwd_start = time.perf_counter()
        inputs = {"x": make_float(x_val)}
        for name, param in params.items():
            inputs[name] = make_float(param)
        result = evaluate(graph, inputs)
        pred = unwrap_number(result)
        loss = pred ** 2
        t_fwd_end = time.perf_counter()
        total_fwd += t_fwd_end - t_fwd_start

        # Backward pass
        t_bwd_start = time.perf_counter()
        loss.backward()
        t_bwd_end = time.perf_counter()
        total_bwd += t_bwd_end - t_bwd_start

        for p in params.values():
            if p.grad is not None:
                p.grad.zero_()

    t_total_end = time.perf_counter()
    profiler.disable()

    total_wall = t_total_end - t_total_start

    # Collect stats
    stream = StringIO()
    stats = pstats.Stats(profiler, stream=stream)

    return total_wall, total_fwd, total_bwd, stats


def aggregate_categories(stats: pstats.Stats) -> dict[str, float]:
    """Aggregate cProfile stats into cost categories.

    Returns dict mapping category name -> cumulative time (seconds).
    """
    categories: dict[str, float] = {}
    # stats.stats is dict: (filename, lineno, funcname) -> (ncalls, totcalls, tottime, cumtime, callers)
    for (filename, _lineno, funcname), (ncalls, _totcalls, tottime, _cumtime, _callers) in stats.stats.items():
        cat = classify_function(filename, funcname)
        if cat is not None:
            categories[cat] = categories.get(cat, 0.0) + tottime
    return categories


# ---------------------------------------------------------------------------
# Table output
# ---------------------------------------------------------------------------

CATEGORY_LABELS = {
    "tagged_value": "Tagged-value wrap/unwrap",
    "heap":         "Heap operations (cons/car/cdr)",
    "dispatch":     "Dispatch (tagged_if/eq?/select)",
    "tagged_ops":   "Tagged arithmetic wrappers",
    "arithmetic":   "Raw arithmetic primitives",
    "evaluator":    "Evaluator graph walking",
    "torch_runtime":"PyTorch runtime / autograd",
}

CATEGORY_ORDER = [
    "tagged_value",
    "heap",
    "dispatch",
    "tagged_ops",
    "arithmetic",
    "evaluator",
    "torch_runtime",
]


def print_table(
    mode_name: str,
    total_wall: float,
    fwd_wall: float,
    bwd_wall: float,
    categories: dict[str, float],
    n_iters: int,
):
    """Print a formatted profiling table."""
    total_ms = total_wall * 1000.0
    fwd_ms = fwd_wall * 1000.0
    bwd_ms = bwd_wall * 1000.0
    per_iter_ms = total_ms / n_iters

    cat_total = sum(categories.values())

    print(f"\n{'=' * 72}")
    print(f"  {mode_name} — {n_iters} iterations")
    print(f"{'=' * 72}")
    print(f"  Total wall time:      {total_ms:10.1f} ms  ({per_iter_ms:.3f} ms/iter)")
    print(f"  Forward pass total:   {fwd_ms:10.1f} ms  ({fwd_ms/n_iters:.3f} ms/iter)")
    print(f"  Backward pass total:  {bwd_ms:10.1f} ms  ({bwd_ms/n_iters:.3f} ms/iter)")
    print(f"  Fwd/Bwd ratio:        {fwd_ms/max(bwd_ms, 1e-9):10.2f}x")
    print(f"{'─' * 72}")
    print(f"  {'Component':<38s} {'Time (ms)':>10s} {'Fraction':>10s} {'ms/iter':>10s}")
    print(f"  {'─' * 38} {'─' * 10} {'─' * 10} {'─' * 10}")

    for cat in CATEGORY_ORDER:
        t = categories.get(cat, 0.0)
        t_ms = t * 1000.0
        frac = t / max(cat_total, 1e-15)
        label = CATEGORY_LABELS.get(cat, cat)
        print(f"  {label:<38s} {t_ms:10.1f} {frac:10.1%} {t_ms/n_iters:10.3f}")

    # Unclassified
    classified_total = sum(categories.get(c, 0.0) for c in CATEGORY_ORDER)
    unclassified = total_wall - classified_total
    if unclassified > 0:
        u_ms = unclassified * 1000.0
        frac = unclassified / max(total_wall, 1e-15)
        print(f"  {'(other / Python overhead)':<38s} {u_ms:10.1f} {frac:10.1%} {u_ms/n_iters:10.3f}")

    print(f"  {'─' * 38} {'─' * 10} {'─' * 10} {'─' * 10}")
    print(f"  {'TOTAL (profiled categories)':<38s} {cat_total*1000:10.1f} {1.0:10.1%} {cat_total*1000/n_iters:10.3f}")
    print(f"{'=' * 72}")


def print_comparison(
    direct_wall: float,
    dmci_wall: float,
    direct_cats: dict[str, float],
    dmci_cats: dict[str, float],
    n_iters: int,
):
    """Print the overhead decomposition comparison table."""
    ratio = dmci_wall / max(direct_wall, 1e-15)

    print(f"\n{'=' * 72}")
    print(f"  OVERHEAD DECOMPOSITION: DMCI / Direct")
    print(f"{'=' * 72}")
    print(f"  Direct:  {direct_wall*1000:.1f} ms total  ({direct_wall*1000/n_iters:.3f} ms/iter)")
    print(f"  DMCI:    {dmci_wall*1000:.1f} ms total  ({dmci_wall*1000/n_iters:.3f} ms/iter)")
    print(f"  Ratio:   {ratio:.1f}x")
    print(f"{'─' * 72}")

    overhead = dmci_wall - direct_wall
    overhead_ms = overhead * 1000.0
    print(f"  Overhead: {overhead_ms:.1f} ms ({overhead_ms/n_iters:.3f} ms/iter)")
    print()
    print(f"  {'Component':<38s} {'Direct':>8s} {'DMCI':>8s} {'Delta':>8s} {'% of OH':>8s}")
    print(f"  {'─' * 38} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 8}")

    for cat in CATEGORY_ORDER:
        d_ms = direct_cats.get(cat, 0.0) * 1000.0
        m_ms = dmci_cats.get(cat, 0.0) * 1000.0
        delta = m_ms - d_ms
        pct = delta / max(overhead_ms, 1e-15)
        label = CATEGORY_LABELS.get(cat, cat)
        print(f"  {label:<38s} {d_ms:8.1f} {m_ms:8.1f} {delta:+8.1f} {pct:8.1%}")

    # Unclassified
    d_classified = sum(direct_cats.get(c, 0.0) for c in CATEGORY_ORDER)
    m_classified = sum(dmci_cats.get(c, 0.0) for c in CATEGORY_ORDER)
    d_other = direct_wall - d_classified
    m_other = dmci_wall - m_classified
    delta_other = (m_other - d_other) * 1000.0
    pct_other = delta_other / max(overhead_ms, 1e-15)
    print(f"  {'(other / Python overhead)':<38s} {d_other*1000:8.1f} {m_other*1000:8.1f} {delta_other:+8.1f} {pct_other:8.1%}")

    print(f"{'=' * 72}")


# ---------------------------------------------------------------------------
# Top-level stats dump
# ---------------------------------------------------------------------------

def dump_top_functions(stats: pstats.Stats, mode_name: str, n: int = 30):
    """Print the top N functions by tottime for debugging."""
    print(f"\n--- Top {n} functions by tottime: {mode_name} ---")
    stats.sort_stats("tottime")
    stats.print_stats(n)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Profile DMCI overhead decomposition"
    )
    parser.add_argument(
        "--program", default="multi_const",
        choices=list(PROGRAMS.keys()),
        help="Program to profile (default: multi_const)",
    )
    parser.add_argument(
        "--n-iters", type=int, default=100,
        help="Number of forward+backward iterations (default: 100)",
    )
    parser.add_argument(
        "--dump-top", type=int, default=0,
        help="Dump top N functions by tottime (0 to skip, default: 0)",
    )
    args = parser.parse_args()

    prog = PROGRAMS[args.program]
    n_iters = args.n_iters

    print(f"Program: {args.program}")
    print(f"Expression: {prog['expr']}")
    print(f"Iterations: {n_iters}")
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Device: cpu")
    sys.setrecursionlimit(5000)

    # --- Compile both modes ---
    print("\nCompiling direct mode...")
    t0 = time.perf_counter()
    all_names = _input_names(prog)
    direct_inputs = {n: None for n in all_names}
    direct_graph = compile_program(
        _build_direct_source(prog), inputs=direct_inputs, prelude=True
    )
    t_direct_compile = time.perf_counter() - t0
    print(f"  Direct compilation: {t_direct_compile:.2f}s")

    print("Compiling DMCI mode...")
    t0 = time.perf_counter()
    dmci_source = _build_dmci_source(prog)
    dmci_graph = compile_program(
        dmci_source, inputs=direct_inputs, prelude=True
    )
    t_dmci_compile = time.perf_counter() - t0
    print(f"  DMCI compilation:   {t_dmci_compile:.2f}s")
    print(f"  Compile ratio:      {t_dmci_compile / max(t_direct_compile, 1e-9):.1f}x")

    # --- Profile direct mode ---
    print(f"\nProfiling direct mode ({n_iters} iters)...")
    direct_wall, direct_fwd, direct_bwd, direct_stats = profile_mode(
        "Direct", direct_graph, all_names, prog["params"], n_iters
    )
    direct_cats = aggregate_categories(direct_stats)

    # --- Profile DMCI mode ---
    print(f"Profiling DMCI mode ({n_iters} iters)...")
    dmci_wall, dmci_fwd, dmci_bwd, dmci_stats = profile_mode(
        "DMCI", dmci_graph, all_names, prog["params"], n_iters
    )
    dmci_cats = aggregate_categories(dmci_stats)

    # --- Output tables ---
    print_table("Direct", direct_wall, direct_fwd, direct_bwd, direct_cats, n_iters)
    print_table("DMCI", dmci_wall, dmci_fwd, dmci_bwd, dmci_cats, n_iters)
    print_comparison(direct_wall, dmci_wall, direct_cats, dmci_cats, n_iters)

    # Optional: dump raw top functions
    if args.dump_top > 0:
        dump_top_functions(direct_stats, "Direct", args.dump_top)
        dump_top_functions(dmci_stats, "DMCI", args.dump_top)


if __name__ == "__main__":
    main()
