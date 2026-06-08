############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# test_batched_dmci.py: Batched DMCI: the heap-using meta-circular interpreter evaluated over a batch. The batched evaluator was...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Batched DMCI: the heap-using meta-circular interpreter evaluated over a batch.

The batched evaluator was originally heap-free and raised ``NotImplementedError`` on
``quote_const``, so the compiled interpreter (DMCI) could only be run one input at a
time. As of v1.1.7 a heap-using tagged graph batches natively: passing ``(N, VALUE_DIM)``
tagged inputs runs a single heap-backed interpreter walk over the whole batch (structural
values stay scalar/data-independent; only numeric leaves carry the batch), matching ``N``
sequential walks. v1.1.8 adds a clear error when a program's branch *decision* depends on
the batched input (which the recursive interpreter cannot vectorize).
"""

from pathlib import Path

import pytest
import torch

from neural_compiler.compiler import compile_program
from neural_compiler.evaluator import evaluate, evaluate_batched
from neural_compiler.runtime.tagged_value import make_float, unwrap_number, VALUE_DIM

BOOTSTRAP_DIR = Path(__file__).parent.parent.parent / "bootstrap"
EVALUATOR_SOURCE = (BOOTSTRAP_DIR / "compiler.scm").read_text()


def _env(names):
    return "(list " + " ".join(f"(cons '{n} {n})" for n in names) + ")"


def _dmci_graph(expr, names):
    """Compile compiler.scm + (scheme-eval 'expr env) -> heap-using tagged graph."""
    src = EVALUATOR_SOURCE + f"\n(scheme-eval '{expr} {_env(names)})\n"
    return compile_program(src, inputs={n: None for n in names}, prelude=True)


def _recursive_dmci_graph():
    """A recursive Euler decay interpreted through DMCI: y_{n+1}=y_n+0.1*(-k*y_n), 10 steps."""
    prog = ("(define (step y n) (if (= n 0) y "
            "(step (+ y (* 0.1 (* (- 0 k) y))) (- n 1))))")
    src = (EVALUATOR_SOURCE + "\n(scheme-eval-program\n  (list\n    '" + prog +
           "\n    '(step y0 10))\n  " + _env(["y0", "k"]) + ")\n")
    return compile_program(src, inputs={"y0": None, "k": None}, prelude=True)


def _seq(graph, driver_name, driver_vec, scalars):
    out = []
    for i in range(driver_vec.shape[0]):
        inp = {driver_name: make_float(driver_vec[i])}
        for n, v in scalars.items():
            inp[n] = make_float(torch.tensor(float(v)))
        out.append(unwrap_number(evaluate(graph, inp)))
    return torch.stack(out)


def _batched(graph, driver_name, driver_vec, scalars, via=evaluate):
    inp = {driver_name: make_float(driver_vec)}
    for n, v in scalars.items():
        inp[n] = make_float(torch.tensor(float(v)))
    return unwrap_number(via(graph, inp))


class TestBatchedDMCIArithmetic:
    """Arithmetic program interpreted through DMCI (heap for env + quoted AST)."""

    def test_batched_matches_sequential(self):
        g = _dmci_graph("(* a (exp (* (- 0 b) x)))", ["x", "a", "b"])
        x = torch.linspace(0.1, 5.0, 7)
        scal = {"a": 2.5, "b": 0.8}
        seq = _seq(g, "x", x, scal)
        bat = _batched(g, "x", x, scal)
        assert bat.shape == (7,)
        assert torch.allclose(bat, seq, atol=1e-5)

    def test_evaluate_batched_routes_heap_graph(self):
        # Regression: evaluate_batched used to raise NotImplementedError('quote_const').
        g = _dmci_graph("(* a (exp (* (- 0 b) x)))", ["x", "a", "b"])
        x = torch.linspace(0.1, 5.0, 5)
        bat = _batched(g, "x", x, {"a": 2.5, "b": 0.8}, via=evaluate_batched)
        assert bat.shape == (5,)

    def test_gradients_flow_through_batch(self):
        g = _dmci_graph("(* a (exp (* (- 0 b) x)))", ["x", "a", "b"])
        x = torch.linspace(0.1, 5.0, 6)
        a = torch.tensor(2.5, requires_grad=True)
        b = torch.tensor(0.8, requires_grad=True)
        inp = {"x": make_float(x), "a": make_float(a), "b": make_float(b)}
        (unwrap_number(evaluate(g, inp)) ** 2).mean().backward()
        for p in (a, b):
            assert p.grad is not None and torch.isfinite(p.grad).all()


class TestBatchedDMCIRecursive:
    """Recursive (Euler-step) program interpreted through DMCI: recursion + heap."""

    def test_batched_matches_sequential(self):
        g = _recursive_dmci_graph()
        y0 = torch.linspace(1.0, 5.0, 6)
        seq = _seq(g, "y0", y0, {"k": 0.5})
        bat = _batched(g, "y0", y0, {"k": 0.5})
        assert torch.allclose(bat, seq, atol=1e-5)


class TestBatchedDMCIDataDependentBranch:
    """A branch whose decision depends on the batched input cannot be vectorized."""

    EXPR = "(if (< x 2.0) (* a x) (* b x))"

    def test_disagreeing_branch_raises_clear_error(self):
        g = _dmci_graph(self.EXPR, ["x", "a", "b"])
        inp = {"x": make_float(torch.tensor([0.5, 3.0])),     # one < 2, one > 2
               "a": make_float(torch.tensor(10.0)),
               "b": make_float(torch.tensor(100.0))}
        with pytest.raises(ValueError, match="data-independent control flow"):
            evaluate(g, inp)

    def test_uniform_branch_still_evaluates(self):
        g = _dmci_graph(self.EXPR, ["x", "a", "b"])
        inp = {"x": make_float(torch.tensor([0.5, 1.0, 1.5])),  # all < 2 -> uniform
               "a": make_float(torch.tensor(10.0)),
               "b": make_float(torch.tensor(100.0))}
        out = unwrap_number(evaluate(g, inp))
        assert torch.allclose(out, torch.tensor([5.0, 10.0, 15.0]), atol=1e-4)

    def test_singleton_unaffected(self):
        g = _dmci_graph(self.EXPR, ["x", "a", "b"])
        out = unwrap_number(evaluate(g, {
            "x": make_float(torch.tensor(3.0)),
            "a": make_float(torch.tensor(10.0)),
            "b": make_float(torch.tensor(100.0))}))
        assert out.item() == pytest.approx(300.0)
