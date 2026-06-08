############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# test_bare_interpreter.py: Bare, distributable meta-circular interpreter: the program is supplied as runtime data....
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Bare, distributable meta-circular interpreter: the program is supplied as runtime data.

``compile_interpreter()`` compiles the evaluator ONCE with ``program`` and ``env`` as runtime
inputs, so a single serializable, backend-agnostic graph runs ANY program handed to it -- no
per-program recompilation. These tests cover: correctness vs direct compilation, running
different programs through one graph, multi-form programs with ``define``, gradient flow to
bindings, batched bindings, and a ``.ncg`` save/load round-trip.
"""

import math

import pytest
import torch

from neural_compiler import (
    compile_interpreter,
    evaluate_program,
    save_compiled,
    load_compiled,
)
from neural_compiler.runtime.tagged_value import unwrap_number


@pytest.fixture(scope="module")
def interp():
    return compile_interpreter()


class TestBareInterpreter:
    def test_inputs_are_program_and_env(self, interp):
        assert set(interp.input_names) == {"program", "env"}

    def test_simple_program(self, interp):
        y = unwrap_number(evaluate_program(
            interp, "(* a (exp (* (- 0 b) x)))", {"x": 1.5, "a": 2.5, "b": 0.8}))
        assert y.item() == pytest.approx(2.5 * math.exp(-0.8 * 1.5), abs=1e-4)

    def test_different_program_same_graph(self, interp):
        # No recompilation: a different program runs through the same interpreter graph.
        y = unwrap_number(evaluate_program(
            interp, "(+ (* a x) (* b (* x x)))", {"x": 2.0, "a": 3.0, "b": 1.5}))
        assert y.item() == pytest.approx(3.0 * 2.0 + 1.5 * 4.0, abs=1e-4)

    def test_multiform_program_with_define(self, interp):
        y = unwrap_number(evaluate_program(
            interp, "(define (sq y) (* y y)) (+ (sq x) a)", {"x": 3.0, "a": 1.0}))
        assert y.item() == pytest.approx(10.0, abs=1e-4)

    def test_gradients_flow_to_bindings(self, interp):
        a = torch.tensor(2.5, requires_grad=True)
        b = torch.tensor(0.8, requires_grad=True)
        unwrap_number(evaluate_program(
            interp, "(* a (exp (* (- 0 b) x)))", {"x": 1.5, "a": a, "b": b})).backward()
        assert a.grad is not None and torch.isfinite(a.grad).all()
        assert b.grad is not None and torch.isfinite(b.grad).all()

    def test_batched_bindings(self, interp):
        x = torch.tensor([0.5, 1.0, 1.5, 2.0])
        out = unwrap_number(evaluate_program(
            interp, "(* a (exp (* (- 0 b) x)))", {"x": x, "a": 2.5, "b": 0.8}))
        assert out.shape == (4,)
        assert torch.allclose(out, 2.5 * torch.exp(-0.8 * x), atol=1e-4)

    def test_ncg_roundtrip_runs_any_program(self, interp, tmp_path):
        p = tmp_path / "interp.ncg"
        save_compiled(interp, str(p), source="bare DMCI interpreter")
        loaded = load_compiled(str(p))                      # no Scheme toolchain / recompile
        assert set(loaded.input_names) == {"program", "env"}
        y1 = unwrap_number(evaluate_program(loaded, "(* a x)", {"x": 4.0, "a": 2.0}))
        y2 = unwrap_number(evaluate_program(loaded, "(+ x a)", {"x": 4.0, "a": 2.0}))
        assert y1.item() == pytest.approx(8.0, abs=1e-4)
        assert y2.item() == pytest.approx(6.0, abs=1e-4)


class TestBareInterpreterMatchesDirect:
    @pytest.mark.parametrize("program,bindings", [
        ("(* a (exp (* (- 0 b) x)))", {"x": 1.2, "a": 2.0, "b": 0.5}),
        ("(/ a (+ 1.0 (exp (* (- 0 b) (- x c)))))", {"x": 2.0, "a": 5.0, "b": 2.0, "c": 2.5}),
        ("(+ (* a x) b)", {"x": 3.0, "a": 1.5, "b": 0.5}),
    ])
    def test_matches_direct_compilation(self, interp, program, bindings):
        from neural_compiler.compiler import compile_program
        from neural_compiler.evaluator import evaluate
        gd = compile_program(program, inputs={k: None for k in bindings}, prelude=False)
        direct = evaluate(gd, {k: float(v) for k, v in bindings.items()})
        direct = direct if isinstance(direct, float) else float(direct)
        via_interp = float(unwrap_number(evaluate_program(interp, program, bindings)))
        assert via_interp == pytest.approx(direct, abs=1e-4)
