############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# test_bootstrap.py: Bootstrap tests: compile a Scheme evaluator, then run programs through it. This is the core self-hosting test:...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Bootstrap tests: compile a Scheme evaluator, then run programs through it.

This is the core self-hosting test: a Scheme interpreter is compiled by the
Python neural_compiler into a differentiable PyTorch program. That PyTorch
program then evaluates Scheme expressions.
"""

import pytest
import torch
from pathlib import Path

from neural_compiler.compiler import compile_program
from neural_compiler.evaluator import evaluate
from neural_compiler.runtime.tagged_value import (
    type_index, unwrap_number, make_float, PAIR, NIL, SYMBOL,
)


BOOTSTRAP_DIR = Path(__file__).parent.parent.parent / "bootstrap"
EVALUATOR_SOURCE = (BOOTSTRAP_DIR / "compiler.scm").read_text()


def _run_in_evaluator(program_sexpr: str) -> object:
    """Compile the Scheme evaluator, then use it to evaluate a program."""
    source = EVALUATOR_SOURCE + f"\n(scheme-eval '{program_sexpr} '())\n"
    graph = compile_program(source, prelude=True)
    return evaluate(graph, {})


class TestBootstrapArithmetic:
    def test_number(self):
        result = _run_in_evaluator("42")
        assert unwrap_number(result).item() == pytest.approx(42.0)

    def test_addition(self):
        result = _run_in_evaluator("(+ 3 4)")
        assert unwrap_number(result).item() == pytest.approx(7.0)

    def test_nested_arithmetic(self):
        result = _run_in_evaluator("(+ (* 3 4) 5)")
        assert unwrap_number(result).item() == pytest.approx(17.0)

    def test_subtraction(self):
        result = _run_in_evaluator("(- 10 3)")
        assert unwrap_number(result).item() == pytest.approx(7.0)

    def test_multiplication(self):
        result = _run_in_evaluator("(* 6 7)")
        assert unwrap_number(result).item() == pytest.approx(42.0)


class TestBootstrapConditional:
    def test_if_true(self):
        result = _run_in_evaluator("(if (> 5 3) 1 0)")
        assert unwrap_number(result).item() == pytest.approx(1.0)

    def test_if_false(self):
        result = _run_in_evaluator("(if (< 5 3) 1 0)")
        assert unwrap_number(result).item() == pytest.approx(0.0)


class TestBootstrapLet:
    def test_simple_let(self):
        result = _run_in_evaluator("(let ((x 10)) (+ x 5))")
        assert unwrap_number(result).item() == pytest.approx(15.0)

    def test_nested_let(self):
        result = _run_in_evaluator("(let ((x 3)) (let ((y 4)) (+ x y)))")
        assert unwrap_number(result).item() == pytest.approx(7.0)


class TestBootstrapLambda:
    def test_lambda_application(self):
        result = _run_in_evaluator("((lambda (x) (+ x 1)) 5)")
        assert unwrap_number(result).item() == pytest.approx(6.0)

    def test_closure(self):
        result = _run_in_evaluator(
            "(let ((make-adder (lambda (x) (lambda (y) (+ x y))))) ((make-adder 10) 5))"
        )
        assert unwrap_number(result).item() == pytest.approx(15.0)

    def test_higher_order(self):
        result = _run_in_evaluator(
            "(let ((apply-fn (lambda (f x) (f x)))) (apply-fn (lambda (n) (* n 2)) 7))"
        )
        assert unwrap_number(result).item() == pytest.approx(14.0)


class TestBootstrapLists:
    def test_cons_car(self):
        result = _run_in_evaluator("(car (cons 1 2))")
        assert unwrap_number(result).item() == pytest.approx(1.0)

    def test_cons_cdr(self):
        result = _run_in_evaluator("(cdr (cons 1 2))")
        assert unwrap_number(result).item() == pytest.approx(2.0)

    def test_null_check(self):
        result = _run_in_evaluator("(null? '())")
        assert unwrap_number(result).item() == 1.0

    def test_pair_check(self):
        result = _run_in_evaluator("(pair? (cons 1 2))")
        assert unwrap_number(result).item() == 1.0

    def test_quote(self):
        result = _run_in_evaluator("(car '(10 20 30))")
        assert unwrap_number(result).item() == pytest.approx(10.0)


class TestBootstrapGradient:
    def test_gradient_through_evaluator(self):
        """Verify gradients flow through the bootstrapped evaluator."""
        source = EVALUATOR_SOURCE + "\n(scheme-eval (list '+ x '1) (list (cons 'x x)))\n"
        graph = compile_program(source, inputs={"x": None}, prelude=True)
        x = torch.tensor(5.0, requires_grad=True)
        x_tagged = make_float(x)
        result = evaluate(graph, {"x": x_tagged})
        loss = unwrap_number(result)
        assert loss.item() == pytest.approx(6.0)
        loss.backward()
        assert x.grad is not None
        assert x.grad.item() == pytest.approx(1.0)

    def test_gradient_multiply(self):
        """Gradient of x*3 through bootstrapped evaluator."""
        source = EVALUATOR_SOURCE + "\n(scheme-eval (list '* x '3) (list (cons 'x x)))\n"
        graph = compile_program(source, inputs={"x": None}, prelude=True)
        x = torch.tensor(4.0, requires_grad=True)
        x_tagged = make_float(x)
        result = evaluate(graph, {"x": x_tagged})
        loss = unwrap_number(result)
        assert loss.item() == pytest.approx(12.0)
        loss.backward()
        assert x.grad is not None
        assert x.grad.item() == pytest.approx(3.0)


class TestBootstrapComplex:
    def test_factorial_via_lambda(self):
        """Factorial computed inside the bootstrapped evaluator using closures."""
        result = _run_in_evaluator("""
            (let ((fact (lambda (self n)
                    (if (= n 0) 1 (* n (self self (- n 1)))))))
              (fact fact 5))
        """)
        assert unwrap_number(result).item() == pytest.approx(120.0)

    def test_sum_list(self):
        """Sum a list inside the bootstrapped evaluator."""
        result = _run_in_evaluator("""
            (let ((sum (lambda (self lst)
                    (if (null? lst)
                      0
                      (+ (car lst) (self self (cdr lst)))))))
              (sum sum (list 1 2 3 4 5)))
        """)
        assert unwrap_number(result).item() == pytest.approx(15.0)
