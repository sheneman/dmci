############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# test_trampoline.py: Integration tests for trampoline: tail calls through closures use O(1) stack.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Integration tests for trampoline: tail calls through closures use O(1) stack."""

import pytest

from neural_compiler.compiler import compile_scheme, compile_program
from neural_compiler.evaluator import evaluate
from neural_compiler.runtime.tagged_value import unwrap_number


def _eval_tagged(source, inputs=None, max_iter=10000):
    inputs = inputs or {}
    input_decl = {k: None for k in inputs}
    graph = compile_scheme(source, inputs=input_decl)
    return evaluate(graph, inputs, max_iter=max_iter)


def _eval_program(source, inputs=None, max_iter=10000):
    inputs = inputs or {}
    input_decl = {k: None for k in inputs}
    graph = compile_program(source, inputs=input_decl)
    return evaluate(graph, inputs, max_iter=max_iter)


class TestTrampolineCorrectness:
    """Basic correctness: tail calls through closures produce correct results."""

    def test_self_call_via_closure(self):
        result = _eval_tagged("""
            (let ((f (lambda (self n)
                       (if (= n 0) 42 (self self (- n 1))))))
              (f f (car (cons 10 0))))
        """)
        assert unwrap_number(result).item() == pytest.approx(42.0)

    def test_accumulator_via_closure(self):
        result = _eval_tagged("""
            (let ((sum (lambda (self n acc)
                         (if (= n 0) acc (self self (- n 1) (+ acc n))))))
              (sum sum (car (cons 100 0)) (car (cons 0 0))))
        """)
        assert unwrap_number(result).item() == pytest.approx(5050.0)

    def test_closure_tail_call_in_else_branch(self):
        result = _eval_tagged("""
            (let ((countdown (lambda (self n)
                               (if (= n 0) 0 (self self (- n 1))))))
              (countdown countdown (car (cons 50 0))))
        """)
        assert unwrap_number(result).item() == pytest.approx(0.0)

    def test_closure_tail_call_in_then_branch(self):
        result = _eval_tagged("""
            (let ((f (lambda (self n)
                       (if (> n 0) (self self (- n 1)) n))))
              (f f (car (cons 20 0))))
        """)
        assert unwrap_number(result).item() == pytest.approx(0.0)

    def test_higher_order_tail_call(self):
        result = _eval_tagged("""
            (let ((apply-f (lambda (f x) (f x))))
              (apply-f (lambda (n) (+ n (car (cons 1 0)))) (car (cons 5 0))))
        """)
        assert unwrap_number(result).item() == pytest.approx(6.0)


class TestTrampolineLargeDepth:
    """Programs that would exhaust max_depth without the trampoline."""

    def test_deep_self_recursion_via_closure(self):
        result = _eval_tagged("""
            (let ((f (lambda (self n)
                       (if (= n 0) 0 (self self (- n 1))))))
              (f f (car (cons 50000 0))))
        """, max_iter=100000)
        assert unwrap_number(result).item() == pytest.approx(0.0)

    def test_deep_accumulator_via_closure(self):
        result = _eval_tagged("""
            (let ((sum (lambda (self n acc)
                         (if (= n 0) acc (self self (- n 1) (+ acc n))))))
              (sum sum (car (cons 1000 0)) (car (cons 0 0))))
        """, max_iter=100000)
        assert unwrap_number(result).item() == pytest.approx(500500.0)

    def test_mutual_recursion_via_closures(self):
        result = _eval_tagged("""
            (let ((even? (lambda (e o n)
                           (if (= n 0) 1 (o e o (- n 1))))))
              (let ((odd? (lambda (e o n)
                            (if (= n 0) 0 (e e o (- n 1))))))
                (even? even? odd? (car (cons 10000 0)))))
        """, max_iter=100000)
        assert unwrap_number(result).item() == pytest.approx(1.0)

    def test_mutual_recursion_odd(self):
        result = _eval_tagged("""
            (let ((even? (lambda (e o n)
                           (if (= n 0) 1 (o e o (- n 1))))))
              (let ((odd? (lambda (e o n)
                            (if (= n 0) 0 (e e o (- n 1))))))
                (odd? even? odd? (car (cons 10001 0)))))
        """, max_iter=100000)
        assert unwrap_number(result).item() == pytest.approx(1.0)


class TestTrampolineWithDefine:
    """Trampoline works with define-based programs that create closures."""

    def test_define_returns_closure_called_in_tail(self):
        result = _eval_program("""
            (define (make-stepper step)
              (lambda (self n acc)
                (if (= n 0) acc (self self (- n 1) (+ acc step)))))
            (let ((f (make-stepper 3)))
              (f f 100 0))
        """)
        assert unwrap_number(result).item() == pytest.approx(300.0)


class TestTrampolineBackwardCompat:
    """Existing closure and recursion patterns must not regress."""

    def test_immediate_lambda(self):
        result = _eval_tagged("((lambda (x) (+ x 1)) (car (cons 5 0)))")
        assert unwrap_number(result).item() == pytest.approx(6.0)

    def test_closure_capture(self):
        result = _eval_tagged("""
            (let ((x (car (cons 10 0))))
              ((lambda (y) (+ x y)) (car (cons 5 0))))
        """)
        assert unwrap_number(result).item() == pytest.approx(15.0)

    def test_returned_closure(self):
        result = _eval_tagged("""
            (let ((make-adder (lambda (x) (lambda (y) (+ x y)))))
              ((make-adder (car (cons 10 0))) (car (cons 5 0))))
        """)
        assert unwrap_number(result).item() == pytest.approx(15.0)

    def test_compose(self):
        result = _eval_tagged("""
            (let ((compose (lambda (f g) (lambda (x) (f (g x))))))
              (let ((add1 (lambda (n) (+ n (car (cons 1 0))))))
                (let ((double (lambda (n) (+ n n))))
                  ((compose add1 double) (car (cons 5 0))))))
        """)
        assert unwrap_number(result).item() == pytest.approx(11.0)

    def test_letrec_recursion_unchanged(self):
        result = _eval_tagged("""
            (letrec ((fib (lambda (n)
                       (if (< n 2) n (+ (fib (- n 1)) (fib (- n 2)))))))
              (fib (car (cons 10 0))))
        """)
        assert unwrap_number(result).item() == pytest.approx(55.0)
