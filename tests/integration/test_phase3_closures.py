############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# test_phase3_closures.py: Phase 3 integration tests: closures, define, higher-order functions, gradient flow.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Phase 3 integration tests: closures, define, higher-order functions, gradient flow."""

import pytest
import torch

from neural_compiler.compiler import compile_scheme, compile_program, run_program
from neural_compiler.evaluator import evaluate
from neural_compiler.runtime.tagged_value import (
    VALUE_DIM, CLOSURE, PAIR,
    type_index, unwrap_number, make_float,
)


def _eval_tagged(source, inputs=None):
    inputs = inputs or {}
    input_decl = {k: None for k in inputs}
    graph = compile_scheme(source, inputs=input_decl)
    assert graph.uses_tagged_values
    return evaluate(graph, inputs)


class TestLambdaApplication:
    def test_immediate_application(self):
        result = _eval_tagged("((lambda (x) (+ x 1)) (car (cons 5 0)))")
        assert unwrap_number(result).item() == pytest.approx(6.0)

    def test_lambda_two_args(self):
        result = _eval_tagged("((lambda (x y) (+ x y)) (car (cons 3 0)) (car (cons 4 0)))")
        assert unwrap_number(result).item() == pytest.approx(7.0)

    def test_lambda_no_args(self):
        result = _eval_tagged("((lambda () (cons 42 0)))")
        assert type_index(result) == PAIR

    def test_lambda_is_closure(self):
        result = _eval_tagged("(procedure? (lambda (x) x))")
        assert unwrap_number(result).item() == 1.0


class TestClosureCapture:
    def test_capture_one_var(self):
        result = _eval_tagged("""
            (let ((x (car (cons 10 0))))
              ((lambda (y) (+ x y)) (car (cons 5 0))))
        """)
        assert unwrap_number(result).item() == pytest.approx(15.0)

    def test_capture_two_vars(self):
        result = _eval_tagged("""
            (let ((a (car (cons 3 0))))
              (let ((b (car (cons 4 0))))
                ((lambda (c) (+ a (+ b c))) (car (cons 5 0)))))
        """)
        assert unwrap_number(result).item() == pytest.approx(12.0)

    def test_let_bound_closure(self):
        result = _eval_tagged("""
            (let ((x (car (cons 10 0))))
              (let ((f (lambda (y) (+ x y))))
                (f (car (cons 5 0)))))
        """)
        assert unwrap_number(result).item() == pytest.approx(15.0)

    def test_closure_captures_cons(self):
        result = _eval_tagged("""
            (let ((p (cons 10 20)))
              (let ((get-sum (lambda () (+ (car p) (cdr p)))))
                (get-sum)))
        """)
        assert unwrap_number(result).item() == pytest.approx(30.0)


class TestHigherOrderFunctions:
    def test_apply_twice(self):
        result = _eval_tagged("""
            (let ((apply-twice (lambda (f x) (f (f x)))))
              (let ((add3 (lambda (n) (+ n (car (cons 3 0))))))
                (apply-twice add3 (car (cons 10 0)))))
        """)
        assert unwrap_number(result).item() == pytest.approx(16.0)

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

    def test_closure_stored_in_cons(self):
        result = _eval_tagged("""
            (let ((f (lambda (x) (+ x (car (cons 1 0))))))
              (let ((p (cons f 0)))
                ((car p) (car (cons 5 0)))))
        """)
        assert unwrap_number(result).item() == pytest.approx(6.0)

    def test_closure_from_cond(self):
        result = _eval_tagged("""
            (let ((pick (lambda (flag)
                    (cond
                      ((null? flag) (lambda (x) (+ x (car (cons 1 0)))))
                      (#t (lambda (x) (+ x (car (cons 10 0)))))))))
              ((pick '()) (car (cons 5 0))))
        """)
        assert unwrap_number(result).item() == pytest.approx(6.0)


class TestDefineProgram:
    def test_simple_define(self):
        result = run_program("""
            (define x 10)
            (+ x 5)
        """)
        assert result == pytest.approx(15.0)

    def test_function_define(self):
        result = run_program("""
            (define (double x) (+ x x))
            (double 7)
        """)
        assert result == pytest.approx(14.0)

    def test_recursive_define(self):
        result = run_program("""
            (define (factorial n)
              (if (= n 0) 1 (* n (factorial (- n 1)))))
            (factorial 5)
        """)
        assert result == pytest.approx(120.0)

    def test_mutual_recursion_define(self):
        result = run_program("""
            (define (even? n)
              (if (= n 0) 1 (odd? (- n 1))))
            (define (odd? n)
              (if (= n 0) 0 (even? (- n 1))))
            (even? 10)
        """)
        assert result == pytest.approx(1.0)

    def test_multiple_defines(self):
        result = run_program("""
            (define pi 3.14159)
            (define (circle-area r) (* pi (* r r)))
            (circle-area 2)
        """)
        assert result == pytest.approx(12.56636)

    def test_define_with_cons(self):
        graph = compile_program("""
            (define (make-pair a b) (cons a b))
            (car (make-pair 10 20))
        """)
        result = evaluate(graph, {})
        assert unwrap_number(result).item() == pytest.approx(10.0)


class TestBeginSequencing:
    def test_begin_returns_last(self):
        result = run_program("""
            (define (f x)
              (begin
                (+ x 1)
                (+ x 2)
                (+ x 3)))
            (f 10)
        """)
        assert result == pytest.approx(13.0)


class TestGradientThroughClosure:
    def test_grad_through_lambda(self):
        graph = compile_scheme("((lambda (y) (+ y y)) x)", inputs={"x": None})
        x = torch.tensor(3.0, requires_grad=True)
        x_tagged = make_float(x)
        result = evaluate(graph, {"x": x_tagged})
        loss = unwrap_number(result)
        loss.backward()
        assert x.grad is not None
        assert x.grad.item() == pytest.approx(2.0)

    def test_grad_through_closure_capture(self):
        graph = compile_scheme(
            "(let ((f (lambda (y) (+ x y)))) (f (car (cons 1 0))))",
            inputs={"x": None},
        )
        x = torch.tensor(5.0, requires_grad=True)
        x_tagged = make_float(x)
        result = evaluate(graph, {"x": x_tagged})
        loss = unwrap_number(result)
        loss.backward()
        assert x.grad is not None
        assert x.grad.item() == pytest.approx(1.0)

    def test_grad_through_returned_closure(self):
        graph = compile_scheme(
            "(let ((make-f (lambda (a) (lambda (b) (+ a b))))) ((make-f x) (car (cons 1 0))))",
            inputs={"x": None},
        )
        x = torch.tensor(5.0, requires_grad=True)
        x_tagged = make_float(x)
        result = evaluate(graph, {"x": x_tagged})
        loss = unwrap_number(result)
        loss.backward()
        assert x.grad is not None
        assert x.grad.item() == pytest.approx(1.0)


class TestBackwardCompatibility:
    def test_letrec_still_works(self):
        result = run_program("""
            (define (fib n)
              (if (< n 2) n (+ (fib (- n 1)) (fib (- n 2)))))
            (fib 10)
        """)
        assert result == pytest.approx(55.0)

    def test_numeric_program_untouched(self):
        from neural_compiler.compiler import run_scheme
        result = run_scheme("(+ (* 3 4) 5)")
        assert result == pytest.approx(17.0)

    def test_loop_still_works(self):
        from neural_compiler.compiler import run_scheme
        result = run_scheme(
            "(loop ((n 10) (acc 1)) (if (= n 0) acc (recur (- n 1) (* acc n))))"
        )
        assert result == pytest.approx(3628800.0)
