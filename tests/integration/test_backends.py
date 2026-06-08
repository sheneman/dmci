############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# test_backends.py: Multi-backend integration tests. Verifies that NumPy, JAX, and CuPy backends produce the same results as the...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Multi-backend integration tests.

Verifies that NumPy, JAX, and CuPy backends produce the same results
as the default PyTorch backend across scalar, tagged, and self-hosting
programs. Covers all evaluator code paths exercised by the torch tests.
"""

import math
import pytest
from pathlib import Path

from neural_compiler.compiler import compile_scheme, compile_program
from neural_compiler.evaluator import evaluate

BACKENDS = ["numpy"]

try:
    import jax  # noqa: F401
    BACKENDS.append("jax")
except ImportError:
    pass

try:
    import cupy  # noqa: F401
    BACKENDS.append("cupy")
except ImportError:
    pass


BOOTSTRAP_DIR = Path(__file__).parent.parent.parent / "bootstrap"
EVALUATOR_SOURCE = (BOOTSTRAP_DIR / "compiler.scm").read_text()


def _unwrap(result, backend_name):
    """Extract numeric value from a tagged result array."""
    if hasattr(result, "__len__") and len(result) == 14:
        return float(result[10])
    return float(result)


def _unwrap_bool(result):
    """Extract boolean truth from a tagged bool result (payload[0])."""
    if hasattr(result, "__len__") and len(result) == 14:
        return float(result[10])
    return float(result)


# ================================================================== #
# Scalar arithmetic
# ================================================================== #

class TestScalarArithmetic:
    @pytest.mark.parametrize("backend", BACKENDS)
    def test_add(self, backend):
        graph = compile_scheme("(+ 3 4)")
        assert evaluate(graph, backend=backend) == pytest.approx(7.0)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_subtract(self, backend):
        graph = compile_scheme("(- 10 3)")
        assert evaluate(graph, backend=backend) == pytest.approx(7.0)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_multiply(self, backend):
        graph = compile_scheme("(* 6 7)")
        assert evaluate(graph, backend=backend) == pytest.approx(42.0)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_divide(self, backend):
        graph = compile_scheme("(/ 15 3)")
        assert evaluate(graph, backend=backend) == pytest.approx(5.0)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_nested(self, backend):
        graph = compile_scheme("(+ (* 3 4) (- 10 5))")
        assert evaluate(graph, backend=backend) == pytest.approx(17.0)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_with_inputs(self, backend):
        graph = compile_scheme("(+ (* 3 x) y)", inputs={"x": None, "y": None})
        assert evaluate(graph, {"x": 4.0, "y": 2.0}, backend=backend) == pytest.approx(14.0)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_comparison_gt(self, backend):
        graph = compile_scheme("(if (> 5 3) 1 0)")
        assert evaluate(graph, backend=backend) == pytest.approx(1.0)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_comparison_lt(self, backend):
        graph = compile_scheme("(if (< 5 3) 1 0)")
        assert evaluate(graph, backend=backend) == pytest.approx(0.0)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_comparison_eq(self, backend):
        graph = compile_scheme("(if (= 5 5) 1 0)")
        assert evaluate(graph, backend=backend) == pytest.approx(1.0)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_math_functions(self, backend):
        graph = compile_scheme("(+ (sin 0) (cos 0))")
        assert evaluate(graph, backend=backend) == pytest.approx(1.0)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_logic_not(self, backend):
        graph = compile_scheme("(not 0)")
        assert evaluate(graph, backend=backend) == pytest.approx(1.0)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_logic_and(self, backend):
        graph = compile_scheme("(and 1 1)")
        assert evaluate(graph, backend=backend) == pytest.approx(1.0)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_logic_or(self, backend):
        graph = compile_scheme("(or 0 1)")
        assert evaluate(graph, backend=backend) == pytest.approx(1.0)


# ================================================================== #
# Loops (TCO)
# ================================================================== #

class TestLoops:
    @pytest.mark.parametrize("backend", BACKENDS)
    def test_factorial(self, backend):
        source = """
        (letrec ((fact (lambda (n acc)
                   (if (= n 0) acc (fact (- n 1) (* acc n))))))
          (fact 5 1))
        """
        graph = compile_scheme(source)
        assert evaluate(graph, backend=backend) == pytest.approx(120.0)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_fibonacci(self, backend):
        source = """
        (letrec ((fib (lambda (n a b)
                   (if (= n 0) a (fib (- n 1) b (+ a b))))))
          (fib 10 0 1))
        """
        graph = compile_scheme(source)
        assert evaluate(graph, backend=backend) == pytest.approx(55.0)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_loop_with_input(self, backend):
        source = """
        (letrec ((fact (lambda (n acc)
                   (if (= n 0) acc (fact (- n 1) (* acc n))))))
          (fact x 1))
        """
        graph = compile_scheme(source, inputs={"x": None})
        assert evaluate(graph, {"x": 6.0}, backend=backend) == pytest.approx(720.0)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_loop_with_let_capture(self, backend):
        source = """
        (let ((scale 3))
          (letrec ((mul-sum (lambda (n acc)
                     (if (= n 0) acc (mul-sum (- n 1) (+ acc (* n scale)))))))
            (mul-sum 4 0)))
        """
        graph = compile_scheme(source)
        assert evaluate(graph, backend=backend) == pytest.approx(30.0)


# ================================================================== #
# Tagged values: cons, lists, quote
# ================================================================== #

class TestTaggedValues:
    @pytest.mark.parametrize("backend", BACKENDS)
    def test_cons_car(self, backend):
        graph = compile_program("(car (cons 1 2))", prelude=True)
        assert _unwrap(evaluate(graph, backend=backend), backend) == pytest.approx(1.0)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_cons_cdr(self, backend):
        graph = compile_program("(cdr (cons 1 2))", prelude=True)
        assert _unwrap(evaluate(graph, backend=backend), backend) == pytest.approx(2.0)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_null_check(self, backend):
        graph = compile_program("(null? '())", prelude=True)
        assert _unwrap(evaluate(graph, backend=backend), backend) == pytest.approx(1.0)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_pair_check(self, backend):
        graph = compile_program("(pair? (cons 1 2))", prelude=True)
        assert _unwrap(evaluate(graph, backend=backend), backend) == pytest.approx(1.0)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_list_car(self, backend):
        graph = compile_program("(car '(10 20 30))", prelude=True)
        assert _unwrap(evaluate(graph, backend=backend), backend) == pytest.approx(10.0)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_nested_arithmetic_tagged(self, backend):
        graph = compile_program("(+ (* 3 4) 5)", prelude=True)
        assert _unwrap(evaluate(graph, backend=backend), backend) == pytest.approx(17.0)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_cadr(self, backend):
        graph = compile_program("(car (cdr '(10 20 30)))", prelude=True)
        assert _unwrap(evaluate(graph, backend=backend), backend) == pytest.approx(20.0)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_caddr(self, backend):
        graph = compile_program("(car (cdr (cdr '(10 20 30))))", prelude=True)
        assert _unwrap(evaluate(graph, backend=backend), backend) == pytest.approx(30.0)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_nested_quote(self, backend):
        graph = compile_program("(car (car '((1 2) (3 4))))", prelude=True)
        assert _unwrap(evaluate(graph, backend=backend), backend) == pytest.approx(1.0)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_quote_symbol(self, backend):
        graph = compile_program("(symbol? 'hello)", prelude=True)
        assert _unwrap(evaluate(graph, backend=backend), backend) == pytest.approx(1.0)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_quote_empty_list(self, backend):
        graph = compile_program("(null? '())", prelude=True)
        assert _unwrap(evaluate(graph, backend=backend), backend) == pytest.approx(1.0)


# ================================================================== #
# Type predicates
# ================================================================== #

class TestTypePredicates:
    @pytest.mark.parametrize("backend", BACKENDS)
    def test_number_p_true(self, backend):
        graph = compile_program("(number? 42)", prelude=True)
        assert _unwrap(evaluate(graph, backend=backend), backend) == pytest.approx(1.0)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_number_p_false(self, backend):
        graph = compile_program("(number? '())", prelude=True)
        assert _unwrap(evaluate(graph, backend=backend), backend) == pytest.approx(0.0)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_pair_p_true(self, backend):
        graph = compile_program("(pair? (cons 1 2))", prelude=True)
        assert _unwrap(evaluate(graph, backend=backend), backend) == pytest.approx(1.0)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_pair_p_false(self, backend):
        graph = compile_program("(pair? 42)", prelude=True)
        assert _unwrap(evaluate(graph, backend=backend), backend) == pytest.approx(0.0)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_null_p_true(self, backend):
        graph = compile_program("(null? '())", prelude=True)
        assert _unwrap(evaluate(graph, backend=backend), backend) == pytest.approx(1.0)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_null_p_false(self, backend):
        graph = compile_program("(null? (cons 1 2))", prelude=True)
        assert _unwrap(evaluate(graph, backend=backend), backend) == pytest.approx(0.0)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_symbol_p_true(self, backend):
        graph = compile_program("(symbol? 'foo)", prelude=True)
        assert _unwrap(evaluate(graph, backend=backend), backend) == pytest.approx(1.0)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_symbol_p_false(self, backend):
        graph = compile_program("(symbol? 42)", prelude=True)
        assert _unwrap(evaluate(graph, backend=backend), backend) == pytest.approx(0.0)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_boolean_p_true(self, backend):
        graph = compile_program("(boolean? #t)", prelude=True)
        assert _unwrap(evaluate(graph, backend=backend), backend) == pytest.approx(1.0)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_boolean_p_false(self, backend):
        graph = compile_program("(boolean? 42)", prelude=True)
        assert _unwrap(evaluate(graph, backend=backend), backend) == pytest.approx(0.0)


# ================================================================== #
# Boolean literals
# ================================================================== #

class TestBooleanLiterals:
    @pytest.mark.parametrize("backend", BACKENDS)
    def test_true_literal(self, backend):
        graph = compile_program("(if #t 1 0)", prelude=True)
        assert _unwrap(evaluate(graph, backend=backend), backend) == pytest.approx(1.0)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_false_literal(self, backend):
        graph = compile_program("(if #f 1 0)", prelude=True)
        assert _unwrap(evaluate(graph, backend=backend), backend) == pytest.approx(0.0)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_not_true(self, backend):
        graph = compile_program("(not #t)", prelude=True)
        assert _unwrap(evaluate(graph, backend=backend), backend) == pytest.approx(0.0)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_not_false(self, backend):
        graph = compile_program("(not #f)", prelude=True)
        assert _unwrap(evaluate(graph, backend=backend), backend) == pytest.approx(1.0)


# ================================================================== #
# Equality operations
# ================================================================== #

class TestEquality:
    @pytest.mark.parametrize("backend", BACKENDS)
    def test_eq_numbers(self, backend):
        graph = compile_program("(eq? 42 42)", prelude=True)
        assert _unwrap(evaluate(graph, backend=backend), backend) == pytest.approx(1.0)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_eq_numbers_different(self, backend):
        graph = compile_program("(eq? 42 43)", prelude=True)
        assert _unwrap(evaluate(graph, backend=backend), backend) == pytest.approx(0.0)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_eq_symbols(self, backend):
        graph = compile_program("(eq? 'foo 'foo)", prelude=True)
        assert _unwrap(evaluate(graph, backend=backend), backend) == pytest.approx(1.0)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_eq_symbols_different(self, backend):
        graph = compile_program("(eq? 'foo 'bar)", prelude=True)
        assert _unwrap(evaluate(graph, backend=backend), backend) == pytest.approx(0.0)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_eq_nil(self, backend):
        graph = compile_program("(eq? '() '())", prelude=True)
        assert _unwrap(evaluate(graph, backend=backend), backend) == pytest.approx(1.0)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_equal_nested_lists(self, backend):
        graph = compile_program("(equal? '(1 2 3) '(1 2 3))", prelude=True)
        assert _unwrap(evaluate(graph, backend=backend), backend) == pytest.approx(1.0)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_equal_different_lists(self, backend):
        graph = compile_program("(equal? '(1 2 3) '(1 2 4))", prelude=True)
        assert _unwrap(evaluate(graph, backend=backend), backend) == pytest.approx(0.0)


# ================================================================== #
# Cond expressions
# ================================================================== #

class TestCond:
    @pytest.mark.parametrize("backend", BACKENDS)
    def test_cond_first_branch(self, backend):
        source = "(cond ((> 5 3) 10) (#t 20))"
        graph = compile_program(source, prelude=True)
        assert _unwrap(evaluate(graph, backend=backend), backend) == pytest.approx(10.0)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_cond_fallthrough(self, backend):
        source = "(cond ((< 5 3) 10) (#t 20))"
        graph = compile_program(source, prelude=True)
        assert _unwrap(evaluate(graph, backend=backend), backend) == pytest.approx(20.0)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_cond_else(self, backend):
        source = "(cond ((< 1 0) 1) (else 99))"
        graph = compile_program(source, prelude=True)
        assert _unwrap(evaluate(graph, backend=backend), backend) == pytest.approx(99.0)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_cond_multiple_clauses(self, backend):
        source = """
        (cond ((= 1 2) 10)
              ((= 1 3) 20)
              ((= 1 1) 30)
              (#t 40))
        """
        graph = compile_program(source, prelude=True)
        assert _unwrap(evaluate(graph, backend=backend), backend) == pytest.approx(30.0)


# ================================================================== #
# Begin expressions
# ================================================================== #

class TestBegin:
    @pytest.mark.parametrize("backend", BACKENDS)
    def test_begin_returns_last(self, backend):
        source = "(begin 1 2 3)"
        graph = compile_program(source, prelude=True)
        assert _unwrap(evaluate(graph, backend=backend), backend) == pytest.approx(3.0)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_begin_with_side_effects(self, backend):
        source = "(begin (+ 1 2) (* 3 4) (- 10 5))"
        graph = compile_program(source, prelude=True)
        assert _unwrap(evaluate(graph, backend=backend), backend) == pytest.approx(5.0)


# ================================================================== #
# Closures, higher-order functions, define
# ================================================================== #

class TestClosuresAndDefine:
    @pytest.mark.parametrize("backend", BACKENDS)
    def test_lambda(self, backend):
        graph = compile_program("((lambda (x) (+ x 1)) 5)", prelude=True)
        assert _unwrap(evaluate(graph, backend=backend), backend) == pytest.approx(6.0)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_closure(self, backend):
        source = """
        (let ((make-adder (lambda (x) (lambda (y) (+ x y)))))
          ((make-adder 10) 5))
        """
        graph = compile_program(source, prelude=True)
        assert _unwrap(evaluate(graph, backend=backend), backend) == pytest.approx(15.0)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_define_function(self, backend):
        source = """
        (define (double x) (* x 2))
        (double 7)
        """
        graph = compile_program(source, prelude=True)
        assert _unwrap(evaluate(graph, backend=backend), backend) == pytest.approx(14.0)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_recursive_define(self, backend):
        source = """
        (define (fact n)
          (if (= n 0) 1 (* n (fact (- n 1)))))
        (fact 5)
        """
        graph = compile_program(source, prelude=True)
        assert _unwrap(evaluate(graph, backend=backend), backend) == pytest.approx(120.0)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_multiple_defines(self, backend):
        source = """
        (define (double x) (* x 2))
        (define (add1 x) (+ x 1))
        (add1 (double 5))
        """
        graph = compile_program(source, prelude=True)
        assert _unwrap(evaluate(graph, backend=backend), backend) == pytest.approx(11.0)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_mutual_recursion(self, backend):
        source = """
        (define (my-even? n)
          (if (= n 0) 1 (my-odd? (- n 1))))
        (define (my-odd? n)
          (if (= n 0) 0 (my-even? (- n 1))))
        (my-even? 4)
        """
        graph = compile_program(source, prelude=True)
        assert _unwrap(evaluate(graph, backend=backend), backend) == pytest.approx(1.0)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_mutual_recursion_odd(self, backend):
        source = """
        (define (my-even? n)
          (if (= n 0) 1 (my-odd? (- n 1))))
        (define (my-odd? n)
          (if (= n 0) 0 (my-even? (- n 1))))
        (my-odd? 3)
        """
        graph = compile_program(source, prelude=True)
        assert _unwrap(evaluate(graph, backend=backend), backend) == pytest.approx(1.0)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_apply_twice(self, backend):
        source = """
        (let ((apply-twice (lambda (f x) (f (f x)))))
          (apply-twice (lambda (n) (+ n 1)) 5))
        """
        graph = compile_program(source, prelude=True)
        assert _unwrap(evaluate(graph, backend=backend), backend) == pytest.approx(7.0)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_compose(self, backend):
        source = """
        (let ((compose (lambda (f g) (lambda (x) (f (g x))))))
          ((compose (lambda (x) (* x 2)) (lambda (x) (+ x 1))) 5))
        """
        graph = compile_program(source, prelude=True)
        assert _unwrap(evaluate(graph, backend=backend), backend) == pytest.approx(12.0)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_higher_order(self, backend):
        source = """
        (let ((apply-fn (lambda (f x) (f x))))
          (apply-fn (lambda (n) (* n 2)) 7))
        """
        graph = compile_program(source, prelude=True)
        assert _unwrap(evaluate(graph, backend=backend), backend) == pytest.approx(14.0)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_define_with_let(self, backend):
        source = """
        (define (f x)
          (let ((y (* x 2)))
            (+ y 1)))
        (f 10)
        """
        graph = compile_program(source, prelude=True)
        assert _unwrap(evaluate(graph, backend=backend), backend) == pytest.approx(21.0)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_returned_closure(self, backend):
        source = """
        (let ((make-adder (lambda (x) (lambda (y) (+ x y)))))
          (let ((add5 (make-adder 5)))
            (add5 10)))
        """
        graph = compile_program(source, prelude=True)
        assert _unwrap(evaluate(graph, backend=backend), backend) == pytest.approx(15.0)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_closure_stored_in_cons(self, backend):
        source = """
        (let ((f (lambda (x) (+ x 1))))
          (let ((p (cons f 0)))
            ((car p) 5)))
        """
        graph = compile_program(source, prelude=True)
        assert _unwrap(evaluate(graph, backend=backend), backend) == pytest.approx(6.0)


# ================================================================== #
# Y-combinator / self-referential recursion
# ================================================================== #

class TestYCombinator:
    @pytest.mark.parametrize("backend", BACKENDS)
    def test_factorial_via_lambda(self, backend):
        source = EVALUATOR_SOURCE + """
        (scheme-eval '(let ((fact (lambda (self n)
                        (if (= n 0) 1 (* n (self self (- n 1)))))))
                       (fact fact 5))
                     '())
        """
        graph = compile_program(source, prelude=True)
        assert _unwrap(evaluate(graph, backend=backend), backend) == pytest.approx(120.0)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_sum_list(self, backend):
        source = EVALUATOR_SOURCE + """
        (scheme-eval '(let ((sum (lambda (self lst)
                        (if (null? lst)
                          0
                          (+ (car lst) (self self (cdr lst)))))))
                       (sum sum (list 1 2 3 4 5)))
                     '())
        """
        graph = compile_program(source, prelude=True)
        assert _unwrap(evaluate(graph, backend=backend), backend) == pytest.approx(15.0)


# ================================================================== #
# Loops with tagged values (build list, sum list)
# ================================================================== #

class TestTaggedLoops:
    @pytest.mark.parametrize("backend", BACKENDS)
    def test_sum_list_in_loop(self, backend):
        source = """
        (define (sum-list lst)
          (loop ((l lst) (acc 0))
            (if (null? l) acc
                (recur (cdr l) (+ acc (car l))))))
        (sum-list '(1 2 3 4 5))
        """
        graph = compile_program(source, prelude=True)
        assert _unwrap(evaluate(graph, backend=backend), backend) == pytest.approx(15.0)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_build_list_in_loop(self, backend):
        source = """
        (define (build n)
          (loop ((i n) (acc '()))
            (if (= i 0) acc
                (recur (- i 1) (cons i acc)))))
        (car (build 5))
        """
        graph = compile_program(source, prelude=True)
        assert _unwrap(evaluate(graph, backend=backend), backend) == pytest.approx(1.0)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_length_of_built_list(self, backend):
        source = """
        (define (build n)
          (loop ((i n) (acc '()))
            (if (= i 0) acc
                (recur (- i 1) (cons i acc)))))
        (length (build 5))
        """
        graph = compile_program(source, prelude=True)
        assert _unwrap(evaluate(graph, backend=backend), backend) == pytest.approx(5.0)


# ================================================================== #
# Bootstrap: evaluator evaluating programs
# ================================================================== #

class TestBootstrap:
    @pytest.mark.parametrize("backend", BACKENDS)
    def test_eval_number(self, backend):
        source = EVALUATOR_SOURCE + "\n(scheme-eval '42 '())\n"
        graph = compile_program(source, prelude=True)
        assert _unwrap(evaluate(graph, backend=backend), backend) == pytest.approx(42.0)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_eval_addition(self, backend):
        source = EVALUATOR_SOURCE + "\n(scheme-eval '(+ 3 4) '())\n"
        graph = compile_program(source, prelude=True)
        assert _unwrap(evaluate(graph, backend=backend), backend) == pytest.approx(7.0)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_eval_nested_arithmetic(self, backend):
        source = EVALUATOR_SOURCE + "\n(scheme-eval '(+ (* 3 4) 5) '())\n"
        graph = compile_program(source, prelude=True)
        assert _unwrap(evaluate(graph, backend=backend), backend) == pytest.approx(17.0)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_eval_subtraction(self, backend):
        source = EVALUATOR_SOURCE + "\n(scheme-eval '(- 10 3) '())\n"
        graph = compile_program(source, prelude=True)
        assert _unwrap(evaluate(graph, backend=backend), backend) == pytest.approx(7.0)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_eval_let(self, backend):
        source = EVALUATOR_SOURCE + "\n(scheme-eval '(let ((x 10)) (+ x 5)) '())\n"
        graph = compile_program(source, prelude=True)
        assert _unwrap(evaluate(graph, backend=backend), backend) == pytest.approx(15.0)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_eval_nested_let(self, backend):
        source = EVALUATOR_SOURCE + "\n(scheme-eval '(let ((x 3)) (let ((y 4)) (+ x y))) '())\n"
        graph = compile_program(source, prelude=True)
        assert _unwrap(evaluate(graph, backend=backend), backend) == pytest.approx(7.0)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_eval_lambda(self, backend):
        source = EVALUATOR_SOURCE + "\n(scheme-eval '((lambda (x) (+ x 1)) 5) '())\n"
        graph = compile_program(source, prelude=True)
        assert _unwrap(evaluate(graph, backend=backend), backend) == pytest.approx(6.0)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_eval_conditional_true(self, backend):
        source = EVALUATOR_SOURCE + "\n(scheme-eval '(if (> 5 3) 42 0) '())\n"
        graph = compile_program(source, prelude=True)
        assert _unwrap(evaluate(graph, backend=backend), backend) == pytest.approx(42.0)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_eval_conditional_false(self, backend):
        source = EVALUATOR_SOURCE + "\n(scheme-eval '(if (< 5 3) 42 0) '())\n"
        graph = compile_program(source, prelude=True)
        assert _unwrap(evaluate(graph, backend=backend), backend) == pytest.approx(0.0)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_eval_cons_car(self, backend):
        source = EVALUATOR_SOURCE + "\n(scheme-eval '(car (cons 1 2)) '())\n"
        graph = compile_program(source, prelude=True)
        assert _unwrap(evaluate(graph, backend=backend), backend) == pytest.approx(1.0)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_eval_cons_cdr(self, backend):
        source = EVALUATOR_SOURCE + "\n(scheme-eval '(cdr (cons 1 2)) '())\n"
        graph = compile_program(source, prelude=True)
        assert _unwrap(evaluate(graph, backend=backend), backend) == pytest.approx(2.0)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_eval_quote(self, backend):
        source = EVALUATOR_SOURCE + "\n(scheme-eval '(car '(10 20 30)) '())\n"
        graph = compile_program(source, prelude=True)
        assert _unwrap(evaluate(graph, backend=backend), backend) == pytest.approx(10.0)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_eval_closure(self, backend):
        source = EVALUATOR_SOURCE + """
        (scheme-eval '(let ((make-adder (lambda (x) (lambda (y) (+ x y)))))
                       ((make-adder 10) 5))
                     '())
        """
        graph = compile_program(source, prelude=True)
        assert _unwrap(evaluate(graph, backend=backend), backend) == pytest.approx(15.0)


# ================================================================== #
# Self-hosting: evaluator-program + define/cond via bootstrap
# ================================================================== #

class TestSelfHosting:
    def _run_program(self, program_sexpr: str, backend: str):
        source = EVALUATOR_SOURCE + f"\n(scheme-eval-program '{program_sexpr} '())\n"
        graph = compile_program(source, prelude=True)
        return evaluate(graph, backend=backend)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_simple_define(self, backend):
        result = self._run_program("""
            ((define (double x) (* x 2))
             (double 5))
        """, backend)
        assert _unwrap(result, backend) == pytest.approx(10.0)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_recursive_define(self, backend):
        result = self._run_program("""
            ((define (fact n)
               (if (= n 0) 1 (* n (fact (- n 1)))))
             (fact 5))
        """, backend)
        assert _unwrap(result, backend) == pytest.approx(120.0)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_mutual_recursion(self, backend):
        result = self._run_program("""
            ((define (my-even? n)
               (if (= n 0) 1 (my-odd? (- n 1))))
             (define (my-odd? n)
               (if (= n 0) 0 (my-even? (- n 1))))
             (my-even? 4))
        """, backend)
        assert _unwrap(result, backend) == pytest.approx(1.0)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_cond_in_eval(self, backend):
        source = EVALUATOR_SOURCE + "\n(scheme-eval '(cond ((> 5 3) 10) (#t 20)) '())\n"
        graph = compile_program(source, prelude=True)
        assert _unwrap(evaluate(graph, backend=backend), backend) == pytest.approx(10.0)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_boolean_in_eval(self, backend):
        source = EVALUATOR_SOURCE + "\n(scheme-eval '#t '())\n"
        graph = compile_program(source, prelude=True)
        assert _unwrap(evaluate(graph, backend=backend), backend) == pytest.approx(1.0)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_not_in_eval(self, backend):
        source = EVALUATOR_SOURCE + "\n(scheme-eval '(not #f) '())\n"
        graph = compile_program(source, prelude=True)
        assert _unwrap(evaluate(graph, backend=backend), backend) == pytest.approx(1.0)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_mini_eval(self, backend):
        result = self._run_program("""
            ((define (mini-eval expr)
               (if (number? expr)
                 expr
                 (if (pair? expr)
                   (let ((op (car expr))
                         (a (mini-eval (car (cdr expr))))
                         (b (mini-eval (car (cdr (cdr expr))))))
                     (if (eq? op '+) (+ a b)
                       (if (eq? op '*) (* a b) 0)))
                   0)))
             (mini-eval '(+ 3 (* 2 5))))
        """, backend)
        assert _unwrap(result, backend) == pytest.approx(13.0)


# ================================================================== #
# Backend parity: torch vs alternative on broad set
# ================================================================== #

class TestBackendParity:
    SCALAR_PROGRAMS = [
        ("(+ 1 2)", {}),
        ("(* x 3)", {"x": 4.0}),
        ("(if (> 5 3) 10 20)", {}),
        ("(if (< 2 1) 10 20)", {}),
        ("(not 0)", {}),
        ("(and 1 1)", {}),
        ("(or 0 1)", {}),
    ]

    @pytest.mark.parametrize("backend", BACKENDS)
    @pytest.mark.parametrize("source,inputs", SCALAR_PROGRAMS)
    def test_scalar_parity(self, backend, source, inputs):
        input_decl = {k: None for k in inputs}
        graph = compile_scheme(source, inputs=input_decl)
        torch_result = evaluate(graph, inputs, backend="torch")
        alt_result = evaluate(graph, inputs, backend=backend)
        assert float(alt_result) == pytest.approx(float(torch_result))

    TAGGED_PROGRAMS = [
        "(+ 3 4)",
        "(car (cons 10 20))",
        "(null? '())",
        "(pair? (cons 1 2))",
        "(number? 42)",
        "(eq? 'foo 'foo)",
        "(if #t 1 0)",
        "(not #f)",
        "(cond ((= 1 1) 99) (#t 0))",
    ]

    @pytest.mark.parametrize("backend", BACKENDS)
    @pytest.mark.parametrize("source", TAGGED_PROGRAMS)
    def test_tagged_parity(self, backend, source):
        graph = compile_program(source, prelude=True)
        torch_result = evaluate(graph, backend="torch")
        alt_result = evaluate(graph, backend=backend)
        assert _unwrap(alt_result, backend) == pytest.approx(
            _unwrap(torch_result, "torch")
        )


# ================================================================== #
# JAX autograd
# ================================================================== #

_HAS_JAX = "jax" in BACKENDS


@pytest.mark.skipif(not _HAS_JAX, reason="JAX not installed")
class TestJaxAutograd:
    """Verify jax.grad flows through compiled programs."""

    def test_identity_grad(self):
        from neural_compiler.evaluator import jax_grad
        graph = compile_scheme("x", inputs={"x": None})
        g = jax_grad(graph, {"x": 3.0}, "x")
        assert g == pytest.approx(1.0)

    def test_addition_grad(self):
        from neural_compiler.evaluator import jax_grad
        graph = compile_scheme("(+ x y)", inputs={"x": None, "y": None})
        gx = jax_grad(graph, {"x": 2.0, "y": 3.0}, "x")
        gy = jax_grad(graph, {"x": 2.0, "y": 3.0}, "y")
        assert gx == pytest.approx(1.0)
        assert gy == pytest.approx(1.0)

    def test_multiplication_grad(self):
        """d/dx(x * y) = y, d/dy(x * y) = x"""
        from neural_compiler.evaluator import jax_grad
        graph = compile_scheme("(* x y)", inputs={"x": None, "y": None})
        gx = jax_grad(graph, {"x": 3.0, "y": 5.0}, "x")
        gy = jax_grad(graph, {"x": 3.0, "y": 5.0}, "y")
        assert gx == pytest.approx(5.0)
        assert gy == pytest.approx(3.0)

    def test_polynomial_grad(self):
        """d/dx(x^2 + 3x + 1) = 2x + 3 = 9 at x=3"""
        from neural_compiler.evaluator import jax_grad
        src = "(+ (+ (* x x) (* 3 x)) 1)"
        graph = compile_scheme(src, inputs={"x": None})
        g = jax_grad(graph, {"x": 3.0}, "x")
        assert g == pytest.approx(9.0)

    def test_division_grad(self):
        """d/dx(1/x) = -1/x^2 = -0.25 at x=2"""
        from neural_compiler.evaluator import jax_grad
        graph = compile_scheme("(/ 1 x)", inputs={"x": None})
        g = jax_grad(graph, {"x": 2.0}, "x")
        assert g == pytest.approx(-0.25)

    def test_nested_arithmetic_grad(self):
        """d/dx((x+1)*(x-1)) = d/dx(x^2-1) = 2x = 8 at x=4"""
        from neural_compiler.evaluator import jax_grad
        src = "(* (+ x 1) (- x 1))"
        graph = compile_scheme(src, inputs={"x": None})
        g = jax_grad(graph, {"x": 4.0}, "x")
        assert g == pytest.approx(8.0)

    def test_multi_wrt(self):
        """Gradient w.r.t. multiple variables at once."""
        from neural_compiler.evaluator import jax_grad
        src = "(+ (* x x) (* y y))"
        graph = compile_scheme(src, inputs={"x": None, "y": None})
        gx, gy = jax_grad(graph, {"x": 3.0, "y": 4.0}, ["x", "y"])
        assert gx == pytest.approx(6.0)
        assert gy == pytest.approx(8.0)

    def test_value_and_grad(self):
        from neural_compiler.evaluator import jax_value_and_grad
        src = "(* x x)"
        graph = compile_scheme(src, inputs={"x": None})
        val, grad = jax_value_and_grad(graph, {"x": 5.0}, "x")
        assert val == pytest.approx(25.0)
        assert grad == pytest.approx(10.0)

    def test_value_and_grad_multi(self):
        from neural_compiler.evaluator import jax_value_and_grad
        src = "(* x y)"
        graph = compile_scheme(src, inputs={"x": None, "y": None})
        val, (gx, gy) = jax_value_and_grad(graph, {"x": 3.0, "y": 7.0}, ["x", "y"])
        assert val == pytest.approx(21.0)
        assert gx == pytest.approx(7.0)
        assert gy == pytest.approx(3.0)

    def test_let_binding_grad(self):
        """Gradient through let bindings: (let ((a (* x 2))) (* a x)) = 2x^2, grad=4x"""
        from neural_compiler.evaluator import jax_grad
        src = "(let ((a (* x 2))) (* a x))"
        graph = compile_scheme(src, inputs={"x": None})
        g = jax_grad(graph, {"x": 3.0}, "x")
        assert g == pytest.approx(12.0)

    def test_chain_rule(self):
        """d/dx(f(g(x))): (let ((y (* x 3))) (* y y)) = 9x^2, grad=18x"""
        from neural_compiler.evaluator import jax_grad
        src = "(let ((y (* x 3))) (* y y))"
        graph = compile_scheme(src, inputs={"x": None})
        g = jax_grad(graph, {"x": 2.0}, "x")
        assert g == pytest.approx(36.0)

    def test_second_derivative(self):
        """Second derivative: d^2/dx^2(x^3) = 6x"""
        import jax
        import jax.numpy as jnp
        from neural_compiler.evaluator.engine_generic import evaluate_generic
        from neural_compiler.backend.jax_backend import JaxBackend

        src = "(* (* x x) x)"
        graph = compile_scheme(src, inputs={"x": None})
        backend = JaxBackend()

        def f(x_val):
            return evaluate_generic(graph, {"x": x_val}, backend, raw_result=True)

        x = jnp.float32(2.0)
        d2 = jax.grad(jax.grad(f))(x)
        assert float(d2) == pytest.approx(12.0)
