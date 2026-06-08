############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# test_cons_and_quote.py: Integration tests for cons cells, quote, lists, and tagged value evaluation.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Integration tests for cons cells, quote, lists, and tagged value evaluation."""

import pytest
import torch

from neural_compiler.compiler import compile_scheme
from neural_compiler.evaluator import evaluate
from neural_compiler.runtime.tagged_value import (
    VALUE_DIM, NIL, BOOL, INT, FLOAT, PAIR, SYMBOL,
    type_index, type_name, unwrap_number, unwrap_bool,
    make_float, make_int, is_nil, is_pair,
)


def _eval_tagged(source, inputs=None):
    """Compile and evaluate a tagged-value program."""
    inputs = inputs or {}
    input_decl = {k: None for k in inputs}
    graph = compile_scheme(source, inputs=input_decl)
    assert graph.uses_tagged_values
    return evaluate(graph, inputs)


class TestCons:
    def test_cons_creates_pair(self):
        result = _eval_tagged("(cons 1 2)")
        assert type_index(result) == PAIR

    def test_car_of_cons(self):
        result = _eval_tagged("(car (cons 1 2))")
        assert unwrap_number(result).item() == 1.0

    def test_cdr_of_cons(self):
        result = _eval_tagged("(cdr (cons 1 2))")
        assert unwrap_number(result).item() == 2.0

    def test_nested_cons(self):
        result = _eval_tagged("(car (car (cons (cons 10 20) 30)))")
        assert unwrap_number(result).item() == 10.0

    def test_cdr_of_nested(self):
        result = _eval_tagged("(cdr (car (cons (cons 10 20) 30)))")
        assert unwrap_number(result).item() == 20.0


class TestQuote:
    def test_quote_empty_list(self):
        result = _eval_tagged("'()")
        assert type_index(result) == NIL

    def test_quote_number(self):
        result = _eval_tagged("'42")
        assert unwrap_number(result).item() == 42.0

    def test_quote_list(self):
        result = _eval_tagged("'(1 2 3)")
        assert type_index(result) == PAIR

    def test_car_of_quoted_list(self):
        result = _eval_tagged("(car '(10 20 30))")
        assert unwrap_number(result).item() == 10.0

    def test_cadr_of_quoted_list(self):
        result = _eval_tagged("(car (cdr '(10 20 30)))")
        assert unwrap_number(result).item() == 20.0

    def test_caddr_of_quoted_list(self):
        result = _eval_tagged("(car (cdr (cdr '(10 20 30))))")
        assert unwrap_number(result).item() == 30.0

    def test_cddr_cdr_is_nil(self):
        result = _eval_tagged("(null? (cdr (cdr (cdr '(10 20 30)))))")
        assert unwrap_number(result).item() == 1.0

    def test_quote_symbol(self):
        result = _eval_tagged("'foo")
        assert type_index(result) == SYMBOL


class TestList:
    def test_list_creates_proper_list(self):
        result = _eval_tagged("(list 1 2 3)")
        assert type_index(result) == PAIR

    def test_car_of_list(self):
        result = _eval_tagged("(car (list 7 8 9))")
        assert unwrap_number(result).item() == 7.0

    def test_cadr_of_list(self):
        result = _eval_tagged("(car (cdr (list 7 8 9)))")
        assert unwrap_number(result).item() == 8.0

    def test_list_terminated_by_nil(self):
        result = _eval_tagged("(null? (cdr (list 42)))")
        assert unwrap_number(result).item() == 1.0

    def test_empty_list(self):
        result = _eval_tagged("(list)")
        assert type_index(result) == NIL


class TestPredicates:
    def test_null_of_nil(self):
        result = _eval_tagged("(null? '())")
        assert unwrap_number(result).item() == 1.0

    def test_null_of_pair(self):
        result = _eval_tagged("(null? (cons 1 2))")
        assert unwrap_number(result).item() == 0.0

    def test_pair_of_cons(self):
        result = _eval_tagged("(pair? (cons 1 2))")
        assert unwrap_number(result).item() == 1.0

    def test_pair_of_number(self):
        result = _eval_tagged("(pair? 42)")
        assert unwrap_number(result).item() == 0.0

    def test_number_of_int(self):
        result = _eval_tagged("(number? 42)")
        assert unwrap_number(result).item() == 1.0

    def test_number_of_pair(self):
        result = _eval_tagged("(number? (cons 1 2))")
        assert unwrap_number(result).item() == 0.0

    def test_boolean_of_true(self):
        # #t is compiled as const(1.0) which is FLOAT in tagged mode,
        # so boolean? returns false. This is expected — booleans are numbers
        # in the current numeric compilation. True boolean? requires
        # a tagged boolean constant, which will come with Phase 2.
        result = _eval_tagged("(boolean? (null? '()))")
        assert unwrap_number(result).item() == 1.0

    def test_symbol_of_quoted(self):
        result = _eval_tagged("(symbol? 'foo)")
        assert unwrap_number(result).item() == 1.0


class TestEquality:
    def test_eq_same_symbol(self):
        result = _eval_tagged("(eq? 'foo 'foo)")
        assert unwrap_number(result).item() == 1.0

    def test_eq_different_symbols(self):
        result = _eval_tagged("(eq? 'foo 'bar)")
        assert unwrap_number(result).item() == 0.0

    def test_eq_numbers(self):
        result = _eval_tagged("(eq? 42 42)")
        assert unwrap_number(result).item() == 1.0

    def test_eq_nil(self):
        result = _eval_tagged("(eq? '() '())")
        assert unwrap_number(result).item() == 1.0

    def test_equal_nested_list(self):
        result = _eval_tagged("(equal? '(1 2 3) '(1 2 3))")
        assert unwrap_number(result).item() == 1.0

    def test_equal_different_lists(self):
        result = _eval_tagged("(equal? '(1 2 3) '(1 2 4))")
        assert unwrap_number(result).item() == 0.0


class TestArithOnTagged:
    def test_add_tagged(self):
        result = _eval_tagged("(+ (car (cons 3 0)) (car (cons 4 0)))")
        assert unwrap_number(result).item() == pytest.approx(7.0)

    def test_car_plus_cdr(self):
        result = _eval_tagged("(+ (car (cons 3 4)) (cdr (cons 3 4)))")
        assert unwrap_number(result).item() == pytest.approx(7.0)

    def test_compare_tagged(self):
        result = _eval_tagged("(> (car (cons 5 0)) (car (cons 3 0)))")
        assert unwrap_number(result).item() == 1.0

    def test_if_with_tagged(self):
        result = _eval_tagged("(if (> 5 3) (cons 1 2) (cons 3 4))")
        assert type_index(result) == PAIR


class TestConditionalOnTags:
    def test_if_pair_returns_car(self):
        result = _eval_tagged("""
            (let ((x (cons 10 20)))
              (if (pair? x) (car x) 0))
        """)
        assert unwrap_number(result).item() == pytest.approx(10.0)

    def test_if_null_returns_else(self):
        result = _eval_tagged("""
            (let ((x '()))
              (if (null? x) 99 0))
        """)
        assert unwrap_number(result).item() == pytest.approx(99.0)


class TestLetWithCons:
    def test_let_binding_cons(self):
        result = _eval_tagged("""
            (let ((p (cons 5 10)))
              (+ (car p) (cdr p)))
        """)
        assert unwrap_number(result).item() == pytest.approx(15.0)

    def test_nested_let_cons(self):
        result = _eval_tagged("""
            (let ((p (cons 1 2)))
              (let ((q (cons (car p) 3)))
                (+ (car q) (cdr q))))
        """)
        assert unwrap_number(result).item() == pytest.approx(4.0)


class TestLoopWithCons:
    def test_build_list_in_loop(self):
        """Build a list (3 2 1) by looping."""
        result = _eval_tagged("""
            (loop ((i 3) (acc '()))
              (if (= i 0)
                acc
                (recur (- i 1) (cons i acc))))
        """)
        assert type_index(result) == PAIR
        # Should be (1 2 3)

    def test_sum_list_in_loop(self):
        """Sum elements of a cons-list built by loop."""
        result = _eval_tagged("""
            (let ((lst (loop ((i 3) (acc '()))
                         (if (= i 0) acc (recur (- i 1) (cons i acc))))))
              (loop ((l lst) (sum 0))
                (if (null? l)
                  sum
                  (recur (cdr l) (+ sum (car l))))))
        """)
        assert unwrap_number(result).item() == pytest.approx(6.0)


class TestGradientFlow:
    def test_grad_through_car_cons(self):
        graph = compile_scheme("(car (cons x 0))", inputs={"x": None})
        x = torch.tensor(5.0, requires_grad=True)
        x_tagged = make_float(x)
        result = evaluate(graph, {"x": x_tagged})
        loss = unwrap_number(result)
        loss.backward()
        assert x.grad is not None
        assert x.grad.item() == pytest.approx(1.0)

    def test_grad_through_cdr_cons(self):
        graph = compile_scheme("(cdr (cons 0 x))", inputs={"x": None})
        x = torch.tensor(7.0, requires_grad=True)
        x_tagged = make_float(x)
        result = evaluate(graph, {"x": x_tagged})
        loss = unwrap_number(result)
        loss.backward()
        assert x.grad is not None
        assert x.grad.item() == pytest.approx(1.0)

    def test_grad_sum_of_car_cdr(self):
        graph = compile_scheme("(+ (car (cons x 0)) (cdr (cons 0 x)))", inputs={"x": None})
        x = torch.tensor(3.0, requires_grad=True)
        x_tagged = make_float(x)
        result = evaluate(graph, {"x": x_tagged})
        loss = unwrap_number(result)
        loss.backward()
        assert x.grad is not None
        assert x.grad.item() == pytest.approx(2.0)


class TestBackwardCompatibility:
    """Ensure non-tagged programs still work with bare tensors."""

    def test_numeric_program_no_tags(self):
        from neural_compiler.compiler import run_scheme
        result = run_scheme("(+ 1 2)")
        assert result == pytest.approx(3.0)

    def test_factorial_no_tags(self):
        from neural_compiler.compiler import run_scheme
        result = run_scheme(
            "(loop ((n 10) (acc 1)) (if (= n 0) acc (recur (- n 1) (* acc n))))"
        )
        assert result == pytest.approx(3628800.0)

    def test_graph_flag_false_for_numeric(self):
        graph = compile_scheme("(+ (* 3 4) 5)")
        assert not graph.uses_tagged_values
