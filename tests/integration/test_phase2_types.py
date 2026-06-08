############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# test_phase2_types.py: Phase 2 integration tests: booleans, characters, symbols, type predicates, cond.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Phase 2 integration tests: booleans, characters, symbols, type predicates, cond."""

import pytest

from neural_compiler.compiler import compile_scheme
from neural_compiler.evaluator import evaluate
from neural_compiler.runtime.tagged_value import (
    VALUE_DIM, NIL, BOOL, INT, FLOAT, CHAR, PAIR, SYMBOL,
    type_index, unwrap_number, unwrap_bool,
    make_float,
)


def _eval_tagged(source, inputs=None):
    inputs = inputs or {}
    input_decl = {k: None for k in inputs}
    graph = compile_scheme(source, inputs=input_decl)
    assert graph.uses_tagged_values
    return evaluate(graph, inputs)


class TestBooleanLiterals:
    def test_true_is_bool_type(self):
        result = _eval_tagged("(pair? #t)")
        assert type_index(result) == BOOL

    def test_false_is_bool_type(self):
        result = _eval_tagged("(pair? #f)")
        assert type_index(result) == BOOL

    def test_boolean_pred_on_true(self):
        result = _eval_tagged("(boolean? (null? '()))")
        assert unwrap_number(result).item() == 1.0

    def test_boolean_pred_on_false(self):
        result = _eval_tagged("(boolean? (pair? 42))")
        assert unwrap_number(result).item() == 1.0

    def test_boolean_pred_on_number(self):
        result = _eval_tagged("(boolean? (car (cons 1 2)))")
        assert unwrap_number(result).item() == 0.0


class TestCharLiterals:
    def test_char_a(self):
        result = _eval_tagged("(cons #\\a 0)")
        assert type_index(result) == PAIR

    def test_char_pred_true(self):
        result = _eval_tagged("(char? (car (cons #\\a 0)))")
        assert unwrap_number(result).item() == 1.0

    def test_char_pred_false_on_number(self):
        result = _eval_tagged("(char? (car (cons 42 0)))")
        assert unwrap_number(result).item() == 0.0

    def test_char_space(self):
        result = _eval_tagged("(char? (car (cons #\\space 0)))")
        assert unwrap_number(result).item() == 1.0

    def test_char_newline(self):
        result = _eval_tagged("(char? (car (cons #\\newline 0)))")
        assert unwrap_number(result).item() == 1.0

    def test_char_eq(self):
        result = _eval_tagged("(eq? (car (cons #\\a 0)) (car (cons #\\a 0)))")
        assert unwrap_number(result).item() == 1.0

    def test_char_neq(self):
        result = _eval_tagged("(eq? (car (cons #\\a 0)) (car (cons #\\b 0)))")
        assert unwrap_number(result).item() == 0.0


class TestSymbolOps:
    def test_symbol_interning_same(self):
        result = _eval_tagged("(eq? 'hello 'hello)")
        assert unwrap_number(result).item() == 1.0

    def test_symbol_interning_different(self):
        result = _eval_tagged("(eq? 'hello 'world)")
        assert unwrap_number(result).item() == 0.0

    def test_symbol_pred_true(self):
        result = _eval_tagged("(symbol? 'foo)")
        assert unwrap_number(result).item() == 1.0

    def test_symbol_pred_false_on_number(self):
        result = _eval_tagged("(symbol? (car (cons 42 0)))")
        assert unwrap_number(result).item() == 0.0

    def test_symbol_in_list(self):
        result = _eval_tagged("(symbol? (car '(foo bar)))")
        assert unwrap_number(result).item() == 1.0

    def test_symbol_eq_in_list(self):
        result = _eval_tagged("(eq? (car '(foo bar)) 'foo)")
        assert unwrap_number(result).item() == 1.0


class TestAllPredicates:
    """Test every type predicate against every value type."""

    def test_number_pred_on_int(self):
        result = _eval_tagged("(number? (car (cons 42 0)))")
        assert unwrap_number(result).item() == 1.0

    def test_number_pred_on_symbol(self):
        result = _eval_tagged("(number? 'foo)")
        assert unwrap_number(result).item() == 0.0

    def test_number_pred_on_nil(self):
        result = _eval_tagged("(number? '())")
        assert unwrap_number(result).item() == 0.0

    def test_null_pred_on_nil(self):
        result = _eval_tagged("(null? '())")
        assert unwrap_number(result).item() == 1.0

    def test_null_pred_on_pair(self):
        result = _eval_tagged("(null? (cons 1 2))")
        assert unwrap_number(result).item() == 0.0

    def test_null_pred_on_number(self):
        result = _eval_tagged("(null? (car (cons 42 0)))")
        assert unwrap_number(result).item() == 0.0

    def test_pair_pred_on_pair(self):
        result = _eval_tagged("(pair? (cons 1 2))")
        assert unwrap_number(result).item() == 1.0

    def test_pair_pred_on_nil(self):
        result = _eval_tagged("(pair? '())")
        assert unwrap_number(result).item() == 0.0

    def test_procedure_pred_on_number(self):
        result = _eval_tagged("(procedure? (car (cons 42 0)))")
        assert unwrap_number(result).item() == 0.0

    def test_string_pred_on_number(self):
        result = _eval_tagged("(string? (car (cons 42 0)))")
        assert unwrap_number(result).item() == 0.0

    def test_vector_pred_on_number(self):
        result = _eval_tagged("(vector? (car (cons 42 0)))")
        assert unwrap_number(result).item() == 0.0


class TestCond:
    def test_cond_first_clause(self):
        result = _eval_tagged("""
            (cond
              ((null? '()) (cons 1 0))
              (#t (cons 2 0)))
        """)
        from neural_compiler.runtime.tagged_value import unwrap_number
        from neural_compiler.evaluator import evaluate as ev
        from neural_compiler.compiler import compile_scheme
        graph = compile_scheme("""
            (car (cond
              ((null? '()) (cons 1 0))
              (#t (cons 2 0))))
        """)
        r = ev(graph, {})
        assert unwrap_number(r).item() == 1.0

    def test_cond_second_clause(self):
        graph = compile_scheme("""
            (car (cond
              ((null? (cons 1 2)) (cons 10 0))
              (#t (cons 20 0))))
        """)
        result = evaluate(graph, {})
        assert unwrap_number(result).item() == 20.0

    def test_cond_else_clause(self):
        graph = compile_scheme("""
            (car (cond
              ((null? (cons 1 2)) (cons 10 0))
              (else (cons 30 0))))
        """)
        result = evaluate(graph, {})
        assert unwrap_number(result).item() == 30.0


class TestEqualityStructural:
    def test_equal_nested_cons(self):
        result = _eval_tagged("(equal? (cons 1 (cons 2 '())) (cons 1 (cons 2 '())))")
        assert unwrap_number(result).item() == 1.0

    def test_equal_different_structure(self):
        result = _eval_tagged("(equal? (cons 1 2) (cons 1 3))")
        assert unwrap_number(result).item() == 0.0

    def test_equal_nil_nil(self):
        result = _eval_tagged("(equal? '() '())")
        assert unwrap_number(result).item() == 1.0

    def test_equal_symbol_and_number(self):
        result = _eval_tagged("(equal? 'foo (car (cons 42 0)))")
        assert unwrap_number(result).item() == 0.0

    def test_eqv_same_as_eq_for_atoms(self):
        result = _eval_tagged("(eqv? 'foo 'foo)")
        assert unwrap_number(result).item() == 1.0


class TestCondWithSymbol:
    def test_dispatch_on_symbol(self):
        graph = compile_scheme("""
            (let ((tag (car '(add 1 2))))
              (car (cond
                ((eq? tag 'add) (cons 100 0))
                ((eq? tag 'sub) (cons 200 0))
                (else (cons 0 0)))))
        """)
        result = evaluate(graph, {})
        assert unwrap_number(result).item() == 100.0

    def test_dispatch_on_symbol_second(self):
        graph = compile_scheme("""
            (let ((tag (car '(sub 1 2))))
              (car (cond
                ((eq? tag 'add) (cons 100 0))
                ((eq? tag 'sub) (cons 200 0))
                (else (cons 0 0)))))
        """)
        result = evaluate(graph, {})
        assert unwrap_number(result).item() == 200.0
