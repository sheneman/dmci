############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# test_self_hosting.py: Self-hosting tests: the Scheme evaluator evaluates itself. The ultimate bootstrap proof: the Scheme evaluator...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Self-hosting tests: the Scheme evaluator evaluates itself.

The ultimate bootstrap proof: the Scheme evaluator (compiled to a
differentiable PyTorch program) evaluates its own source code as data,
producing a working inner evaluator that correctly evaluates Scheme
programs. Gradients flow through the entire chain.
"""

import pytest
import torch
from pathlib import Path

from neural_compiler.compiler import compile_program
from neural_compiler.evaluator import evaluate
from neural_compiler.runtime.tagged_value import unwrap_number, make_float


BOOTSTRAP_DIR = Path(__file__).parent.parent.parent / "bootstrap"
EVALUATOR_SOURCE = (BOOTSTRAP_DIR / "compiler.scm").read_text()


def _run_program(program_sexpr: str) -> object:
    """Compile the evaluator, then use scheme-eval-program to run a program."""
    source = EVALUATOR_SOURCE + f"\n(scheme-eval-program '{program_sexpr} '())\n"
    graph = compile_program(source, prelude=True)
    return evaluate(graph, {})


class TestDefineSupport:
    """scheme-eval-program handles define with mutual recursion."""

    def test_simple_define(self):
        result = _run_program("""
            ((define (double x) (* x 2))
             (double 5))
        """)
        assert unwrap_number(result).item() == pytest.approx(10.0)

    def test_multiple_defines(self):
        result = _run_program("""
            ((define (double x) (* x 2))
             (define (add1 x) (+ x 1))
             (add1 (double 5)))
        """)
        assert unwrap_number(result).item() == pytest.approx(11.0)

    def test_recursive_define(self):
        result = _run_program("""
            ((define (fact n)
               (if (= n 0) 1 (* n (fact (- n 1)))))
             (fact 5))
        """)
        assert unwrap_number(result).item() == pytest.approx(120.0)

    def test_recursive_define_le_base_case(self):
        """<= and >= are interpreter primitives, so a base case using them terminates.
        Regression: an unsupported comparison silently fell through to 0 in eval-apply, so the
        base case never fired and the recursion ran forever (until a heap overflow)."""
        result = _run_program("""
            ((define (sum-to n acc)
               (if (<= n 0) acc (sum-to (- n 1) (+ acc n))))
             (sum-to 5 0))
        """)
        assert unwrap_number(result).item() == pytest.approx(15.0)

    def test_le_and_ge_primitives(self):
        assert unwrap_number(_run_program("((if (<= 2 2) 10 20))")).item() == pytest.approx(10.0)
        assert unwrap_number(_run_program("((if (<= 3 2) 10 20))")).item() == pytest.approx(20.0)
        assert unwrap_number(_run_program("((if (>= 3 3) 10 20))")).item() == pytest.approx(10.0)
        assert unwrap_number(_run_program("((if (>= 2 3) 10 20))")).item() == pytest.approx(20.0)

    def test_mutual_recursion(self):
        result = _run_program("""
            ((define (my-even? n)
               (if (= n 0) 1 (my-odd? (- n 1))))
             (define (my-odd? n)
               (if (= n 0) 0 (my-even? (- n 1))))
             (my-even? 4))
        """)
        assert unwrap_number(result).item() == pytest.approx(1.0)

    def test_define_with_let(self):
        result = _run_program("""
            ((define (f x)
               (let ((y (* x 2)))
                 (+ y 1)))
             (f 10))
        """)
        assert unwrap_number(result).item() == pytest.approx(21.0)

    def test_define_with_closures(self):
        result = _run_program("""
            ((define (make-adder n) (lambda (x) (+ x n)))
             (let ((add5 (make-adder 5)))
               (add5 10)))
        """)
        assert unwrap_number(result).item() == pytest.approx(15.0)


class TestCondSupport:
    """scheme-eval handles cond expressions."""

    def test_cond_first_branch(self):
        source = EVALUATOR_SOURCE + "\n(scheme-eval '(cond ((> 5 3) 10) (#t 20)) '())\n"
        graph = compile_program(source, prelude=True)
        result = evaluate(graph, {})
        assert unwrap_number(result).item() == pytest.approx(10.0)

    def test_cond_fallthrough(self):
        source = EVALUATOR_SOURCE + "\n(scheme-eval '(cond ((< 5 3) 10) (#t 20)) '())\n"
        graph = compile_program(source, prelude=True)
        result = evaluate(graph, {})
        assert unwrap_number(result).item() == pytest.approx(20.0)

    def test_cond_with_else(self):
        source = EVALUATOR_SOURCE + "\n(scheme-eval '(cond ((< 1 0) 1) (else 99)) '())\n"
        graph = compile_program(source, prelude=True)
        result = evaluate(graph, {})
        assert unwrap_number(result).item() == pytest.approx(99.0)

    def test_cond_multiple_clauses(self):
        source = EVALUATOR_SOURCE + """
        (scheme-eval
          '(cond ((= 1 2) 10)
                 ((= 1 3) 20)
                 ((= 1 1) 30)
                 (#t 40))
          '())
        """
        graph = compile_program(source, prelude=True)
        result = evaluate(graph, {})
        assert unwrap_number(result).item() == pytest.approx(30.0)


class TestBooleanSupport:
    """scheme-eval handles booleans as self-evaluating values."""

    def test_true_literal(self):
        source = EVALUATOR_SOURCE + "\n(scheme-eval '#t '())\n"
        graph = compile_program(source, prelude=True)
        result = evaluate(graph, {})
        assert unwrap_number(result).item() == pytest.approx(1.0)

    def test_false_literal(self):
        source = EVALUATOR_SOURCE + "\n(scheme-eval '#f '())\n"
        graph = compile_program(source, prelude=True)
        result = evaluate(graph, {})
        assert unwrap_number(result).item() == pytest.approx(0.0)

    def test_not_true(self):
        source = EVALUATOR_SOURCE + "\n(scheme-eval '(not #t) '())\n"
        graph = compile_program(source, prelude=True)
        result = evaluate(graph, {})
        assert unwrap_number(result).item() == pytest.approx(0.0)

    def test_not_false(self):
        source = EVALUATOR_SOURCE + "\n(scheme-eval '(not #f) '())\n"
        graph = compile_program(source, prelude=True)
        result = evaluate(graph, {})
        assert unwrap_number(result).item() == pytest.approx(1.0)


class TestMiniEvaluator:
    """A mini evaluator running inside the full evaluator."""

    def test_mini_arithmetic_eval(self):
        result = _run_program("""
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
        """)
        assert unwrap_number(result).item() == pytest.approx(13.0)

    def test_mini_eval_with_cond(self):
        result = _run_program("""
            ((define (mini-eval expr)
               (cond
                 ((number? expr) expr)
                 ((pair? expr)
                  (let ((op (car expr))
                        (a (mini-eval (car (cdr expr))))
                        (b (mini-eval (car (cdr (cdr expr))))))
                    (cond
                      ((eq? op '+) (+ a b))
                      ((eq? op '-) (- a b))
                      ((eq? op '*) (* a b))
                      (#t 0))))
                 (#t 0)))
             (mini-eval '(- (* 3 4) (+ 1 2))))
        """)
        assert unwrap_number(result).item() == pytest.approx(9.0)


class TestMetacircular:
    """Full self-hosting: the evaluator evaluates its own source."""

    def _metacircular(self, inner_expr: str) -> object:
        """Compile evaluator, feed it its own source as data, evaluate inner_expr."""
        source = (
            EVALUATOR_SOURCE
            + "\n(scheme-eval-program\n  '(\n"
            + EVALUATOR_SOURCE
            + f"\n    {inner_expr}\n  )\n  '())\n"
        )
        graph = compile_program(source, prelude=True)
        return evaluate(graph, {})

    def test_eval_evaluates_arithmetic(self):
        result = self._metacircular("(scheme-eval '(+ 3 4) '())")
        assert unwrap_number(result).item() == pytest.approx(7.0)

    def test_eval_evaluates_nested(self):
        result = self._metacircular("(scheme-eval '(+ (* 3 4) (- 10 5)) '())")
        assert unwrap_number(result).item() == pytest.approx(17.0)

    def test_eval_evaluates_let(self):
        result = self._metacircular("(scheme-eval '(let ((x 10)) (+ x 5)) '())")
        assert unwrap_number(result).item() == pytest.approx(15.0)

    def test_eval_evaluates_lambda(self):
        result = self._metacircular("(scheme-eval '((lambda (x) (+ x 1)) 5) '())")
        assert unwrap_number(result).item() == pytest.approx(6.0)

    def test_eval_evaluates_conditional(self):
        result = self._metacircular("(scheme-eval '(if (> 5 3) 42 0) '())")
        assert unwrap_number(result).item() == pytest.approx(42.0)


class TestMetacircularGradient:
    """Gradients flow through programs evaluated via scheme-eval-program."""

    def test_gradient_through_define(self):
        source = EVALUATOR_SOURCE + """
        (scheme-eval-program
          (list
            '(define (add1 x) (+ x 1))
            (list 'add1 'x))
          (list (cons 'x x)))
        """
        graph = compile_program(source, inputs={"x": None}, prelude=True)
        x = torch.tensor(5.0, requires_grad=True)
        x_tagged = make_float(x)
        result = evaluate(graph, {"x": x_tagged})
        loss = unwrap_number(result)
        assert loss.item() == pytest.approx(6.0)
        loss.backward()
        assert x.grad is not None
        assert x.grad.item() == pytest.approx(1.0)

    def test_gradient_through_mutual_define(self):
        source = EVALUATOR_SOURCE + """
        (scheme-eval-program
          (list
            '(define (scale x) (* x 3))
            (list 'scale 'x))
          (list (cons 'x x)))
        """
        graph = compile_program(source, inputs={"x": None}, prelude=True)
        x = torch.tensor(4.0, requires_grad=True)
        x_tagged = make_float(x)
        result = evaluate(graph, {"x": x_tagged})
        loss = unwrap_number(result)
        assert loss.item() == pytest.approx(12.0)
        loss.backward()
        assert x.grad is not None
        assert x.grad.item() == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Tier 1: new interpreter ops (min/max/modulo/remainder), variadic arithmetic,
# and the loud-failure prescan that replaces silent eval-apply (#t 0) -> 0.
# ---------------------------------------------------------------------------
import re

from neural_compiler.dmci import (
    INTERPRETER_OPS,
    UnsupportedOperatorError,
    check_interpreter_supported,
    compile_dmci,
    unsupported_interpreter_ops,
)


class TestNewArithmeticOps:
    """min/max/modulo/remainder now dispatch in eval-apply instead of silently -> 0."""

    def test_min(self):
        assert unwrap_number(_run_program("((min 5 2))")).item() == pytest.approx(2.0)
        assert unwrap_number(_run_program("((min 2 5))")).item() == pytest.approx(2.0)

    def test_max(self):
        assert unwrap_number(_run_program("((max 5 2))")).item() == pytest.approx(5.0)
        assert unwrap_number(_run_program("((max 2 5))")).item() == pytest.approx(5.0)

    def test_modulo(self):
        assert unwrap_number(_run_program("((modulo 7 3))")).item() == pytest.approx(1.0)

    def test_remainder(self):
        assert unwrap_number(_run_program("((remainder 7 3))")).item() == pytest.approx(1.0)

    def test_min_as_recursion_guard(self):
        """Regression: min used in a guard must compute, not silently fall through to 0."""
        result = _run_program("""
            ((define (clip n) (if (> n 0) (min n 3) 0))
             (clip 10))
        """)
        assert unwrap_number(result).item() == pytest.approx(3.0)

    def test_min_gradient(self):
        """Gradient flows to the selected argument (hard subgradient)."""
        source = EVALUATOR_SOURCE + """
        (scheme-eval-program (list (list 'min 'x '5.0)) (list (cons 'x x)))
        """
        graph = compile_program(source, inputs={"x": None}, prelude=True)
        x = torch.tensor(2.0, requires_grad=True)
        result = evaluate(graph, {"x": make_float(x)})
        loss = unwrap_number(result)
        assert loss.item() == pytest.approx(2.0)
        loss.backward()
        assert x.grad is not None and x.grad.item() == pytest.approx(1.0)


class TestVariadicArithmetic:
    """+ - * / fold over all args instead of silently dropping the 3rd+."""

    def test_binary_unchanged(self):
        assert unwrap_number(_run_program("((+ 2 3))")).item() == pytest.approx(5.0)
        assert unwrap_number(_run_program("((- 7 2))")).item() == pytest.approx(5.0)
        assert unwrap_number(_run_program("((* 4 5))")).item() == pytest.approx(20.0)
        assert unwrap_number(_run_program("((/ 20 4))")).item() == pytest.approx(5.0)

    def test_variadic_add(self):
        assert unwrap_number(_run_program("((+ 1 2 3 4))")).item() == pytest.approx(10.0)

    def test_variadic_mul(self):
        assert unwrap_number(_run_program("((* 2 3 4))")).item() == pytest.approx(24.0)

    def test_left_assoc_sub(self):
        assert unwrap_number(_run_program("((- 10 1 2 3))")).item() == pytest.approx(4.0)

    def test_left_assoc_div(self):
        assert unwrap_number(_run_program("((/ 100 2 5))")).item() == pytest.approx(10.0)

    def test_unary_minus(self):
        assert unwrap_number(_run_program("((- 5))")).item() == pytest.approx(-5.0)

    def test_unary_reciprocal(self):
        assert unwrap_number(_run_program("((/ 4))")).item() == pytest.approx(0.25)

    def test_variadic_add_gradient(self):
        source = EVALUATOR_SOURCE + """
        (scheme-eval-program (list (list '+ 'x 'x 'x)) (list (cons 'x x)))
        """
        graph = compile_program(source, inputs={"x": None}, prelude=True)
        x = torch.tensor(2.0, requires_grad=True)
        result = evaluate(graph, {"x": make_float(x)})
        loss = unwrap_number(result)
        assert loss.item() == pytest.approx(6.0)
        loss.backward()
        assert x.grad.item() == pytest.approx(3.0)


class TestUnsupportedOpDetection:
    """The compile-time prescan turns silent eval-apply (#t 0) -> 0 into a loud error."""

    def test_unknown_op_raises(self):
        with pytest.raises(UnsupportedOperatorError):
            check_interpreter_supported("(tan x)")

    def test_supported_ops_pass(self):
        # no raise
        check_interpreter_supported("(+ (min a b) (max c (modulo d 2)))")

    def test_bound_function_not_flagged(self):
        check_interpreter_supported("(define (sq x) (* x x)) (sq 3)")

    def test_higher_order_param_not_flagged(self):
        check_interpreter_supported(
            "(define (twice g x) (g (g x))) (define (inc n) (+ n 1)) (twice inc 5)")

    def test_quoted_data_skipped(self):
        # 'tan here is quoted DATA, not an applied operator
        check_interpreter_supported("(car (quote (tan 1)))")

    def test_unsupported_set_reports_op(self):
        assert "tan" in unsupported_interpreter_ops("(tan x)")
        assert unsupported_interpreter_ops("(+ 1 2)") == set()

    def test_compile_dmci_rejects_unknown(self):
        with pytest.raises(UnsupportedOperatorError):
            compile_dmci("(expt x 2)")  # interpreter has pow, not expt

    def test_compile_dmci_accepts_new_ops(self):
        # min is now a real interpreter op; should compile (not raise)
        compile_dmci("(min x 5.0)")


class TestInterpreterOpsInSync:
    """INTERPRETER_OPS must match the ops actually dispatched by eval-apply in the .scm."""

    def test_ops_match_eval_apply(self):
        dispatched = set(re.findall(r"\(eq\?\s+func-expr\s+'([^\s)]+)\)", EVALUATOR_SOURCE))
        assert dispatched == INTERPRETER_OPS, (
            "INTERPRETER_OPS is out of sync with bootstrap/compiler.scm eval-apply.\n"
            f"  in .scm but not in INTERPRETER_OPS: {sorted(dispatched - INTERPRETER_OPS)}\n"
            f"  in INTERPRETER_OPS but not in .scm: {sorted(INTERPRETER_OPS - dispatched)}")


class TestDirectDmciEquivalence:
    """Multi-arity arithmetic must agree between direct compilation and DMCI (Theorem 1).
    Regression: variadic `/` once diverged (direct dropped args / crashed on unary) from
    the interpreter's left-associative fold."""

    @pytest.mark.parametrize("expr,expected", [
        ("(+ 1 2 3 4)", 10.0),
        ("(* 2 3 4)", 24.0),
        ("(- 10 1 2 3)", 4.0),
        ("(/ 100 2 5)", 10.0),
        ("(/ 4)", 0.25),
        ("(- 5)", -5.0),
        ("(min 5 2)", 2.0),
        ("(max 5 2)", 5.0),
        ("(modulo 7 3)", 1.0),
        ("(remainder 7 3)", 1.0),
    ])
    def test_direct_equals_dmci(self, expr, expected):
        from neural_compiler.compiler import compile_program
        direct = float(evaluate(compile_program(expr), {}))
        dmci = float(unwrap_number(evaluate(compile_dmci(expr), {})))
        assert direct == pytest.approx(expected), f"direct {expr} = {direct}"
        assert dmci == pytest.approx(expected), f"dmci {expr} = {dmci}"
        assert direct == pytest.approx(dmci), f"DIVERGENCE {expr}: direct={direct} dmci={dmci}"


# ---------------------------------------------------------------------------
# DMCI language parity: sugar (when/unless/let*/multi-body), loop/recur, and the
# list-backed vector/matrix library. Every feature is checked for direct-compile
# == DMCI equivalence (Theorem 1, up to floating-point reassociation).
# ---------------------------------------------------------------------------

def _direct_eval(src, feed=None):
    from neural_compiler.compiler import compile_program
    feed = feed or {}
    g = compile_program(src, inputs={k: None for k in feed} or None)
    return float(evaluate(g, feed))


def _dmci_eval(src, feed=None, **kw):
    from neural_compiler.runtime.tagged_value import make_float, unwrap_number
    feed = feed or {}
    g = compile_dmci(src)
    tagged = {n: make_float(torch.tensor(float(v))) for n, v in feed.items()}
    return float(unwrap_number(evaluate(g, tagged, **kw)))


class TestSugarEquivalence:
    """when/unless/let*/multi-expr-body lower to core forms; direct == DMCI."""

    @pytest.mark.parametrize("src,expected", [
        ("(let* ((a 2) (b (+ a 3))) (* a b))", 10.0),          # sequential binding deps
        ("(let* ((a 2) (b (* a a)) (c (+ b 1))) c)", 5.0),
        ("(when (< 1 2) (+ 3 4))", 7.0),                        # when true
        ("(unless (> 1 2) (* 4 2))", 8.0),                      # unless (test false) -> body
        ("(cond ((< 5 0) 1) ((< 2 1) 2) (else 99))", 99.0),    # cond else
        ("(cond ((= 2 2) (+ 1 1)) (else 0))", 2.0),
    ])
    def test_equiv(self, src, expected):
        d, m = _direct_eval(src), _dmci_eval(src)
        assert d == pytest.approx(expected), f"direct {src} = {d}"
        assert m == pytest.approx(expected), f"dmci {src} = {m}"
        assert d == pytest.approx(m)

    def test_when_false_agrees(self):
        # bare false `when` returns #f; just assert the two paths agree (no expected value)
        assert _dmci_eval("(when (> 1 2) 5)") == pytest.approx(_direct_eval("(when (> 1 2) 5)"))

    def test_multi_expr_body_not_dropped(self):
        # The interpreter takes a SINGLE body expr; without begin-wrapping it would return the
        # FIRST form (6), not the last (10). This guards the silent-drop trap.
        src = "(let ((x 5)) (+ x 1) (* x 2))"
        assert _direct_eval(src) == pytest.approx(10.0)
        assert _dmci_eval(src) == pytest.approx(10.0)


class TestLoopRecur:
    """loop/recur desugar to trampolined letrec; constant-stack at scale."""

    @pytest.mark.parametrize("src,expected", [
        ("(loop ((i 5) (acc 0)) (if (= i 0) acc (recur (- i 1) (+ acc i))))", 15.0),
        ("(loop ((i 3) (acc 1)) (cond ((= i 0) acc) (else (recur (- i 1) (* acc 2)))))", 8.0),
        ("(loop ((n 10) (a 0) (b 1)) (if (= n 0) a (recur (- n 1) b (+ a b))))", 55.0),  # fib(10)
    ])
    def test_equiv(self, src, expected):
        d, m = _direct_eval(src), _dmci_eval(src)
        assert d == pytest.approx(expected), f"direct {src} = {d}"
        assert m == pytest.approx(expected), f"dmci {src} = {m}"
        assert d == pytest.approx(m)

    def test_long_loop_constant_stack(self):
        # 1500 tail iterations must run via the trampoline (no Python recursion-limit crash);
        # heap is a no-GC bump allocator, so raise the caps for a long loop.
        src = "(loop ((i 1500) (acc 0)) (if (= i 0) acc (recur (- i 1) (+ acc 1))))"
        g = compile_dmci(src)
        from neural_compiler.runtime.tagged_value import unwrap_number
        val = float(unwrap_number(evaluate(g, {}, max_iter=400000, max_depth=400000,
                                            max_heap=4_000_000)))
        assert val == pytest.approx(1500.0)

    def test_non_tail_recur_rejected(self):
        # recur buried in an operand is not a tail call -> must raise, not silently degrade
        with pytest.raises(UnsupportedOperatorError):
            compile_dmci("(loop ((i 3)) (+ 1 (recur (- i 1))))")

    def test_recur_arity_checked(self):
        with pytest.raises(UnsupportedOperatorError):
            compile_dmci("(loop ((i 3) (acc 0)) (if (= i 0) acc (recur (- i 1))))")


class TestVectorMatrixEquivalence:
    """List-backed vec/ref/dot/matmul/... == direct torch primitives (up to FP reassoc).
    Each program reduces to a scalar so the two representations are directly comparable."""

    @pytest.mark.parametrize("src,expected", [
        ("(dot (vec 1 2 3) (vec 4 5 6))", 32.0),
        ("(ref (vec 10 20 30) 1)", 20.0),
        ("(vsum (vec 1 2 3 4))", 10.0),
        ("(vlen (vec 1 2 3))", 3.0),
        ("(vsum (scale 2 (vec 1 2 3)))", 12.0),
        ("(norm (vec 3 4))", 5.0),
        ("(vsum (matvec (mat (vec 1 2) (vec 3 4)) (vec 1 1)))", 10.0),
        ("(trace (matmul (mat (vec 1 2) (vec 3 4)) (mat (vec 5 6) (vec 7 8))))", 69.0),
        ("(trace (transpose (mat (vec 1 2) (vec 3 4))))", 5.0),
        ("(trace (mat (vec 1 2) (vec 3 4)))", 5.0),
        ("(trace (outer (vec 1 2) (vec 3 4)))", 11.0),
        ("(vsum (cross (vec 1 0 0) (vec 0 1 0)))", 1.0),
        ("(trace (eye 3))", 3.0),
        ("(vsum (zeros 5))", 0.0),
        ("(vsum (ones 4))", 4.0),
    ])
    def test_equiv(self, src, expected):
        d, m = _direct_eval(src), _dmci_eval(src)
        assert d == pytest.approx(expected, abs=1e-4), f"direct {src} = {d}"
        assert m == pytest.approx(expected, abs=1e-4), f"dmci {src} = {m}"
        assert d == pytest.approx(m, abs=1e-4)

    def test_gradient_through_dot(self):
        # dot([x, 2], [3, 4]) = 3x + 8  ->  d/dx = 3, flowing through the list-backed prelude
        from neural_compiler.runtime.tagged_value import make_float, unwrap_number
        g = compile_dmci("(dot (vec x 2.0) (vec 3.0 4.0))")
        x = torch.tensor(1.0, requires_grad=True)
        out = unwrap_number(evaluate(g, {"x": make_float(x)}))
        assert out.item() == pytest.approx(11.0)
        out.backward()
        assert x.grad.item() == pytest.approx(3.0)

    def test_det_inv_supported(self):
        # Strategy B v2: det/inv are now native (torch.linalg, batched + differentiable).
        from neural_compiler.runtime.tagged_value import unwrap_number
        g = compile_dmci("(det (mat (vec 1 2) (vec 3 4)))")
        assert float(unwrap_number(evaluate(g, {}))) == pytest.approx(-2.0, abs=1e-4)
        # inv([[1,2],[3,4]]) = [[-2,1],[1.5,-.5]]; trace = -2.5
        g2 = compile_dmci("(trace (inv (mat (vec 1 2) (vec 3 4))))")
        assert float(unwrap_number(evaluate(g2, {}))) == pytest.approx(-2.5, abs=1e-4)


# ---------------------------------------------------------------------------
# logdet (slogdet-backed): log|det M| that stays accurate when det underflows.
# (log (det S)) forms the determinant PRODUCT (~1e-20 for a D=20 sub-unit-eigenvalue
# covariance) and then logs it -- losing float32 precision AND hitting the 1e-8 clamp in
# the `log` primitive. logdet sums log|LU pivots| and is exact far below that. Needed for
# the Gaussian log-likelihood log det S term in the LIM/ENSO Kalman flagship.
# ---------------------------------------------------------------------------

class TestLogDet:

    def test_logdet_basic_and_direct_equiv(self):
        from neural_compiler.runtime.tagged_value import unwrap_number
        import math
        # logdet(diag(4,9)) = log(36)
        src = "(logdet (mat (vec 4.0 0.0) (vec 0.0 9.0)))"
        assert float(unwrap_number(evaluate(compile_dmci(src), {}))) == pytest.approx(math.log(36.0), abs=1e-4)
        # direct == DMCI (Theorem 1) -- direct path returns a plain number
        assert float(_direct_eval(src)) == pytest.approx(math.log(36.0), abs=1e-4)

    def test_logdet_beats_log_det_on_tiny_determinant(self):
        # rotated 0.1*I at D=20: true logdet = 20*log(0.1) = -46.05; (log (det S)) clamps to
        # ~-18.42 (= ln 1e-8), logdet is exact. This is the flagship-critical case.
        from neural_compiler.dmci import as_matrix
        from neural_compiler.runtime.tagged_value import unwrap_number
        D = 20
        torch.manual_seed(0)
        Vr, _ = torch.linalg.qr(torch.randn(D, D))
        S = (Vr @ torch.diag(torch.full((D,), 0.1)) @ Vr.T).float()
        true = float(torch.linalg.slogdet(S.double())[1])     # = -46.05
        ld = float(unwrap_number(evaluate(compile_dmci("(logdet S)"), {"S": as_matrix(S.clone())},
                                          max_iter=500000, max_depth=500000, max_heap=2_000_000)))
        assert ld == pytest.approx(true, abs=1e-2)
        # confirm the old path really was broken (clamped), so this test is load-bearing
        bad = float(unwrap_number(evaluate(compile_dmci("(log (det S))"), {"S": as_matrix(S.clone())},
                                           max_iter=500000, max_depth=500000, max_heap=2_000_000)))
        assert abs(bad - true) > 20.0   # (log (det S)) clamps to ~-18.4, off by ~27

    def test_logdet_gradient(self):
        # d/dS logdet(S) = inv(S)^T
        from neural_compiler.dmci import as_matrix
        from neural_compiler.runtime.tagged_value import unwrap_number
        D = 6
        torch.manual_seed(1)
        A = torch.randn(D, D)
        S = (A @ A.T + D * torch.eye(D)).float().requires_grad_(True)
        out = unwrap_number(evaluate(compile_dmci("(logdet S)"), {"S": as_matrix(S)},
                                     max_iter=500000, max_depth=500000, max_heap=2_000_000))
        out.backward()
        ref = torch.linalg.inv(S.detach()).T
        assert torch.allclose(S.grad, ref, atol=1e-4)

    def test_logdet_in_vec_ops(self):
        from neural_compiler.ops.tagged_ops import VEC_OPS
        from neural_compiler.ops.primitives import OP_TABLE
        from neural_compiler.dmci import INTERPRETER_OPS
        from neural_compiler.parser.ast_nodes import PRIMITIVES
        assert "logdet" in VEC_OPS and "logdet" in OP_TABLE
        assert "logdet" in INTERPRETER_OPS and "logdet" in PRIMITIVES


# ---------------------------------------------------------------------------
# Strategy B: tensor-payload tagged vectors. vec/ref/dot/matmul/... now store a
# real torch tensor in one heap slot and dispatch to the vectorized primitives
# (not list-folds). The TestVectorMatrixEquivalence suite above is the regression
# net (same programs/assertions, now via the tensor path). These add the
# tensor-specific guarantees: chained-op gradients, the large-D win, safety.
# ---------------------------------------------------------------------------

class TestStrategyBTensorVectors:

    def test_vec_ops_native_incl_det_inv(self):
        from neural_compiler.ops.tagged_ops import VEC_OPS
        from neural_compiler.ops.primitives import OP_TABLE
        assert VEC_OPS <= set(OP_TABLE), "every VEC_OP must reuse a primitives.OP_TABLE impl"
        assert {"det", "inv"} <= VEC_OPS, "det/inv are native as of v2"

    def test_gradient_through_matvec(self):
        # matvec([[x,0],[0,1]], [1,1]) = [x, 1]; vsum = x + 1; d/dx = 1.
        # Exercises mat (stack of vec refs) -> matvec -> vsum, grad through nested payloads.
        from neural_compiler.runtime.tagged_value import make_float, unwrap_number
        g = compile_dmci("(vsum (matvec (mat (vec x 0.0) (vec 0.0 1.0)) (vec 1.0 1.0)))")
        x = torch.tensor(3.0, requires_grad=True)
        out = unwrap_number(evaluate(g, {"x": make_float(x)}))
        assert out.item() == pytest.approx(4.0)
        out.backward()
        assert x.grad.item() == pytest.approx(1.0)

    def test_chained_matmul_gradient(self):
        # trace(outer([x,1],[1,1])) = trace([[x,x],[1,1]]) = x + 1; d/dx = 1
        from neural_compiler.runtime.tagged_value import make_float, unwrap_number
        g = compile_dmci("(trace (outer (vec x 1.0) (vec 1.0 1.0)))")
        x = torch.tensor(2.0, requires_grad=True)
        out = unwrap_number(evaluate(g, {"x": make_float(x)}))
        assert out.item() == pytest.approx(3.0)
        out.backward()
        assert x.grad.item() == pytest.approx(1.0)

    @pytest.mark.parametrize("n", [256, 4096])
    def test_large_d_runs(self, n):
        # The list-backed prelude needed ~n cons cells per vector and was O(n) per element
        # access; tensor payloads use ONE heap slot per tensor and a single vectorized kernel,
        # so large D runs correctly and fast (this would crawl / overflow heap before).
        from neural_compiler.runtime.tagged_value import unwrap_number
        g = compile_dmci(f"(dot (ones {n}) (ones {n}))")
        assert float(unwrap_number(evaluate(g, {}))) == pytest.approx(float(n))

    def test_rank_error_guard(self):
        # The batch-aware scalar-result guard: a non-reduced (matrix) result for a
        # scalar-result op must raise, not be silently mis-tagged as a batched scalar.
        # dot of two matrices reduces only the last axis -> [m] which, unbatched, is dim 1
        # (a legitimate batched-scalar shape) so dot([2x2],[2x2]) is allowed; trace of a
        # vector, however, is a genuine rank error. Use det of a vector:
        from neural_compiler.runtime.tagged_value import unwrap_number
        # sanity: this is just confirming the guard path exists; det needs a square matrix,
        # det of a vector is a torch error -> surfaces loudly (not a silent mis-tag).
        with pytest.raises(Exception):
            evaluate(compile_dmci("(det (vec 1 2 3))"), {})

    def test_user_function_shadows_vec_op(self):
        # A program may define a function named like a vector op; the user definition must
        # win over the native op. Regression: native vec clauses once shadowed user
        # defined-fns (broke test_gradient_through_mutual_define with a user `scale`).
        assert _dmci_eval("(define (scale x) (* x 3)) (scale 4)") == pytest.approx(12.0)
        assert _dmci_eval("(define (dot a) (+ a 100)) (dot 5)") == pytest.approx(105.0)


# ---------------------------------------------------------------------------
# Strategy B v2: batched tensor-vectors (2-axis encoding) + matrix row-ref + det/inv.
# Batch lives in the LEADING dims of the stored tensor, feature in the TRAILING dims;
# disambiguation is by tag (VECTOR vs FLOAT) + the ref's feature_ndim, so a coincidental
# batch-size == feature-dim is safe. The v1 unbatched suites above remain the regression net.
# ---------------------------------------------------------------------------

class TestStrategyBV2Batched:

    @staticmethod
    def _batched_scalar(src, **feed):
        """Compile, feed batched [N] params, assert the result is a FLOAT-tagged batched
        scalar (catches a wrong output-rank that returns a VECTOR ref by reading a heap
        address), and return the [N] payload."""
        from neural_compiler.runtime.tagged_value import make_float, unwrap_number, is_type, FLOAT
        g = compile_dmci(src)
        out = evaluate(g, {k: make_float(v) for k, v in feed.items()})
        assert bool(is_type(out, FLOAT).reshape(-1)[0].item() > 0.5), \
            "expected a FLOAT-tagged scalar result, got a non-scalar tag"
        return unwrap_number(out)

    def test_batched_dot_gradient(self):
        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)            # dot([x,2],[3,4]) = 3x+8
        out = self._batched_scalar("(dot (vec x 2.0) (vec 3.0 4.0))", x=x)
        assert torch.allclose(out, torch.tensor([11.0, 14.0, 17.0]), atol=1e-4)
        out.sum().backward()
        assert torch.allclose(x.grad, torch.full((3,), 3.0), atol=1e-4)

    def test_batched_matvec_gradient(self):
        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)            # matvec([[x,0],[0,1]],[1,1]) -> [x,1]
        out = self._batched_scalar(
            "(vsum (matvec (mat (vec x 0.0) (vec 0.0 1.0)) (vec 1.0 1.0)))", x=x)
        assert torch.allclose(out, torch.tensor([2.0, 3.0, 4.0]), atol=1e-4)   # x+1
        out.sum().backward()
        assert torch.allclose(x.grad, torch.ones(3), atol=1e-4)

    def test_batched_matmul(self):
        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        out = self._batched_scalar(
            "(trace (matmul (mat (vec x 0.0) (vec 0.0 1.0)) (mat (vec 1.0 0.0) (vec 0.0 1.0))))", x=x)
        assert torch.allclose(out, torch.tensor([2.0, 3.0, 4.0]), atol=1e-4)   # x+1
        out.sum().backward()
        assert x.grad is not None

    def test_batch_ne_feature_no_axis_confusion(self):
        # N (batch) != n (feature): if the batch and feature axes were confused this would
        # error or mis-broadcast. This is the key catch the 2-axis encoding must pass.
        x4 = torch.tensor([1.0, 2.0, 3.0, 4.0])                          # N=4, n=3: vsum(scale x [1,1,1]) = 3x
        out = self._batched_scalar("(vsum (scale x (ones 3)))", x=x4)
        assert torch.allclose(out, torch.tensor([3.0, 6.0, 9.0, 12.0]), atol=1e-4)
        x5 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])                     # N=5, n=2: dot([x,x],[1,1]) = 2x
        out2 = self._batched_scalar("(dot (vec x x) (vec 1.0 1.0))", x=x5)
        assert torch.allclose(out2, torch.tensor([2.0, 4.0, 6.0, 8.0, 10.0]), atol=1e-4)

    def test_matrix_row_ref(self):
        from neural_compiler.runtime.tagged_value import unwrap_number
        g = compile_dmci("(vsum (ref (mat (vec 1 2 3) (vec 4 5 6)) 1))")  # row 1 = [4,5,6] -> 15
        assert float(unwrap_number(evaluate(g, {}))) == pytest.approx(15.0)
        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)            # batched row 0 = [x,2,3] -> x+5
        out = self._batched_scalar("(vsum (ref (mat (vec x 2.0 3.0) (vec 4.0 5.0 6.0)) 0))", x=x)
        assert torch.allclose(out, torch.tensor([6.0, 7.0, 8.0]), atol=1e-4)
        out.sum().backward()
        assert torch.allclose(x.grad, torch.ones(3), atol=1e-4)

    def test_batched_vector_ref(self):
        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        out = self._batched_scalar("(ref (vec x 20.0 30.0) 0)", x=x)
        assert torch.allclose(out, torch.tensor([1.0, 2.0, 3.0]), atol=1e-4)
        out.sum().backward()
        assert torch.allclose(x.grad, torch.ones(3), atol=1e-4)

    def test_batched_det_inv_gradient(self):
        x = torch.tensor([2.0, 3.0, 4.0], requires_grad=True)            # det([[x,0],[0,1]]) = x
        out = self._batched_scalar("(det (mat (vec x 0.0) (vec 0.0 1.0)))", x=x)
        assert torch.allclose(out, torch.tensor([2.0, 3.0, 4.0]), atol=1e-4)
        out.sum().backward()
        assert torch.allclose(x.grad, torch.ones(3), atol=1e-4)          # d(det)/dx = 1
        x2 = torch.tensor([2.0, 4.0], requires_grad=True)                # trace(inv([[x,0],[0,1]])) = 1/x + 1
        out2 = self._batched_scalar("(trace (inv (mat (vec x 0.0) (vec 0.0 1.0))))", x=x2)
        assert torch.allclose(out2, torch.tensor([1.5, 1.25]), atol=1e-4)
        out2.sum().backward()
        assert torch.allclose(x2.grad, torch.tensor([-0.25, -0.0625]), atol=1e-4)  # -1/x^2

    def test_per_batch_ref_index_rejected(self):
        from neural_compiler.runtime.tagged_value import make_float
        g = compile_dmci("(ref (vec 10.0 20.0 30.0) i)")                 # per-batch (gather) index deferred
        with pytest.raises((ValueError, RuntimeError)):
            evaluate(g, {"i": make_float(torch.tensor([0.0, 1.0, 2.0]))})


# ---------------------------------------------------------------------------
# O(T) loop fix + reopened time-series pattern: a time loop gathers a per-step
# forcing value via (ref f k) and carries state. Lowering loop to a self-passing
# closure (fixed captured env) makes this O(T), not O(T^2) (defined-fn call-site
# env growth). Correctness here; scaling is benchmarked separately on n128.
# ---------------------------------------------------------------------------

class TestReopenedTimeLoop:

    def test_forcing_driven_loop(self):
        # leaky bucket s_{k+1} = 0.5 s_k + forcing[k], forcing=[1,1,1,1], s0=0
        # -> 1, 1.5, 1.75, 1.875
        from neural_compiler.runtime.tagged_value import unwrap_number
        src = ("(loop ((k 0) (s 0.0)) "
               "  (if (= k 4) s (recur (+ k 1) (+ (* 0.5 s) (ref (vec 1.0 1.0 1.0 1.0) k)))))")
        g = compile_dmci(src)
        assert float(unwrap_number(evaluate(g, {}, max_heap=200000))) == pytest.approx(1.875, abs=1e-5)

    def test_forcing_loop_gradient(self):
        # s_{k+1} = 0.5 s_k + a*forcing[k], forcing=[1,1,1], N=3 -> result = 1.75*a ; d/da = 1.75
        from neural_compiler.runtime.tagged_value import make_float, unwrap_number
        src = ("(loop ((k 0) (s 0.0)) "
               "  (if (= k 3) s (recur (+ k 1) (+ (* 0.5 s) (* a (ref (vec 1.0 1.0 1.0) k))))))")
        g = compile_dmci(src)
        a = torch.tensor(2.0, requires_grad=True)
        out = unwrap_number(evaluate(g, {"a": make_float(a)}, max_heap=200000))
        assert out.item() == pytest.approx(3.5, abs=1e-4)
        out.backward()
        assert a.grad.item() == pytest.approx(1.75, abs=1e-4)

    def test_long_forcing_loop_runs(self):
        # 1000-step forcing-driven loop must run (O(T) now); s accumulates 1 per step -> 1000
        from neural_compiler.runtime.tagged_value import unwrap_number
        src = ("(loop ((k 0) (s 0.0)) "
               "  (if (= k 1000) s (recur (+ k 1) (+ s 1.0))))")
        g = compile_dmci(src)
        val = float(unwrap_number(evaluate(g, {}, max_iter=500000, max_depth=500000, max_heap=4000000)))
        assert val == pytest.approx(1000.0)


# ---------------------------------------------------------------------------
# Elementwise tensor +/- (and the rest of arithmetic) on VECTOR/MATRIX payloads.
# Previously +/- on tensors unwrapped payload[0] (heap addresses) -> garbage; now
# _tagged_arith broadcasts via torch and returns a VECTOR ref. Required for Kalman
# (P F^T + Q, y - Hx, x + Ke) and compartmental (A C + b) models.
# ---------------------------------------------------------------------------

class TestElementwiseTensorArith:

    @pytest.mark.parametrize("src,expected", [
        ("(vsum (+ (vec 1 2 3) (vec 4 5 6)))", 21.0),                  # vector add -> [5,7,9]
        ("(vsum (- (vec 5 7 9) (vec 1 2 3)))", 15.0),                  # vector sub -> [4,5,6]
        ("(vsum (+ (vec 1 1) (vec 2 2) (vec 3 3)))", 12.0),            # variadic vector add -> [6,6]
        ("(trace (+ (mat (vec 1 2) (vec 3 4)) (mat (vec 10 0) (vec 0 10))))", 25.0),  # matrix add: 11+14
        ("(trace (+ (matmul (mat (vec 1 0) (vec 0 1)) (mat (vec 2 0) (vec 0 3))) (mat (vec 1 0) (vec 0 1))))", 7.0),  # F P F^T + Q pattern: diag(3,4)
        ("(vsum (- (vec 10 10 10) 1))", 27.0),                         # scalar broadcasts over vector -> [9,9,9]
    ])
    def test_equiv(self, src, expected):
        # direct compilation already broadcasts +/- over tensors; DMCI must now match it.
        d, m = _direct_eval(src), _dmci_eval(src)
        assert d == pytest.approx(expected, abs=1e-4), f"direct {src} = {d}"
        assert m == pytest.approx(expected, abs=1e-4), f"dmci {src} = {m}"
        assert d == pytest.approx(m, abs=1e-4)

    def test_gradient_through_vector_add(self):
        # vsum((vec x x x) - (vec 1 1 1)) = 3x - 3 ; d/dx = 3
        from neural_compiler.runtime.tagged_value import make_float, unwrap_number
        g = compile_dmci("(vsum (- (vec x x x) (vec 1.0 1.0 1.0)))")
        x = torch.tensor(2.0, requires_grad=True)
        out = unwrap_number(evaluate(g, {"x": make_float(x)}))
        assert out.item() == pytest.approx(3.0)   # 3*2 - 3
        out.backward()
        assert x.grad.item() == pytest.approx(3.0)

    def test_batched_vector_add(self):
        # batched: vsum((vec x 2) + (vec 3 4)) = (x+3) + 6 per batch element
        from neural_compiler.runtime.tagged_value import make_float, unwrap_number, is_type, FLOAT
        g = compile_dmci("(vsum (+ (vec x 2.0) (vec 3.0 4.0)))")
        xb = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        out = evaluate(g, {"x": make_float(xb)})
        assert bool(is_type(out, FLOAT).reshape(-1)[0].item() > 0.5)
        vals = unwrap_number(out)
        assert torch.allclose(vals, torch.tensor([10.0, 11.0, 12.0]), atol=1e-4)  # x+9
        vals.sum().backward()
        assert torch.allclose(xb.grad, torch.ones(3), atol=1e-4)

    def test_kalman_covariance_update_shape(self):
        # the actual P_pred = F P F^T + Q step on real 2x2 matrices, reduced via trace
        from neural_compiler.runtime.tagged_value import unwrap_number
        src = ("(trace (+ (matmul (matmul (mat (vec 1.0 1.0) (vec 0.0 1.0)) "
               "                          (mat (vec 1.0 0.0) (vec 0.0 1.0))) "
               "                  (transpose (mat (vec 1.0 1.0) (vec 0.0 1.0)))) "
               "          (mat (vec 0.1 0.0) (vec 0.0 0.1))))")
        g = compile_dmci(src)
        # F=[[1,1],[0,1]], P=I, Q=0.1 I -> F P F^T = [[2,1],[1,1]]; +Q -> trace = 2.1+1.1 = 3.2
        assert float(unwrap_number(evaluate(g, {}))) == pytest.approx(3.2, abs=1e-4)


# ---------------------------------------------------------------------------
# Data-ingestion helper: bind a real [T] / [T,D] tensor as a VECTOR/MATRIX payload
# input via as_vector / as_matrix, so (ref obs k)/matvec/... use it and gradients flow.
# (Without it, a raw tensor input is make_float'd into a batch of scalars -> wrong.)
# ---------------------------------------------------------------------------

class TestTensorInput:

    def test_vector_series_input(self):
        from neural_compiler.dmci import as_vector
        from neural_compiler.runtime.tagged_value import unwrap_number
        # sum a forcing series fed as a tensor input: sum([10,20,30,40]) = 100
        src = "(loop ((k 0) (s 0.0)) (if (= k 4) s (recur (+ k 1) (+ s (ref series k)))))"
        g = compile_dmci(src)
        series = torch.tensor([10.0, 20.0, 30.0, 40.0])
        out = unwrap_number(evaluate(g, {"series": as_vector(series)}, max_heap=200000))
        assert out.item() == pytest.approx(100.0)

    def test_matrix_input_via_ref(self):
        from neural_compiler.dmci import as_matrix
        from neural_compiler.runtime.tagged_value import unwrap_number
        # sum all elements of a [3,2] obs matrix bound as input, reading row k each step
        src = "(loop ((k 0) (s 0.0)) (if (= k 3) s (recur (+ k 1) (+ s (vsum (ref obs k))))))"
        g = compile_dmci(src)
        obs = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        out = unwrap_number(evaluate(g, {"obs": as_matrix(obs)}, max_heap=200000))
        assert out.item() == pytest.approx(21.0)

    def test_matrix_input_gradient(self):
        from neural_compiler.dmci import as_matrix
        from neural_compiler.runtime.tagged_value import unwrap_number
        src = "(loop ((k 0) (s 0.0)) (if (= k 2) s (recur (+ k 1) (+ s (vsum (ref obs k))))))"
        g = compile_dmci(src)
        obs = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        out = unwrap_number(evaluate(g, {"obs": as_matrix(obs)}, max_heap=200000))
        assert out.item() == pytest.approx(10.0)              # 1+2+3+4
        out.backward()
        assert torch.allclose(obs.grad, torch.ones(2, 2), atol=1e-5)  # d(sum)/d(each)=1

    def test_kalman_innovation_with_obs_input(self):
        # the flagship data-feed pattern: e_k = y_k - H x, with y_k gathered from a bound
        # obs matrix, H a matrix (matvec), and elementwise vector subtract. H=I, x=[1,1].
        from neural_compiler.dmci import as_matrix
        from neural_compiler.runtime.tagged_value import unwrap_number
        src = ("(vsum (- (ref obs 0) "
               "         (matvec (mat (vec 1.0 0.0) (vec 0.0 1.0)) (vec 1.0 1.0))))")
        g = compile_dmci(src)
        obs = torch.tensor([[5.0, 7.0], [9.0, 11.0]])
        out = unwrap_number(evaluate(g, {"obs": as_matrix(obs)}, max_heap=200000))
        assert out.item() == pytest.approx(10.0)              # [5,7]-[1,1] = [4,6] -> 10
