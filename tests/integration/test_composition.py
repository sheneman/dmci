############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# test_composition.py: Integration tests for nested loops / recursion composition (v0.4.0). Tests all composition patterns across all...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Integration tests for nested loops / recursion composition (v0.4.0).

Tests all composition patterns across all three evaluators:
  - Loop inside function body (letrec with loop in lambda)
  - Letrec inside loop body (loop body defines recursive functions)
  - Nested loops (loop inside loop)
  - Letrec calling loop-containing functions
  - Loop calling letrec-defined functions
"""

import pytest
import torch
from neural_compiler.compiler import compile_scheme, run_scheme
from neural_compiler.evaluator import evaluate, SchemeGNN, DirectModule


def _all_eval(source, inputs=None):
    """Evaluate with all three evaluators and return (seq, gnn, pyg)."""
    inputs = inputs or {}
    input_decl = {k: None for k in inputs}
    graph = compile_scheme(source, inputs=input_decl)

    seq = evaluate(graph, inputs)
    gnn = SchemeGNN(graph)(
        {k: torch.tensor(v) for k, v in inputs.items()}
    ).item()
    pyg = DirectModule(graph)(
        {k: torch.tensor(v) for k, v in inputs.items()}
    ).item()

    return seq, gnn, pyg


class TestLoopInsideFunction:
    """Letrec where the function body contains a loop."""

    def test_factorial_with_internal_loop(self):
        src = """
        (letrec ((factorial (lambda (n)
          (loop ((i n) (acc 1))
            (if (= i 0) acc (recur (- i 1) (* acc i)))))))
          (factorial 10))
        """
        seq, gnn, pyg = _all_eval(src)
        assert seq == pytest.approx(3628800.0)
        assert gnn == pytest.approx(3628800.0)
        assert pyg == pytest.approx(3628800.0)

    def test_sum_with_internal_loop(self):
        src = """
        (letrec ((sum-to (lambda (n)
          (loop ((i n) (acc 0))
            (if (= i 0) acc (recur (- i 1) (+ acc i)))))))
          (sum-to 100))
        """
        seq, gnn, pyg = _all_eval(src)
        assert seq == pytest.approx(5050.0)
        assert gnn == pytest.approx(5050.0)
        assert pyg == pytest.approx(5050.0)

    def test_gcd_with_internal_loop(self):
        src = """
        (letrec ((gcd (lambda (a b)
          (loop ((x a) (y b))
            (if (= y 0) x (recur y (modulo x y)))))))
          (gcd 48 18))
        """
        seq, gnn, pyg = _all_eval(src)
        assert seq == pytest.approx(6.0)
        assert gnn == pytest.approx(6.0)
        assert pyg == pytest.approx(6.0)

    def test_function_with_loop_and_input(self):
        src = """
        (letrec ((power (lambda (base exp)
          (loop ((e exp) (acc 1))
            (if (= e 0) acc (recur (- e 1) (* acc base)))))))
          (power x n))
        """
        seq, gnn, pyg = _all_eval(src, {"x": 2.0, "n": 10.0})
        assert seq == pytest.approx(1024.0)
        assert gnn == pytest.approx(1024.0)
        assert pyg == pytest.approx(1024.0)

    def test_graph_structure_loop_in_function(self):
        """Non-tail-recursive function with internal loop keeps letrec structure."""
        src = """
        (letrec ((f (lambda (n)
          (if (= n 0) 0
            (+ (f (- n 1))
               (loop ((i n) (acc 0))
                 (if (= i 0) acc (recur (- i 1) (+ acc 1)))))))))
          (f 3))
        """
        graph = compile_scheme(src)
        assert graph.has_functions
        func_body = graph.functions["f"]
        assert func_body.body_graph.has_loops
        seq, gnn, pyg = _all_eval(src)
        assert seq == pytest.approx(6.0)
        assert gnn == pytest.approx(6.0)
        assert pyg == pytest.approx(6.0)


class TestLetrecInsideLoop:
    """Loop body that defines and uses recursive functions via letrec."""

    def test_loop_with_recursive_function(self):
        src = """
        (loop ((i 5) (sum 0))
          (if (= i 0) sum
            (recur (- i 1)
              (+ sum (letrec ((factorial (lambda (n)
                        (if (= n 0) 1 (* n (factorial (- n 1)))))))
                       (factorial i))))))
        """
        seq, gnn, pyg = _all_eval(src)
        expected = 1.0 + 2.0 + 6.0 + 24.0 + 120.0  # 1!+2!+3!+4!+5! = 153
        assert seq == pytest.approx(expected)
        assert gnn == pytest.approx(expected)
        assert pyg == pytest.approx(expected)

    def test_loop_body_with_tree_recursion(self):
        src = """
        (loop ((i 5) (sum 0))
          (if (= i 0) sum
            (recur (- i 1)
              (+ sum (letrec ((fib (lambda (n)
                        (if (< n 2) n (+ (fib (- n 1)) (fib (- n 2)))))))
                       (fib i))))))
        """
        seq, gnn, pyg = _all_eval(src)
        expected = 1.0 + 1.0 + 2.0 + 3.0 + 5.0  # fib(1)+fib(2)+fib(3)+fib(4)+fib(5)
        assert seq == pytest.approx(expected)
        assert gnn == pytest.approx(expected)
        assert pyg == pytest.approx(expected)

    def test_graph_structure_letrec_in_loop(self):
        src = """
        (loop ((i 3) (sum 0))
          (if (= i 0) sum
            (recur (- i 1)
              (+ sum (letrec ((f (lambda (n) (if (= n 0) 1 (* n (f (- n 1)))))))
                       (f i))))))
        """
        graph = compile_scheme(src)
        assert graph.has_loops
        loop_nid = next(nid for nid, n in graph.nodes.items() if n.op_type == "loop")
        loop_body = graph.loops[loop_nid]
        assert loop_body.body_graph.has_functions


class TestNestedLoops:
    """Loop inside another loop body."""

    def test_nested_sum(self):
        """Sum of sums: sum_{i=1}^{3} sum_{j=1}^{i} j"""
        src = """
        (loop ((i 3) (total 0))
          (if (= i 0) total
            (recur (- i 1)
              (+ total (loop ((j i) (acc 0))
                         (if (= j 0) acc (recur (- j 1) (+ acc j))))))))
        """
        seq, gnn, pyg = _all_eval(src)
        expected = (1.0) + (1.0 + 2.0) + (1.0 + 2.0 + 3.0)  # 1 + 3 + 6 = 10
        assert seq == pytest.approx(expected)
        assert gnn == pytest.approx(expected)
        assert pyg == pytest.approx(expected)

    def test_multiplication_table_sum(self):
        """Sum of all products i*j for i=1..3, j=1..3."""
        src = """
        (loop ((i 3) (total 0))
          (if (= i 0) total
            (recur (- i 1)
              (+ total (loop ((j 3) (row 0))
                         (if (= j 0) row (recur (- j 1) (+ row (* i j)))))))))
        """
        seq, gnn, pyg = _all_eval(src)
        expected = sum(i * j for i in range(1, 4) for j in range(1, 4))
        assert seq == pytest.approx(float(expected))
        assert gnn == pytest.approx(float(expected))
        assert pyg == pytest.approx(float(expected))

    def test_nested_loop_with_input(self):
        src = """
        (loop ((i N) (total 0))
          (if (= i 0) total
            (recur (- i 1)
              (+ total (loop ((j i) (acc 0))
                         (if (= j 0) acc (recur (- j 1) (+ acc 1))))))))
        """
        seq, gnn, pyg = _all_eval(src, {"N": 4.0})
        expected = 1.0 + 2.0 + 3.0 + 4.0  # triangular number T(4) = 10
        assert seq == pytest.approx(expected)
        assert gnn == pytest.approx(expected)
        assert pyg == pytest.approx(expected)

    def test_graph_structure_nested_loops(self):
        src = """
        (loop ((i 3) (total 0))
          (if (= i 0) total
            (recur (- i 1)
              (+ total (loop ((j i) (acc 0))
                         (if (= j 0) acc (recur (- j 1) (+ acc j))))))))
        """
        graph = compile_scheme(src)
        assert graph.has_loops
        loop_nid = next(nid for nid, n in graph.nodes.items() if n.op_type == "loop")
        loop_body = graph.loops[loop_nid]
        assert loop_body.body_graph.has_loops


class TestLetrecWithLoopCallingFunction:
    """Letrec where the loop body calls a sibling function."""

    def test_loop_calling_function(self):
        """A function defines a helper and uses it inside a loop."""
        src = """
        (letrec ((square (lambda (x) (* x x))))
          (loop ((i 4) (sum 0))
            (if (= i 0) sum (recur (- i 1) (+ sum (square i))))))
        """
        seq, gnn, pyg = _all_eval(src)
        expected = 1.0 + 4.0 + 9.0 + 16.0  # 1^2 + 2^2 + 3^2 + 4^2 = 30
        assert seq == pytest.approx(expected)
        assert gnn == pytest.approx(expected)
        assert pyg == pytest.approx(expected)

    def test_loop_calling_recursive_function(self):
        """Loop body calls a recursive function defined in enclosing letrec."""
        src = """
        (letrec ((factorial (lambda (n) (if (= n 0) 1 (* n (factorial (- n 1)))))))
          (loop ((i 5) (sum 0))
            (if (= i 0) sum (recur (- i 1) (+ sum (factorial i))))))
        """
        seq, gnn, pyg = _all_eval(src)
        expected = 1.0 + 2.0 + 6.0 + 24.0 + 120.0  # sum of factorials
        assert seq == pytest.approx(expected)
        assert gnn == pytest.approx(expected)
        assert pyg == pytest.approx(expected)

    def test_two_functions_loop_calls_both(self):
        src = """
        (letrec ((square (lambda (x) (* x x)))
                 (cube (lambda (x) (* x (* x x)))))
          (loop ((i 3) (sum 0))
            (if (= i 0) sum (recur (- i 1) (+ sum (+ (square i) (cube i)))))))
        """
        seq, gnn, pyg = _all_eval(src)
        expected = sum(i**2 + i**3 for i in range(1, 4))
        assert seq == pytest.approx(float(expected))
        assert gnn == pytest.approx(float(expected))
        assert pyg == pytest.approx(float(expected))


class TestFunctionWithInternalLoopCalledFromLoop:
    """A function uses a loop internally, and is called from another loop."""

    def test_loop_calls_loop_function(self):
        src = """
        (letrec ((sum-to (lambda (n)
          (loop ((i n) (acc 0))
            (if (= i 0) acc (recur (- i 1) (+ acc i)))))))
          (loop ((k 4) (total 0))
            (if (= k 0) total (recur (- k 1) (+ total (sum-to k))))))
        """
        seq, gnn, pyg = _all_eval(src)
        expected = sum(sum(range(1, k + 1)) for k in range(1, 5))
        assert seq == pytest.approx(float(expected))
        assert gnn == pytest.approx(float(expected))
        assert pyg == pytest.approx(float(expected))


class TestCompositionWithTCO:
    """TCO'd functions still compose correctly with loops."""

    def test_tco_function_in_loop(self):
        """A tail-recursive function (TCO'd to loop) called from an outer loop."""
        src = """
        (letrec ((sum-to (lambda (n acc)
          (if (= n 0) acc (sum-to (- n 1) (+ acc n))))))
          (loop ((k 4) (total 0))
            (if (= k 0) total (recur (- k 1) (+ total (sum-to k 0))))))
        """
        seq, gnn, pyg = _all_eval(src)
        expected = sum(sum(range(1, k + 1)) for k in range(1, 5))
        assert seq == pytest.approx(float(expected))
        assert gnn == pytest.approx(float(expected))
        assert pyg == pytest.approx(float(expected))

    def test_tco_factorial_used_in_let(self):
        src = """
        (let ((result (letrec ((f (lambda (n acc) (if (= n 0) acc (f (- n 1) (* acc n))))))
                        (f 10 1))))
          (+ result 1))
        """
        seq, gnn, pyg = _all_eval(src)
        assert seq == pytest.approx(3628801.0)
        assert gnn == pytest.approx(3628801.0)
        assert pyg == pytest.approx(3628801.0)


class TestDefineWithInnerLetrec:
    """Define + inner letrec composition (previously broken: function dict overwrite)."""

    @staticmethod
    def _val(result):
        """Extract numeric value from a possibly-tagged result."""
        if hasattr(result, "__len__") and len(result) == 14:
            return float(result[10])
        return float(result)

    def test_define_with_inner_letrec(self):
        """A define'd function that uses letrec for an internal helper."""
        from neural_compiler.compiler import run_program
        src = """
        (define (sum-to n)
          (letrec ((go (lambda (i acc)
                         (if (= i 0) acc (go (- i 1) (+ acc i))))))
            (go n 0)))
        (sum-to 10)
        """
        result = run_program(src, prelude=True)
        assert self._val(result) == pytest.approx(55.0)

    def test_define_with_inner_letrec_mutual(self):
        """Define'd function with mutually recursive inner helpers."""
        from neural_compiler.compiler import run_program
        src = """
        (define (even-odd n)
          (letrec ((is-even (lambda (x) (if (= x 0) 1 (is-odd (- x 1)))))
                   (is-odd  (lambda (x) (if (= x 0) 0 (is-even (- x 1))))))
            (is-even n)))
        (even-odd 6)
        """
        result = run_program(src, prelude=True)
        assert self._val(result) == pytest.approx(1.0)

    def test_two_defines_with_inner_letrec(self):
        """Two independent define'd functions, each with inner letrec."""
        from neural_compiler.compiler import run_program
        src = """
        (define (fact n)
          (letrec ((go (lambda (i acc)
                         (if (= i 0) acc (go (- i 1) (* acc i))))))
            (go n 1)))
        (define (fib n)
          (letrec ((go (lambda (a b count)
                         (if (= count 0) a (go b (+ a b) (- count 1))))))
            (go 0 1 n)))
        (+ (fact 5) (fib 10))
        """
        result = run_program(src, prelude=True)
        assert self._val(result) == pytest.approx(120.0 + 55.0)

    def test_define_returning_lambda(self):
        """A define'd function that returns a closure."""
        from neural_compiler.compiler import run_program
        src = """
        (define (make-adder n) (lambda (x) (+ x n)))
        (let ((add5 (make-adder 5)))
          (add5 10))
        """
        result = run_program(src, prelude=True)
        assert self._val(result) == pytest.approx(15.0)

    def test_define_returning_lambda_composed(self):
        """Compose two define'd closures."""
        from neural_compiler.compiler import run_program
        src = """
        (define (make-adder n) (lambda (x) (+ x n)))
        (let ((add3 (make-adder 3))
              (add7 (make-adder 7)))
          (+ (add3 10) (add7 20)))
        """
        result = run_program(src, prelude=True)
        assert self._val(result) == pytest.approx(40.0)

    def test_define_with_lambda_inside_letrec(self):
        """A letrec function that returns a lambda."""
        from neural_compiler.compiler import run_program
        src = """
        (define (make-counter start)
          (letrec ((count (lambda (n) (lambda () n))))
            (count start)))
        (let ((c (make-counter 42)))
          (c))
        """
        result = run_program(src, prelude=True)
        assert self._val(result) == pytest.approx(42.0)


class TestMutualTailRecursion:
    """Integration tests for mutual tail recursion (dispatch loop TCO)."""

    def test_even_odd_small(self):
        """Basic is-even/is-odd mutual recursion."""
        src = """
        (letrec ((even? (lambda (n) (if (= n 0) 1 (odd? (- n 1)))))
                 (odd?  (lambda (n) (if (= n 0) 0 (even? (- n 1))))))
          (even? 10))
        """
        seq, gnn, pyg = _all_eval(src)
        assert seq == pytest.approx(1.0)
        assert gnn == pytest.approx(1.0)
        assert pyg == pytest.approx(1.0)

    def test_even_odd_returns_odd(self):
        src = """
        (letrec ((even? (lambda (n) (if (= n 0) 1 (odd? (- n 1)))))
                 (odd?  (lambda (n) (if (= n 0) 0 (even? (- n 1))))))
          (even? 7))
        """
        seq, gnn, pyg = _all_eval(src)
        assert seq == pytest.approx(0.0)
        assert gnn == pytest.approx(0.0)
        assert pyg == pytest.approx(0.0)

    def test_odd_entry_point(self):
        """Enter through the second binding."""
        src = """
        (letrec ((even? (lambda (n) (if (= n 0) 1 (odd? (- n 1)))))
                 (odd?  (lambda (n) (if (= n 0) 0 (even? (- n 1))))))
          (odd? 7))
        """
        seq, gnn, pyg = _all_eval(src)
        assert seq == pytest.approx(1.0)
        assert gnn == pytest.approx(1.0)
        assert pyg == pytest.approx(1.0)

    def test_mutual_with_accumulator(self):
        """Mutual recursion with accumulator passing."""
        src = """
        (letrec ((count-down-a (lambda (n acc)
                   (if (= n 0) acc
                     (count-down-b (- n 1) (+ acc 1)))))
                 (count-down-b (lambda (n acc)
                   (if (= n 0) acc
                     (count-down-a (- n 1) (+ acc 2))))))
          (count-down-a 6 0))
        """
        # n=6: a(6,0)→b(5,1)→a(4,3)→b(3,4)→a(2,6)→b(1,7)→a(0,9) = 9
        seq, gnn, pyg = _all_eval(src)
        assert seq == pytest.approx(9.0)
        assert gnn == pytest.approx(9.0)
        assert pyg == pytest.approx(9.0)

    def test_mutual_deep(self):
        """Mutual recursion deep enough to overflow without TCO."""
        src = """
        (letrec ((even? (lambda (n) (if (= n 0) 1 (odd? (- n 1)))))
                 (odd?  (lambda (n) (if (= n 0) 0 (even? (- n 1))))))
          (even? 200))
        """
        seq, gnn, pyg = _all_eval(src)
        assert seq == pytest.approx(1.0)
        assert gnn == pytest.approx(1.0)
        assert pyg == pytest.approx(1.0)

    def test_mutual_different_arities(self):
        """Mutual recursion where functions have different parameter counts."""
        src = """
        (letrec ((f (lambda (a b) (if (= a 0) b (g (+ a b)))))
                 (g (lambda (n) (if (= n 0) 0 (f (- n 1) 1)))))
          (f 3 1))
        """
        # f(3,1)→g(4)→f(3,1)→g(4)→... wait, that loops.
        # Let me rethink: f(3,1) → g(3+1=4) → f(3,1) → infinite loop
        # Need decreasing: f(a,b) → g(a-1) if a>0
        seq, gnn, pyg = _all_eval("""
        (letrec ((f (lambda (a b) (if (= a 0) b (g (- a 1)))))
                 (g (lambda (n) (if (= n 0) 42 (f n 1)))))
          (f 3 1))
        """)
        # f(3,1) → g(2) → f(2,1) → g(1) → f(1,1) → g(0) → 42
        assert seq == pytest.approx(42.0)
        assert gnn == pytest.approx(42.0)
        assert pyg == pytest.approx(42.0)

    def test_mutual_with_input(self):
        """Mutual TCO with external input variable."""
        src = """
        (letrec ((even? (lambda (n) (if (= n 0) 1 (odd? (- n 1)))))
                 (odd?  (lambda (n) (if (= n 0) 0 (even? (- n 1))))))
          (even? x))
        """
        seq, gnn, pyg = _all_eval(src, {"x": 8.0})
        assert seq == pytest.approx(1.0)
        assert gnn == pytest.approx(1.0)
        assert pyg == pytest.approx(1.0)


class TestCompositionGPU:
    """GPU tests for composition patterns."""

    @pytest.fixture
    def gpu_device(self):
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        pytest.skip("No GPU available")

    def test_loop_in_function_gpu(self, gpu_device):
        src = """
        (letrec ((sum-to (lambda (n)
          (loop ((i n) (acc 0))
            (if (= i 0) acc (recur (- i 1) (+ acc i)))))))
          (sum-to 10))
        """
        graph = compile_scheme(src)
        model = DirectModule(graph).to(gpu_device)
        assert model({}).item() == pytest.approx(55.0)

    def test_letrec_in_loop_gpu(self, gpu_device):
        src = """
        (loop ((i 4) (sum 0))
          (if (= i 0) sum
            (recur (- i 1)
              (+ sum (letrec ((f (lambda (n) (if (= n 0) 1 (* n (f (- n 1)))))))
                       (f i))))))
        """
        graph = compile_scheme(src)
        model = DirectModule(graph).to(gpu_device)
        expected = 1.0 + 2.0 + 6.0 + 24.0
        assert model({}).item() == pytest.approx(expected)

    def test_nested_loops_gpu(self, gpu_device):
        src = """
        (loop ((i 3) (total 0))
          (if (= i 0) total
            (recur (- i 1)
              (+ total (loop ((j i) (acc 0))
                         (if (= j 0) acc (recur (- j 1) (+ acc j))))))))
        """
        graph = compile_scheme(src)
        model = DirectModule(graph).to(gpu_device)
        assert model({}).item() == pytest.approx(10.0)

    def test_loop_calling_function_gpu(self, gpu_device):
        src = """
        (letrec ((square (lambda (x) (* x x))))
          (loop ((i 4) (sum 0))
            (if (= i 0) sum (recur (- i 1) (+ sum (square i))))))
        """
        graph = compile_scheme(src)
        model = DirectModule(graph).to(gpu_device)
        assert model({}).item() == pytest.approx(30.0)

    def test_composition_with_input_gpu(self, gpu_device):
        src = """
        (letrec ((power (lambda (base exp)
          (loop ((e exp) (acc 1))
            (if (= e 0) acc (recur (- e 1) (* acc base)))))))
          (power x n))
        """
        graph = compile_scheme(src, inputs={"x": None, "n": None})
        model = DirectModule(graph).to(gpu_device)
        result = model({
            "x": torch.tensor(3.0, device=gpu_device),
            "n": torch.tensor(4.0, device=gpu_device),
        })
        assert result.item() == pytest.approx(81.0)
