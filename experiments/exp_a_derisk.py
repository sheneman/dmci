############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# exp_a_derisk.py: Experiment A de-risk: learn a constant inside a program by gradient descent through the compiled Scheme...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Experiment A de-risk: learn a constant inside a program by gradient descent
through the compiled Scheme interpreter.

Three levels of ambition, tested in order:

Level 1 — Fixed program, learnable external parameter:
    Compile (define (f x) (* alpha (* x x))), pass alpha as an input.
    Learn alpha = 0.5 from y = 0.5*x^2 data.
    This is baseline: gradients through a compiled recursive closure.

Level 2 — Learnable constant *inside* program data, through the interpreter:
    Compile the self-hosted evaluator. Feed it a program as quoted data:
        (scheme-eval '(* alpha (* x x)) env)
    where alpha is a learnable nn.Parameter injected into the environment.
    Gradients flow: loss -> interpreter dispatch -> heap reads -> arithmetic
    -> back to the parameter in the environment.
    This is the novel result.

Level 3 — (future) Relaxed dispatch for program structure search.
    Requires sigmoid-softened tagged_if. Not attempted here.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from neural_compiler.compiler import compile_program
from neural_compiler.evaluator import evaluate
from neural_compiler.runtime.tagged_value import make_float, unwrap_number


BOOTSTRAP_DIR = Path(__file__).parent.parent / "bootstrap"
EVALUATOR_SOURCE = (BOOTSTRAP_DIR / "compiler.scm").read_text()


def level_1_direct():
    """Level 1: learn alpha in a directly compiled program."""
    print("=" * 60)
    print("LEVEL 1: Direct compilation, learnable external parameter")
    print("=" * 60)

    source = """
    (define (f x alpha) (* alpha (* x x)))
    (f x alpha)
    """
    graph = compile_program(source, inputs={"x": None, "alpha": None}, prelude=True)

    alpha = nn.Parameter(torch.tensor(1.0))
    target_alpha = 0.5
    optimizer = torch.optim.Adam([alpha], lr=0.01)

    xs = torch.linspace(0.5, 3.0, 8)
    ys = target_alpha * xs ** 2

    for epoch in range(200):
        total_loss = torch.tensor(0.0)
        for x_val, y_val in zip(xs, ys):
            x_tagged = make_float(x_val)
            alpha_tagged = make_float(alpha)
            result = evaluate(graph, {"x": x_tagged, "alpha": alpha_tagged})
            pred = unwrap_number(result)
            total_loss = total_loss + (pred - y_val) ** 2

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if epoch % 50 == 0 or epoch == 199:
            print(f"  epoch {epoch:3d}  loss={total_loss.item():.6f}  "
                  f"alpha={alpha.item():.4f}  (target={target_alpha})")

    err = abs(alpha.item() - target_alpha)
    print(f"  Final alpha={alpha.item():.6f}, error={err:.6f}")
    assert err < 0.01, f"Level 1 failed: alpha={alpha.item()}, expected {target_alpha}"
    print("  PASSED\n")
    return True


def level_2_through_interpreter():
    """Level 2: learn alpha injected into an environment, through the
    compiled self-hosted interpreter evaluating (* alpha (* x x))."""
    print("=" * 60)
    print("LEVEL 2: Through compiled interpreter, learnable env constant")
    print("=" * 60)

    # The compiled interpreter evaluates the expression '(* alpha (* x x))
    # in an environment where alpha and x are bound.
    # We inject alpha as a learnable parameter and x as data.
    source = EVALUATOR_SOURCE + """
    (scheme-eval '(* alpha (* x x))
                 (list (cons 'alpha alpha) (cons 'x x)))
    """
    graph = compile_program(source, inputs={"x": None, "alpha": None}, prelude=True)

    alpha = nn.Parameter(torch.tensor(1.0))
    target_alpha = 0.5
    optimizer = torch.optim.Adam([alpha], lr=0.01)

    xs = torch.linspace(0.5, 3.0, 8)
    ys = target_alpha * xs ** 2

    for epoch in range(300):
        total_loss = torch.tensor(0.0)
        for x_val, y_val in zip(xs, ys):
            x_tagged = make_float(x_val)
            alpha_tagged = make_float(alpha)
            result = evaluate(graph, {"x": x_tagged, "alpha": alpha_tagged})
            pred = unwrap_number(result)
            total_loss = total_loss + (pred - y_val) ** 2

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if epoch % 50 == 0 or epoch == 299:
            print(f"  epoch {epoch:3d}  loss={total_loss.item():.6f}  "
                  f"alpha={alpha.item():.4f}  (target={target_alpha})")

    err = abs(alpha.item() - target_alpha)
    print(f"  Final alpha={alpha.item():.6f}, error={err:.6f}")
    assert err < 0.01, f"Level 2 failed: alpha={alpha.item()}, expected {target_alpha}"
    print("  PASSED\n")
    return True


def level_2b_multiconst():
    """Level 2b: learn TWO constants (a, b) in a + b*x^2 through the
    interpreter. Harder: two independent gradient paths through the
    same interpreter evaluation."""
    print("=" * 60)
    print("LEVEL 2b: Two constants through the interpreter")
    print("=" * 60)

    source = EVALUATOR_SOURCE + """
    (scheme-eval '(+ a (* b (* x x)))
                 (list (cons 'a a) (cons 'b b) (cons 'x x)))
    """
    graph = compile_program(source, inputs={"x": None, "a": None, "b": None}, prelude=True)

    a_param = nn.Parameter(torch.tensor(0.0))
    b_param = nn.Parameter(torch.tensor(1.0))
    target_a, target_b = 3.0, 0.5
    optimizer = torch.optim.Adam([a_param, b_param], lr=0.05)

    xs = torch.linspace(0.5, 3.0, 8)
    ys = target_a + target_b * xs ** 2

    for epoch in range(600):
        total_loss = torch.tensor(0.0)
        for x_val, y_val in zip(xs, ys):
            x_tagged = make_float(x_val)
            a_tagged = make_float(a_param)
            b_tagged = make_float(b_param)
            result = evaluate(graph, {"x": x_tagged, "a": a_tagged, "b": b_tagged})
            pred = unwrap_number(result)
            total_loss = total_loss + (pred - y_val) ** 2

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if epoch % 100 == 0 or epoch == 599:
            print(f"  epoch {epoch:3d}  loss={total_loss.item():.6f}  "
                  f"a={a_param.item():.4f} b={b_param.item():.4f}  "
                  f"(target a={target_a}, b={target_b})")

    err_a = abs(a_param.item() - target_a)
    err_b = abs(b_param.item() - target_b)
    print(f"  Final a={a_param.item():.6f}, b={b_param.item():.6f}")
    print(f"  Errors: a={err_a:.6f}, b={err_b:.6f}")
    assert err_a < 0.05 and err_b < 0.05, (
        f"Level 2b failed: a={a_param.item()}, b={b_param.item()}")
    print("  PASSED\n")
    return True


def level_2c_recursive():
    """Level 2c: learn a constant inside a RECURSIVE program evaluated
    by the interpreter. This is the hardest Level 2 variant: gradients
    must flow through recursive interpreter calls.

    Program: (define (poly x n) (if (= n 0) 0 (+ (* alpha x) (poly x (- n 1)))))
    This computes alpha*x*n. Learn alpha from data for y = 2*x*3 = 6x.
    """
    print("=" * 60)
    print("LEVEL 2c: Recursive program through the interpreter")
    print("=" * 60)

    source = EVALUATOR_SOURCE + """
    (scheme-eval-program
      (list
        '(define (poly x n)
           (if (= n 0) 0 (+ (* alpha x) (poly x (- n 1)))))
        '(poly x 3))
      (list (cons 'alpha alpha) (cons 'x x)))
    """
    graph = compile_program(source, inputs={"x": None, "alpha": None}, prelude=True)

    alpha = nn.Parameter(torch.tensor(1.0))
    target_alpha = 2.0  # poly(x,3) = alpha*x*3, target = 2*x*3 = 6x
    optimizer = torch.optim.Adam([alpha], lr=0.01)

    xs = torch.linspace(0.5, 3.0, 8)
    ys = target_alpha * xs * 3  # 6x

    for epoch in range(300):
        total_loss = torch.tensor(0.0)
        for x_val, y_val in zip(xs, ys):
            x_tagged = make_float(x_val)
            alpha_tagged = make_float(alpha)
            result = evaluate(graph, {"x": x_tagged, "alpha": alpha_tagged})
            pred = unwrap_number(result)
            total_loss = total_loss + (pred - y_val) ** 2

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if epoch % 50 == 0 or epoch == 299:
            print(f"  epoch {epoch:3d}  loss={total_loss.item():.6f}  "
                  f"alpha={alpha.item():.4f}  (target={target_alpha})")

    err = abs(alpha.item() - target_alpha)
    print(f"  Final alpha={alpha.item():.6f}, error={err:.6f}")
    assert err < 0.05, f"Level 2c failed: alpha={alpha.item()}, expected {target_alpha}"
    print("  PASSED\n")
    return True


if __name__ == "__main__":
    results = {}

    print("\nExperiment A: Differentiable Program Synthesis De-Risk\n")

    for name, fn in [
        ("Level 1", level_1_direct),
        ("Level 2", level_2_through_interpreter),
        ("Level 2b", level_2b_multiconst),
        ("Level 2c", level_2c_recursive),
    ]:
        try:
            results[name] = fn()
        except Exception as e:
            print(f"  FAILED: {e}\n")
            results[name] = False

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")

    all_passed = all(results.values())
    print(f"\nOverall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    sys.exit(0 if all_passed else 1)
