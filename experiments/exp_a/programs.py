############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# programs.py: Experiment A program suite: the Scheme test programs and their ground-truth parameters
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


BOOTSTRAP_DIR = Path(__file__).parent.parent.parent / "bootstrap"
EVALUATOR_SOURCE = (BOOTSTRAP_DIR / "compiler.scm").read_text()


@dataclass
class ProgramSpec:
    name: str
    description: str
    difficulty: int

    # For compiled-interpreter methods: Scheme expression or program forms
    # passed to scheme-eval or scheme-eval-program
    interp_source: str  # full source including evaluator + scheme-eval call

    # For direct compilation: the target program itself (no interpreter)
    direct_source: str

    param_names: list[str]
    target_values: dict[str, float]
    init_values: dict[str, float]
    input_names: list[str]  # non-learnable inputs (e.g., ["x"])

    data_fn: Callable[[float], float]

    # For hand-coded interpreter: Python AST representation
    # Expression programs use hc_expr; multi-form programs use hc_program
    hc_expr: object = None       # nested list AST for single-expression
    hc_program: object = None    # list of (define_name, define_params, define_body) + final expr


def _all_input_names(spec: ProgramSpec) -> list[str]:
    return spec.input_names + spec.param_names


# ---------------------------------------------------------------------------
# P1: Single constant, arithmetic
# ---------------------------------------------------------------------------

P1 = ProgramSpec(
    name="P1_single_const",
    description="Single learnable constant in arithmetic expression",
    difficulty=1,
    interp_source=EVALUATOR_SOURCE + """
(scheme-eval '(* alpha (* x x))
             (list (cons 'alpha alpha) (cons 'x x)))
""",
    direct_source="(define (f x alpha) (* alpha (* x x)))\n(f x alpha)",
    param_names=["alpha"],
    target_values={"alpha": 0.5},
    init_values={"alpha": 1.0},
    input_names=["x"],
    data_fn=lambda x: 0.5 * x * x,
    hc_expr=["*", "alpha", ["*", "x", "x"]],
)

# ---------------------------------------------------------------------------
# P2: Multiple constants, arithmetic
# ---------------------------------------------------------------------------

P2 = ProgramSpec(
    name="P2_multi_const",
    description="Two learnable constants in arithmetic expression",
    difficulty=2,
    interp_source=EVALUATOR_SOURCE + """
(scheme-eval '(+ a (* b (* x x)))
             (list (cons 'a a) (cons 'b b) (cons 'x x)))
""",
    direct_source="(define (f x a b) (+ a (* b (* x x))))\n(f x a b)",
    param_names=["a", "b"],
    target_values={"a": 3.0, "b": 0.5},
    init_values={"a": 0.0, "b": 1.0},
    input_names=["x"],
    data_fn=lambda x: 3.0 + 0.5 * x * x,
    hc_expr=["+", "a", ["*", "b", ["*", "x", "x"]]],
)

# ---------------------------------------------------------------------------
# P3: Recursive with constant
# ---------------------------------------------------------------------------

P3 = ProgramSpec(
    name="P3_recursive",
    description="Learnable constant inside recursive function",
    difficulty=3,
    interp_source=EVALUATOR_SOURCE + """
(scheme-eval-program
  (list
    '(define (poly x n)
       (if (= n 0) 0 (+ (* alpha x) (poly x (- n 1)))))
    '(poly x 3))
  (list (cons 'alpha alpha) (cons 'x x)))
""",
    direct_source="""\
(define (poly x n alpha)
  (if (= n 0) 0 (+ (* alpha x) (poly x (- n 1) alpha))))
(poly x 3 alpha)
""",
    param_names=["alpha"],
    target_values={"alpha": 2.0},
    init_values={"alpha": 1.0},
    input_names=["x"],
    data_fn=lambda x: 2.0 * x * 3,  # alpha * x * n
    hc_program={
        "defines": [
            ("poly", ["x", "n"], [
                "if", ["=", "n", 0],
                0,
                ["+", ["*", "alpha", "x"], ["poly", "x", ["-", "n", 1]]]
            ]),
        ],
        "body": ["poly", "x", 3],
    },
)

# ---------------------------------------------------------------------------
# P4: Higher-order with constant (closure application)
# ---------------------------------------------------------------------------

P4 = ProgramSpec(
    name="P4_higher_order",
    description="Learnable constant in closure passed to higher-order function",
    difficulty=4,
    interp_source=EVALUATOR_SOURCE + """
(scheme-eval-program
  (list
    '(define (twice f x) (f (f x)))
    '(twice (lambda (y) (+ y alpha)) x))
  (list (cons 'alpha alpha) (cons 'x x)))
""",
    direct_source="""\
(define (twice f x) (f (f x)))
(twice (lambda (y) (+ y alpha)) x)
""",
    param_names=["alpha"],
    target_values={"alpha": 1.5},
    init_values={"alpha": 0.0},
    input_names=["x"],
    # twice (lambda (y) (+ y alpha)) x = x + 2*alpha
    data_fn=lambda x: x + 2 * 1.5,
    hc_program={
        "defines": [
            ("twice", ["f", "x"], ["f", ["f", "x"]]),
        ],
        "body": ["twice", ["lambda", ["y"], ["+", "y", "alpha"]], "x"],
    },
)

# ---------------------------------------------------------------------------
# P5: Multi-function program
# ---------------------------------------------------------------------------

P5 = ProgramSpec(
    name="P5_multi_function",
    description="Learnable constants in separate function scopes",
    difficulty=5,
    interp_source=EVALUATOR_SOURCE + """
(scheme-eval-program
  (list
    '(define (f x) (* a x))
    '(define (g x) (+ b (* x x)))
    '(+ (f x) (g x)))
  (list (cons 'a a) (cons 'b b) (cons 'x x)))
""",
    direct_source="""\
(define (f x a) (* a x))
(define (g x b) (+ b (* x x)))
(+ (f x a) (g x b))
""",
    param_names=["a", "b"],
    target_values={"a": 2.0, "b": 1.5},
    init_values={"a": 1.0, "b": 0.0},
    input_names=["x"],
    # f(x) + g(x) = a*x + b + x^2
    data_fn=lambda x: 2.0 * x + 1.5 + x * x,
    hc_program={
        "defines": [
            ("f", ["x"], ["*", "a", "x"]),
            ("g", ["x"], ["+", "b", ["*", "x", "x"]]),
        ],
        "body": ["+", ["f", "x"], ["g", "x"]],
    },
)

# ---------------------------------------------------------------------------
# P6: Composed functions
# ---------------------------------------------------------------------------

P6 = ProgramSpec(
    name="P6_composed",
    description="Learnable constants in composed function chain (identifiable)",
    difficulty=6,
    # f(g(x)) = a*(x+b)^2 + c — all params identifiable via x^2, x, const terms
    interp_source=EVALUATOR_SOURCE + """
(scheme-eval-program
  (list
    '(define (f x) (+ (* a (* x x)) c))
    '(define (g x) (+ x b))
    '(f (g x)))
  (list (cons 'a a) (cons 'b b) (cons 'c c) (cons 'x x)))
""",
    direct_source="""\
(define (f x a c) (+ (* a (* x x)) c))
(define (g x b) (+ x b))
(f (g x b) a c)
""",
    param_names=["a", "b", "c"],
    target_values={"a": 2.0, "b": 0.5, "c": 1.0},
    init_values={"a": 1.0, "b": 0.0, "c": 0.0},
    input_names=["x"],
    # f(g(x)) = a*(x+b)^2 + c = 2*(x+0.5)^2 + 1
    data_fn=lambda x: 2.0 * (x + 0.5) ** 2 + 1.0,
    hc_expr=None,
    hc_program={
        "defines": [
            ("f", ["x"], ["+", ["*", "a", ["*", "x", "x"]], "c"]),
            ("g", ["x"], ["+", "x", "b"]),
        ],
        "body": ["f", ["g", "x"]],
    },
)

ALL_PROGRAMS: list[ProgramSpec] = [P1, P2, P3, P4, P5, P6]

PROGRAMS_BY_NAME: dict[str, ProgramSpec] = {p.name: p for p in ALL_PROGRAMS}
