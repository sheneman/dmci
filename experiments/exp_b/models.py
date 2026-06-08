############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# models.py: Scientific models for Experiment B. Each model has: - A natural-language description (what a scientist would...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Scientific models for Experiment B.

Each model has:
- A natural-language description (what a scientist would tell an LLM)
- An interp_source: the self-hosted evaluator + the model as quoted data (DMCI)
- A direct_source: the model compiled directly (no interpreter, for comparison)
- A ground-truth Python function for generating training data
- Learnable parameters and their true values
- A complexity tier: 'equation' (paper 1 could handle) or 'program' (requires complete Scheme)

All training in Experiment B runs through the compiled self-hosted interpreter.
The program is DATA; the interpreter is the frozen differentiable module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


BOOTSTRAP_DIR = Path(__file__).parent.parent.parent / "bootstrap"
EVALUATOR_SOURCE = (BOOTSTRAP_DIR / "compiler.scm").read_text()


@dataclass
class ModelSpec:
    name: str
    domain: str
    tier: str  # 'equation' or 'program'
    nl_description: str
    interp_source: str   # evaluator + scheme-eval call (DMCI)
    direct_source: str   # direct compilation (no interpreter)
    param_names: list[str]
    target_values: dict[str, float]
    init_values: dict[str, float]
    input_names: list[str]
    ground_truth: Callable[..., float]
    x_range: tuple[float, float] = (0.1, 5.0)


def _all_input_names(spec: ModelSpec) -> list[str]:
    return spec.input_names + spec.param_names


def _make_env(names: list[str]) -> str:
    """Build Scheme environment list: (list (cons 'name name) ...)"""
    pairs = " ".join(f"(cons '{n} {n})" for n in names)
    return f"(list {pairs})"


# ===================================================================
# EQUATION TIER — could be expressed in paper 1's first-order language
# ===================================================================

M01_COULOMB = ModelSpec(
    name="M01_coulomb",
    domain="physics",
    tier="equation",
    nl_description=(
        "Coulomb's law: the electrostatic force between two point charges. "
        "Force equals a constant k times the product of charges q1 and q2, "
        "divided by the square of the distance r. Learn the constant k."
    ),
    interp_source=EVALUATOR_SOURCE + """
(scheme-eval '(/ (* k (* q1 q2)) (* r r))
             """ + _make_env(["k", "q1", "q2", "r"]) + ")",
    direct_source="""\
(define (coulomb r q1 q2 k)
  (/ (* k (* q1 q2)) (* r r)))
(coulomb r q1 q2 k)
""",
    param_names=["k"],
    target_values={"k": 8.99},
    init_values={"k": 1.0},
    input_names=["r", "q1", "q2"],
    ground_truth=lambda r, q1, q2, k=8.99: k * q1 * q2 / (r * r),
    x_range=(0.5, 3.0),
)

M02_BEER_LAMBERT = ModelSpec(
    name="M02_beer_lambert",
    domain="chemistry",
    tier="equation",
    nl_description=(
        "Beer-Lambert law: the absorbance of light through a solution. "
        "Absorbance A equals the molar absorptivity epsilon times "
        "the concentration c times the path length l. Learn epsilon."
    ),
    interp_source=EVALUATOR_SOURCE + """
(scheme-eval '(* epsilon (* c l))
             """ + _make_env(["epsilon", "c", "l"]) + ")",
    direct_source="""\
(define (absorbance c l epsilon)
  (* epsilon (* c l)))
(absorbance c l epsilon)
""",
    param_names=["epsilon"],
    target_values={"epsilon": 0.45},
    init_values={"epsilon": 1.0},
    input_names=["c", "l"],
    ground_truth=lambda c, l, epsilon=0.45: epsilon * c * l,
    x_range=(0.1, 5.0),
)

M03_MICHAELIS_MENTEN = ModelSpec(
    name="M03_michaelis_menten",
    domain="biochemistry",
    tier="equation",
    nl_description=(
        "Michaelis-Menten enzyme kinetics: reaction rate V equals "
        "Vmax times substrate concentration S divided by (Km + S). "
        "Learn Vmax and Km."
    ),
    interp_source=EVALUATOR_SOURCE + """
(scheme-eval '(/ (* Vmax S) (+ Km S))
             """ + _make_env(["Vmax", "Km", "S"]) + ")",
    direct_source="""\
(define (mm-rate S Vmax Km)
  (/ (* Vmax S) (+ Km S)))
(mm-rate S Vmax Km)
""",
    param_names=["Vmax", "Km"],
    target_values={"Vmax": 10.0, "Km": 2.5},
    init_values={"Vmax": 1.0, "Km": 1.0},
    input_names=["S"],
    ground_truth=lambda S, Vmax=10.0, Km=2.5: Vmax * S / (Km + S),
    x_range=(0.1, 10.0),
)

M04_ARRHENIUS = ModelSpec(
    name="M04_arrhenius",
    domain="chemistry",
    tier="equation",
    nl_description=(
        "Arrhenius-like rate equation: k = A * exp(-Ea * T). "
        "A simplified form where Ea absorbs the gas constant. "
        "Learn A and Ea."
    ),
    interp_source=EVALUATOR_SOURCE + """
(scheme-eval '(* A (exp (- 0 (* Ea T))))
             """ + _make_env(["A", "Ea", "T"]) + ")",
    direct_source="""\
(define (arrhenius T A Ea)
  (* A (exp (- 0 (* Ea T)))))
(arrhenius T A Ea)
""",
    param_names=["A", "Ea"],
    target_values={"A": 5.0, "Ea": 0.3},
    init_values={"A": 1.0, "Ea": 0.1},
    input_names=["T"],
    ground_truth=lambda T, A=5.0, Ea=0.3: A * math.exp(-Ea * T),
    x_range=(0.5, 5.0),
)

M05_DAMPED_OSCILLATOR = ModelSpec(
    name="M05_hookes_spring",  # data-key ID kept stable (used across cached LLM output, exp_h config/results, exp_b_summary.json)
    domain="physics",
    tier="equation",
    nl_description=(
        "A damped harmonic oscillator: the position of a mass on a spring "
        "with damping at time t. x(t) = A * exp(-b*t) * cos(omega*t). "
        "Learn A, b, and omega."
    ),
    interp_source=EVALUATOR_SOURCE + """
(scheme-eval '(* A (* (exp (- 0 (* b t))) (cos (* omega t))))
             """ + _make_env(["A", "b", "omega", "t"]) + ")",
    direct_source="""\
(define (damped-oscillator t A b omega)
  (* A (* (exp (- 0 (* b t))) (cos (* omega t)))))
(damped-oscillator t A b omega)
""",
    param_names=["A", "b", "omega"],
    target_values={"A": 2.0, "b": 0.3, "omega": 3.14},
    init_values={"A": 1.0, "b": 0.1, "omega": 1.0},
    input_names=["t"],
    ground_truth=lambda t, A=2.0, b=0.3, omega=3.14: A * math.exp(-b * t) * math.cos(omega * t),
    x_range=(0.0, 6.0),
)

M06_LOGISTIC_GROWTH = ModelSpec(
    name="M06_logistic_growth",
    domain="biology",
    tier="equation",
    nl_description=(
        "Logistic growth model: population P at time t. "
        "P(t) = K / (1 + C * exp(-r*t)) where C = (K-P0)/P0 and P0=1.0. "
        "Learn K and r."
    ),
    interp_source=EVALUATOR_SOURCE + """
(scheme-eval '(/ K (+ 1.0 (* (- K 1.0) (exp (- 0 (* r t))))))
             """ + _make_env(["K", "r", "t"]) + ")",
    direct_source="""\
(define (logistic t K r)
  (/ K (+ 1.0 (* (- K 1.0) (exp (- 0 (* r t)))))))
(logistic t K r)
""",
    param_names=["K", "r"],
    target_values={"K": 10.0, "r": 0.5},
    init_values={"K": 5.0, "r": 0.1},
    input_names=["t"],
    ground_truth=lambda t, K=10.0, r=0.5: K / (1.0 + (K - 1.0) * math.exp(-r * t)),
    x_range=(0.1, 10.0),
)

M07_POWER_LAW = ModelSpec(
    name="M07_power_law",
    domain="physics",
    tier="equation",
    nl_description=(
        "Allometric scaling / power law: y = a * x^b. Common in biology "
        "(metabolic rate vs body mass) and physics. Learn a and b."
    ),
    interp_source=EVALUATOR_SOURCE + """
(scheme-eval '(* a (pow x b))
             """ + _make_env(["a", "b", "x"]) + ")",
    direct_source="""\
(define (power-law x a b)
  (* a (pow x b)))
(power-law x a b)
""",
    param_names=["a", "b"],
    target_values={"a": 3.0, "b": 0.75},
    init_values={"a": 1.0, "b": 0.5},
    input_names=["x"],
    ground_truth=lambda x, a=3.0, b=0.75: a * (x ** b),
    x_range=(0.5, 10.0),
)


# ===================================================================
# PROGRAM TIER — requires closures, recursion, or higher-order functions
# These programs use scheme-eval-program for multi-form definitions
# ===================================================================

M08_EULER_ODE = ModelSpec(
    name="M08_euler_ode",
    domain="numerical_methods",
    tier="program",
    nl_description=(
        "Euler integration of a simple ODE: dy/dt = -k*y. "
        "Starting from y0=x, take 10 steps of size dt=0.1 to approximate y(1.0). "
        "Uses a recursive stepping chain. Learn k."
    ),
    interp_source=EVALUATOR_SOURCE + """
(scheme-eval-program
  (list
    '(define (step y) (+ y (* (- 0 (* k y)) 0.1)))
    '(step (step (step (step (step (step (step (step (step (step x)))))))))))
  """ + _make_env(["k", "x"]) + ")",
    direct_source="""\
(define (euler-step y k dt)
  (+ y (* (- 0 (* k y)) dt)))
(define (euler-10 y k dt)
  (let ((y1 (euler-step y k dt)))
  (let ((y2 (euler-step y1 k dt)))
  (let ((y3 (euler-step y2 k dt)))
  (let ((y4 (euler-step y3 k dt)))
  (let ((y5 (euler-step y4 k dt)))
  (let ((y6 (euler-step y5 k dt)))
  (let ((y7 (euler-step y6 k dt)))
  (let ((y8 (euler-step y7 k dt)))
  (let ((y9 (euler-step y8 k dt)))
  (euler-step y9 k dt)))))))))))
(euler-10 x k 0.1)
""",
    param_names=["k"],
    target_values={"k": 0.5},
    init_values={"k": 0.1},
    input_names=["x"],
    ground_truth=lambda x, k=0.5: x * (1.0 + (-k * 0.1)) ** 10,
    x_range=(0.5, 3.0),
)

M09_RECURSIVE_SERIES = ModelSpec(
    name="M09_taylor_exp",
    domain="numerical_methods",
    tier="program",
    nl_description=(
        "Taylor series approximation of exp(a*x) using recursive summation. "
        "sum_{i=0}^{8} (a*x)^i / i!. "
        "Requires a recursive function with factorial. Learn a."
    ),
    interp_source=EVALUATOR_SOURCE + """
(scheme-eval-program
  (list
    '(define (factorial n)
       (if (= n 0) 1.0 (* n (factorial (- n 1)))))
    '(define (taylor-term ax i)
       (/ (pow ax i) (factorial i)))
    '(define (taylor-sum ax n)
       (if (= n 0)
         1.0
         (+ (taylor-term ax n) (taylor-sum ax (- n 1)))))
    '(taylor-sum (* a x) 8))
  """ + _make_env(["a", "x"]) + ")",
    direct_source="""\
(define (factorial n)
  (if (= n 0) 1.0 (* n (factorial (- n 1)))))

(define (taylor-term ax i)
  (/ (pow ax i) (factorial i)))

(define (taylor-sum ax n)
  (if (= n 0)
    1.0
    (+ (taylor-term ax n) (taylor-sum ax (- n 1)))))

(taylor-sum (* a x) 8)
""",
    param_names=["a"],
    target_values={"a": 0.5},
    init_values={"a": 1.0},
    input_names=["x"],
    ground_truth=lambda x, a=0.5: math.exp(a * x),
    x_range=(0.1, 3.0),
)

M10_SMOOTH_ACTIVATION = ModelSpec(
    name="M10_smooth_activation",
    domain="ml",
    tier="program",
    nl_description=(
        "A smooth parametric activation function: "
        "f(x) = a * x * sigmoid(b * x), a soft-gating activation "
        "similar to SiLU/Swish with learnable scale and sharpness. Learn a and b."
    ),
    interp_source=EVALUATOR_SOURCE + """
(scheme-eval-program
  (list
    '(define (sigmoid z) (/ 1.0 (+ 1.0 (exp (- 0 z)))))
    '(* a (* x (sigmoid (* b x)))))
  """ + _make_env(["a", "b", "x"]) + ")",
    direct_source="""\
(define (sigmoid z) (/ 1.0 (+ 1.0 (exp (- 0 z)))))
(* a (* x (sigmoid (* b x))))
""",
    param_names=["a", "b"],
    target_values={"a": 1.0, "b": 2.0},
    init_values={"a": 0.5, "b": 0.5},
    input_names=["x"],
    ground_truth=lambda x, a=1.0, b=2.0: a * x / (1.0 + math.exp(-b * x)),
    x_range=(-3.0, 3.0),
)

M11_RECURSIVE_FILTER = ModelSpec(
    name="M11_recursive_filter",
    domain="signal_processing",
    tier="program",
    nl_description=(
        "A first-order IIR filter applied 8 times: y[n] = alpha*x + (1-alpha)*y[n-1]. "
        "Starting from y0=0, apply the filter 8 steps on input signal x. Learn alpha."
    ),
    interp_source=EVALUATOR_SOURCE + """
(scheme-eval-program
  (list
    '(define (step y) (+ (* alpha x) (* (- 1.0 alpha) y)))
    '(step (step (step (step (step (step (step (step 0.0)))))))))
  """ + _make_env(["alpha", "x"]) + ")",
    direct_source="""\
(define (iir-step x alpha y)
  (+ (* alpha x) (* (- 1.0 alpha) y)))
(define (iir-8 x alpha)
  (let ((y1 (iir-step x alpha 0.0)))
  (let ((y2 (iir-step x alpha y1)))
  (let ((y3 (iir-step x alpha y2)))
  (let ((y4 (iir-step x alpha y3)))
  (let ((y5 (iir-step x alpha y4)))
  (let ((y6 (iir-step x alpha y5)))
  (let ((y7 (iir-step x alpha y6)))
  (iir-step x alpha y7)))))))))
(iir-8 x alpha)
""",
    param_names=["alpha"],
    target_values={"alpha": 0.3},
    init_values={"alpha": 0.8},
    input_names=["x"],
    ground_truth=lambda x, alpha=0.3: x * (1.0 - (1.0 - alpha) ** 8),
    x_range=(0.5, 3.0),
)

M12_NEWTON_SQRT = ModelSpec(
    name="M12_newton_sqrt",
    domain="numerical_methods",
    tier="program",
    nl_description=(
        "Newton's method for computing sqrt(x) with a learnable initial guess "
        "scaling factor. guess0 = a * x. Each iteration: guess = (guess + x/guess) / 2. "
        "Run for 5 iterations. Learn a (optimal is 0.5)."
    ),
    interp_source=EVALUATOR_SOURCE + """
(scheme-eval-program
  (list
    '(define (newton-step g) (/ (+ g (/ x g)) 2.0))
    '(newton-step (newton-step (newton-step (newton-step (newton-step (* a x)))))))
  """ + _make_env(["a", "x"]) + ")",
    direct_source="""\
(define (newton-step guess x)
  (/ (+ guess (/ x guess)) 2.0))
(define (newton-sqrt x a)
  (let ((g0 (* a x)))
  (let ((g1 (newton-step g0 x)))
  (let ((g2 (newton-step g1 x)))
  (let ((g3 (newton-step g2 x)))
  (let ((g4 (newton-step g3 x)))
  (newton-step g4 x)))))))
(newton-sqrt x a)
""",
    param_names=["a"],
    target_values={"a": 0.5},
    init_values={"a": 1.0},
    input_names=["x"],
    ground_truth=lambda x, a=0.5: math.sqrt(x),
    x_range=(0.5, 10.0),
)

M13_HIGHER_ORDER_COMPOSITION = ModelSpec(
    name="M13_composed_transforms",
    domain="engineering",
    tier="program",
    nl_description=(
        "A data processing pipeline using higher-order function composition. "
        "Define a scale function and a shift function, then compose them: "
        "pipeline(x) = shift(scale(x)) = a*x + b. Expressed using "
        "explicit lambda and function application. Learn a and b."
    ),
    interp_source=EVALUATOR_SOURCE + """
(scheme-eval-program
  (list
    '(define (compose f g) (lambda (z) (f (g z))))
    '(define (make-scaler s) (lambda (z) (* s z)))
    '(define (make-shifter d) (lambda (z) (+ z d)))
    '(let ((pipeline (compose (make-shifter b) (make-scaler a))))
       (pipeline x)))
  """ + _make_env(["a", "b", "x"]) + ")",
    direct_source="""\
(define (compose f g) (lambda (z) (f (g z))))
(define (make-scaler s) (lambda (z) (* s z)))
(define (make-shifter d) (lambda (z) (+ z d)))
(let ((pipeline (compose (make-shifter b) (make-scaler a))))
  (pipeline x))
""",
    param_names=["a", "b"],
    target_values={"a": 2.5, "b": 1.0},
    init_values={"a": 1.0, "b": 0.0},
    input_names=["x"],
    ground_truth=lambda x, a=2.5, b=1.0: a * x + b,
    x_range=(0.5, 5.0),
)

M14_RULE_ENGINE = ModelSpec(
    name="M14_anomaly_scorer",
    domain="cybersecurity",
    tier="program",
    nl_description=(
        "An anomaly scoring rule engine: combine three risk features "
        "with learnable weights. "
        "score = w1*feature1 + w2*feature2 + w3*feature3. Learn w1, w2, w3."
    ),
    interp_source=EVALUATOR_SOURCE + """
(scheme-eval '(+ (* w1 f1) (+ (* w2 f2) (* w3 f3)))
             """ + _make_env(["w1", "w2", "w3", "f1", "f2", "f3"]) + ")",
    direct_source="""\
(define (score f1 f2 f3 w1 w2 w3)
  (+ (* w1 f1) (+ (* w2 f2) (* w3 f3))))
(score f1 f2 f3 w1 w2 w3)
""",
    param_names=["w1", "w2", "w3"],
    target_values={"w1": 0.5, "w2": 0.3, "w3": 0.2},
    init_values={"w1": 1.0, "w2": 1.0, "w3": 1.0},
    input_names=["f1", "f2", "f3"],
    ground_truth=lambda f1, f2, f3, w1=0.5, w2=0.3, w3=0.2: w1 * f1 + w2 * f2 + w3 * f3,
    x_range=(0.0, 5.0),
)

M15_RECURSIVE_POLYNOMIAL = ModelSpec(
    name="M15_horner_eval",
    domain="numerical_methods",
    tier="program",
    nl_description=(
        "Horner's method for evaluating a polynomial: "
        "p(x) = a0 + x*(a1 + x*(a2 + x*a3)). "
        "Implemented using recursive list processing. Learn a0, a1, a2, a3."
    ),
    interp_source=EVALUATOR_SOURCE + """
(scheme-eval-program
  (list
    '(define (horner x coeffs)
       (if (null? coeffs)
         0.0
         (+ (car coeffs) (* x (horner x (cdr coeffs))))))
    '(horner x (list a0 a1 a2 a3)))
  """ + _make_env(["a0", "a1", "a2", "a3", "x"]) + ")",
    direct_source="""\
(define (horner x coeffs)
  (if (null? coeffs)
    0.0
    (+ (car coeffs) (* x (horner x (cdr coeffs))))))
(horner x (list a0 a1 a2 a3))
""",
    param_names=["a0", "a1", "a2", "a3"],
    target_values={"a0": 1.0, "a1": -0.5, "a2": 0.3, "a3": -0.1},
    init_values={"a0": 0.0, "a1": 0.0, "a2": 0.0, "a3": 0.0},
    input_names=["x"],
    ground_truth=lambda x, a0=1.0, a1=-0.5, a2=0.3, a3=-0.1: a0 + x * (a1 + x * (a2 + x * a3)),
    x_range=(0.5, 3.0),
)


# ===================================================================
# Registry
# ===================================================================

EQUATION_MODELS = [
    M01_COULOMB,
    M02_BEER_LAMBERT,
    M03_MICHAELIS_MENTEN,
    M04_ARRHENIUS,
    M05_DAMPED_OSCILLATOR,
    M06_LOGISTIC_GROWTH,
    M07_POWER_LAW,
]

PROGRAM_MODELS = [
    M08_EULER_ODE,
    M09_RECURSIVE_SERIES,
    M10_SMOOTH_ACTIVATION,
    M11_RECURSIVE_FILTER,
    M12_NEWTON_SQRT,
    M13_HIGHER_ORDER_COMPOSITION,
    M14_RULE_ENGINE,
    M15_RECURSIVE_POLYNOMIAL,
]

ALL_MODELS = EQUATION_MODELS + PROGRAM_MODELS
MODEL_BY_NAME = {m.name: m for m in ALL_MODELS}
