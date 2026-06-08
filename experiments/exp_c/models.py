############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# models.py: Recursive scientific models for Experiment C. Each model is an inherently recursive algorithm that cannot be...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Recursive scientific models for Experiment C.

Each model is an inherently recursive algorithm that cannot be expressed
as a first-order equation.  All models use recursive Scheme functions with
learnable parameters trained via gradient descent through the recursion.

Three domains:
  1. Coupled ODE systems  — multi-variable state evolved by recursive Euler steps
  2. Iterative convergence — fixed-point maps, continued fractions, nonlinear dynamics
  3. Recursive signal processing — filters with temporal feedback

Every model provides:
  - interp_source:  self-hosted evaluator + program as quoted data (DMCI)
  - direct_source:  same recursive program compiled directly
  - ground_truth:   Python function matching the Euler/iteration logic exactly
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
    category: str  # 'coupled_ode', 'iterative', 'recursive_filter'
    nl_description: str
    interp_source: str
    direct_source: str
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
# Ground-truth Python implementations
# (must match the Scheme programs exactly — same Euler steps, same n)
# ===================================================================

def _lv_ground_truth(x0, a=1.0, b=0.1):
    """Lotka-Volterra: 20 Euler steps, dt=0.05, y0=5.0, c=0.5, d=0.02."""
    x, y = x0, 5.0
    for _ in range(20):
        dx = 0.05 * (a * x - b * x * y)
        dy = 0.05 * (0.02 * x * y - 0.5 * y)
        x, y = x + dx, y + dy
    return x


def _sir_ground_truth(S0, beta=0.5, gamma=0.1):
    """SIR epidemic: 30 Euler steps, dt=0.1, I0=0.01."""
    s, i = S0, 0.01
    for _ in range(30):
        ds = 0.1 * (-beta * s * i)
        di = 0.1 * (beta * s * i - gamma * i)
        s, i = s + ds, i + di
    return s


def _decay_ground_truth(A0, lam1=0.3, lam2=0.1):
    """Radioactive decay chain: 25 Euler steps, dt=0.1, B0=0."""
    a, b = A0, 0.0
    for _ in range(25):
        da = 0.1 * (-lam1 * a)
        db = 0.1 * (lam1 * a - lam2 * b)
        a, b = a + da, b + db
    return b


def _logmap_ground_truth(x0, r=2.5):
    """Logistic map: 10 iterations."""
    x = x0
    for _ in range(10):
        x = r * x * (1.0 - x)
    return x


def _cf_ground_truth(x, a=0.5):
    """Generalized continued fraction: depth 8, bottom-up."""
    result = 0.0
    for _ in range(8):
        result = a * x / (1.0 + result)
    return result


def _pendulum_ground_truth(theta0, gL=5.0, b=0.3):
    """Damped pendulum: 20 Euler steps, dt=0.05, omega0=0."""
    theta, omega = theta0, 0.0
    for _ in range(20):
        dtheta = 0.05 * omega
        domega = 0.05 * (-gL * math.sin(theta) - b * omega)
        theta, omega = theta + dtheta, omega + domega
    return theta


def _iir2_ground_truth(x, a1=0.5, a2=-0.3, b0=1.0):
    """Second-order IIR filter: 12 steps, constant input."""
    y_prev1, y_prev2 = 0.0, 0.0
    for _ in range(12):
        y = a1 * y_prev1 + a2 * y_prev2 + b0 * x
        y_prev2, y_prev1 = y_prev1, y
    return y_prev1


def _ema_cascade_ground_truth(x, alpha=0.3, beta=0.5):
    """Cascaded EMA: 10 steps, two stages."""
    y1, y2 = 0.0, 0.0
    for _ in range(10):
        ny1 = alpha * x + (1.0 - alpha) * y1
        y2 = beta * ny1 + (1.0 - beta) * y2
        y1 = ny1
    return y2


# ===================================================================
# COUPLED ODE SYSTEMS
# ===================================================================

C01_LOTKA_VOLTERRA = ModelSpec(
    name="C01_lotka_volterra",
    domain="ecology",
    category="coupled_ode",
    nl_description=(
        "Lotka-Volterra predator-prey dynamics. Two coupled ODEs: "
        "dx/dt = a*x - b*x*y (prey), dy/dt = d*x*y - c*y (predator). "
        "Euler-integrate 20 steps from (x0, y0=5). Learn prey birth rate a "
        "and predation rate b."
    ),
    interp_source=EVALUATOR_SOURCE + """
(scheme-eval-program
  (list
    '(define (lv x y n)
       (if (= n 0) x
         (lv (+ x (* 0.05 (- (* a x) (* b (* x y)))))
             (+ y (* 0.05 (- (* 0.02 (* x y)) (* 0.5 y))))
             (- n 1))))
    '(lv x0 5.0 20))
  """ + _make_env(["a", "b", "x0"]) + ")",
    direct_source="""\
(define (lv x y n a b)
  (if (= n 0) x
    (lv (+ x (* 0.05 (- (* a x) (* b (* x y)))))
        (+ y (* 0.05 (- (* 0.02 (* x y)) (* 0.5 y))))
        (- n 1) a b)))
(lv x0 5.0 20 a b)
""",
    param_names=["a", "b"],
    target_values={"a": 1.0, "b": 0.1},
    init_values={"a": 0.5, "b": 0.5},
    input_names=["x0"],
    ground_truth=_lv_ground_truth,
    x_range=(2.0, 15.0),
)

C02_SIR_EPIDEMIC = ModelSpec(
    name="C02_sir_epidemic",
    domain="epidemiology",
    category="coupled_ode",
    nl_description=(
        "SIR epidemic model. Two coupled ODEs: "
        "dS/dt = -beta*S*I, dI/dt = beta*S*I - gamma*I. "
        "Euler-integrate 30 steps from (S0, I0=0.01). "
        "Learn transmission rate beta and recovery rate gamma."
    ),
    interp_source=EVALUATOR_SOURCE + """
(scheme-eval-program
  (list
    '(define (sir s i n)
       (if (= n 0) s
         (sir (+ s (* 0.1 (- 0 (* beta (* s i)))))
              (+ i (* 0.1 (- (* beta (* s i)) (* gamma i))))
              (- n 1))))
    '(sir S0 0.01 30))
  """ + _make_env(["beta", "gamma", "S0"]) + ")",
    direct_source="""\
(define (sir s i n beta gamma)
  (if (= n 0) s
    (sir (+ s (* 0.1 (- 0 (* beta (* s i)))))
         (+ i (* 0.1 (- (* beta (* s i)) (* gamma i))))
         (- n 1) beta gamma)))
(sir S0 0.01 30 beta gamma)
""",
    param_names=["beta", "gamma"],
    target_values={"beta": 0.5, "gamma": 0.1},
    init_values={"beta": 0.2, "gamma": 0.5},
    input_names=["S0"],
    ground_truth=_sir_ground_truth,
    x_range=(0.5, 0.99),
)

C03_DECAY_CHAIN = ModelSpec(
    name="C03_decay_chain",
    domain="nuclear_physics",
    category="coupled_ode",
    nl_description=(
        "Radioactive decay chain A -> B -> C. Two coupled ODEs: "
        "dA/dt = -lam1*A, dB/dt = lam1*A - lam2*B. "
        "Euler-integrate 25 steps from (A0, B0=0). "
        "Learn decay constants lam1 and lam2."
    ),
    interp_source=EVALUATOR_SOURCE + """
(scheme-eval-program
  (list
    '(define (decay aa bb n)
       (if (= n 0) bb
         (decay (+ aa (* 0.1 (- 0 (* lam1 aa))))
                (+ bb (* 0.1 (- (* lam1 aa) (* lam2 bb))))
                (- n 1))))
    '(decay A0 0.0 25))
  """ + _make_env(["lam1", "lam2", "A0"]) + ")",
    direct_source="""\
(define (decay aa bb n lam1 lam2)
  (if (= n 0) bb
    (decay (+ aa (* 0.1 (- 0 (* lam1 aa))))
           (+ bb (* 0.1 (- (* lam1 aa) (* lam2 bb))))
           (- n 1) lam1 lam2)))
(decay A0 0.0 25 lam1 lam2)
""",
    param_names=["lam1", "lam2"],
    target_values={"lam1": 0.3, "lam2": 0.1},
    init_values={"lam1": 0.1, "lam2": 0.5},
    input_names=["A0"],
    ground_truth=_decay_ground_truth,
    x_range=(1.0, 10.0),
)


# ===================================================================
# ITERATIVE CONVERGENCE
# ===================================================================

C04_LOGISTIC_MAP = ModelSpec(
    name="C04_logistic_map",
    domain="nonlinear_dynamics",
    category="iterative",
    nl_description=(
        "Logistic map: x_{n+1} = r * x_n * (1 - x_n). "
        "Iterate 10 times from x0. Learn the growth parameter r. "
        "Classic model of population dynamics and chaos theory."
    ),
    interp_source=EVALUATOR_SOURCE + """
(scheme-eval-program
  (list
    '(define (logmap x n)
       (if (= n 0) x
         (logmap (* r (* x (- 1.0 x))) (- n 1))))
    '(logmap x0 10))
  """ + _make_env(["r", "x0"]) + ")",
    direct_source="""\
(define (logmap x n r)
  (if (= n 0) x
    (logmap (* r (* x (- 1.0 x))) (- n 1) r)))
(logmap x0 10 r)
""",
    param_names=["r"],
    target_values={"r": 2.5},
    init_values={"r": 1.5},
    input_names=["x0"],
    ground_truth=_logmap_ground_truth,
    x_range=(0.1, 0.9),
)

C05_CONTINUED_FRACTION = ModelSpec(
    name="C05_continued_fraction",
    domain="mathematical_analysis",
    category="iterative",
    nl_description=(
        "Generalized continued fraction: f(x) = a*x / (1 + a*x / (1 + ...)). "
        "Computed to depth 8 via tree recursion. "
        "Learn the scaling parameter a."
    ),
    interp_source=EVALUATOR_SOURCE + """
(scheme-eval-program
  (list
    '(define (cf x n)
       (if (= n 0) 0.0
         (/ (* a x) (+ 1.0 (cf x (- n 1))))))
    '(cf x 8))
  """ + _make_env(["a", "x"]) + ")",
    direct_source="""\
(define (cf x n a)
  (if (= n 0) 0.0
    (/ (* a x) (+ 1.0 (cf x (- n 1) a)))))
(cf x 8 a)
""",
    param_names=["a"],
    target_values={"a": 0.5},
    init_values={"a": 1.5},
    input_names=["x"],
    ground_truth=_cf_ground_truth,
    x_range=(0.5, 5.0),
)

C06_DAMPED_PENDULUM = ModelSpec(
    name="C06_damped_pendulum",
    domain="physics",
    category="iterative",
    nl_description=(
        "Damped nonlinear pendulum: dtheta/dt = omega, "
        "domega/dt = -(g/L)*sin(theta) - b*omega. "
        "Euler-integrate 20 steps from (theta0, omega0=0). "
        "Learn g/L ratio and damping coefficient b. "
        "The sin(theta) nonlinearity makes this irreducible to a flat equation."
    ),
    interp_source=EVALUATOR_SOURCE + """
(scheme-eval-program
  (list
    '(define (pend theta omega n)
       (if (= n 0) theta
         (pend (+ theta (* 0.05 omega))
               (+ omega (* 0.05 (- 0 (+ (* gL (sin theta)) (* b omega)))))
               (- n 1))))
    '(pend theta0 0.0 20))
  """ + _make_env(["gL", "b", "theta0"]) + ")",
    direct_source="""\
(define (pend theta omega n gL b)
  (if (= n 0) theta
    (pend (+ theta (* 0.05 omega))
          (+ omega (* 0.05 (- 0 (+ (* gL (sin theta)) (* b omega)))))
          (- n 1) gL b)))
(pend theta0 0.0 20 gL b)
""",
    param_names=["gL", "b"],
    target_values={"gL": 5.0, "b": 0.3},
    init_values={"gL": 2.0, "b": 0.1},
    input_names=["theta0"],
    ground_truth=_pendulum_ground_truth,
    x_range=(0.1, 1.5),
)


# ===================================================================
# RECURSIVE SIGNAL PROCESSING
# ===================================================================

C07_IIR_FILTER = ModelSpec(
    name="C07_iir_filter",
    domain="signal_processing",
    category="recursive_filter",
    nl_description=(
        "Second-order IIR (infinite impulse response) filter: "
        "y[n] = a1*y[n-1] + a2*y[n-2] + b0*x. "
        "Process constant input x for 12 steps. "
        "Two-step memory makes this inherently recursive. "
        "Learn filter coefficients a1, a2, b0."
    ),
    interp_source=EVALUATOR_SOURCE + """
(scheme-eval-program
  (list
    '(define (iir y1 y2 n)
       (if (= n 0) y1
         (iir (+ (* a1 y1) (+ (* a2 y2) (* b0 x)))
              y1
              (- n 1))))
    '(iir 0.0 0.0 12))
  """ + _make_env(["a1", "a2", "b0", "x"]) + ")",
    direct_source="""\
(define (iir y1 y2 n a1 a2 b0 x)
  (if (= n 0) y1
    (iir (+ (* a1 y1) (+ (* a2 y2) (* b0 x)))
         y1
         (- n 1) a1 a2 b0 x)))
(iir 0.0 0.0 12 a1 a2 b0 x)
""",
    param_names=["a1", "a2", "b0"],
    target_values={"a1": 0.5, "a2": -0.3, "b0": 1.0},
    init_values={"a1": 0.1, "a2": 0.0, "b0": 0.5},
    input_names=["x"],
    ground_truth=_iir2_ground_truth,
    x_range=(0.5, 5.0),
)

C08_CASCADED_EMA = ModelSpec(
    name="C08_cascaded_ema",
    domain="signal_processing",
    category="recursive_filter",
    nl_description=(
        "Cascaded exponential moving average: two-stage recursive filter. "
        "Stage 1: y1[n] = alpha*x + (1-alpha)*y1[n-1]. "
        "Stage 2: y2[n] = beta*y1[n] + (1-beta)*y2[n-1]. "
        "10 steps. Learn smoothing constants alpha and beta."
    ),
    interp_source=EVALUATOR_SOURCE + """
(scheme-eval-program
  (list
    '(define (ema y1 y2 n)
       (if (= n 0) y2
         (ema (+ (* alpha x) (* (- 1.0 alpha) y1))
              (+ (* beta (+ (* alpha x) (* (- 1.0 alpha) y1)))
                 (* (- 1.0 beta) y2))
              (- n 1))))
    '(ema 0.0 0.0 10))
  """ + _make_env(["alpha", "beta", "x"]) + ")",
    direct_source="""\
(define (ema y1 y2 n alpha beta x)
  (if (= n 0) y2
    (ema (+ (* alpha x) (* (- 1.0 alpha) y1))
         (+ (* beta (+ (* alpha x) (* (- 1.0 alpha) y1)))
            (* (- 1.0 beta) y2))
         (- n 1) alpha beta x)))
(ema 0.0 0.0 10 alpha beta x)
""",
    param_names=["alpha", "beta"],
    target_values={"alpha": 0.3, "beta": 0.5},
    init_values={"alpha": 0.8, "beta": 0.8},
    input_names=["x"],
    ground_truth=_ema_cascade_ground_truth,
    x_range=(0.5, 5.0),
)


# ===================================================================
# Registry
# ===================================================================

COUPLED_ODE_MODELS = [
    C01_LOTKA_VOLTERRA,
    C02_SIR_EPIDEMIC,
    C03_DECAY_CHAIN,
]

ITERATIVE_MODELS = [
    C04_LOGISTIC_MAP,
    C05_CONTINUED_FRACTION,
    C06_DAMPED_PENDULUM,
]

FILTER_MODELS = [
    C07_IIR_FILTER,
    C08_CASCADED_EMA,
]

ALL_MODELS = COUPLED_ODE_MODELS + ITERATIVE_MODELS + FILTER_MODELS
MODEL_BY_NAME = {m.name: m for m in ALL_MODELS}
