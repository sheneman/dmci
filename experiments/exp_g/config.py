############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# config.py: Experiment G configuration: Runtime Compositional Scientific Modeling.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Experiment G configuration: Runtime Compositional Scientific Modeling."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class ModuleDef:
    name: str
    template: str
    param_names: list[str]
    description: str


MODULE_LIBRARY = [
    ModuleDef(
        name="exponential_decay",
        template="(* {pfx}a (exp (* (- 0 {pfx}b) x)))",
        param_names=["a", "b"],
        description="a * exp(-b*x)",
    ),
    ModuleDef(
        name="oscillation",
        template="(* {pfx}a (sin (+ (* {pfx}b x) {pfx}c)))",
        param_names=["a", "b", "c"],
        description="a * sin(b*x + c)",
    ),
    ModuleDef(
        name="polynomial2",
        template="(+ (* {pfx}a (* x x)) (+ (* {pfx}b x) {pfx}c))",
        param_names=["a", "b", "c"],
        description="a*x^2 + b*x + c",
    ),
    ModuleDef(
        name="sigmoid",
        template="(/ {pfx}a (+ 1.0 (exp (* (- 0 {pfx}b) (- x {pfx}c)))))",
        param_names=["a", "b", "c"],
        description="a / (1 + exp(-b*(x-c)))",
    ),
    ModuleDef(
        name="power_law",
        template="(* {pfx}a (pow x {pfx}b))",
        param_names=["a", "b"],
        description="a * x^b",
    ),
    ModuleDef(
        name="gaussian",
        template=(
            "(* {pfx}a (exp (* (- 0 1) (/ (* (- x {pfx}b) "
            "(- x {pfx}b)) (* 2 (* {pfx}c {pfx}c))))))"
        ),
        param_names=["a", "b", "c"],
        description="a * exp(-(x-b)^2 / (2*c^2))",
    ),
]

MODULE_BY_NAME = {m.name: m for m in MODULE_LIBRARY}


@dataclass
class TestProblem:
    name: str
    description: str
    ground_truth_fn: Callable[[float], float]
    correct_composition: tuple[str, str, str]  # (op, module1, module2)
    true_params: dict[str, float]
    x_range: tuple[float, float] = (0.0, 5.0)


PROBLEMS = [
    TestProblem(
        name="G1_decay_plus_oscillation",
        description="y = 2.0*exp(-0.5*x) + 1.5*sin(3.0*x)",
        ground_truth_fn=lambda x: (
            2.0 * math.exp(-0.5 * x) + 1.5 * math.sin(3.0 * x)),
        correct_composition=("sum", "exponential_decay", "oscillation"),
        true_params={
            "m1_a": 2.0, "m1_b": 0.5,
            "m2_a": 1.5, "m2_b": 3.0, "m2_c": 0.0,
        },
    ),
    TestProblem(
        name="G2_damped_oscillation",
        description="y = 3.0*sin(2.0*x)*exp(-0.3*x)",
        ground_truth_fn=lambda x: (
            3.0 * math.sin(2.0 * x) * math.exp(-0.3 * x)),
        correct_composition=("product", "oscillation", "exponential_decay"),
        true_params={
            "m1_a": 3.0, "m1_b": 2.0, "m1_c": 0.0,
            "m2_a": 1.0, "m2_b": 0.3,
        },
    ),
    TestProblem(
        name="G3_sigmoid_of_polynomial",
        description="y = 5.0 / (1 + exp(-2.0*(x^2 + 0.5*x - 2.5)))",
        ground_truth_fn=lambda x: (
            5.0 / (1.0 + math.exp(
                max(-500, min(500, -2.0 * (x**2 + 0.5*x - 2.5)))))),
        correct_composition=("chain", "sigmoid", "polynomial2"),
        true_params={
            "m1_a": 5.0, "m1_b": 2.0, "m1_c": 0.0,
            "m2_a": 1.0, "m2_b": 0.5, "m2_c": -2.5,
        },
        x_range=(0.0, 3.0),
    ),
]

N_PROBLEMS = len(PROBLEMS)


@dataclass
class ExpGConfig:
    max_epochs: int = 2000
    lr: float = 0.01
    convergence_threshold: float = 1e-4
    early_stop_patience: int = 200
    n_data_points: int = 64
    noise_std: float = 0.02
    n_seeds: int = 5


DEFAULT = ExpGConfig()
