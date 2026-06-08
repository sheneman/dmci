############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# config.py: Experiment F configuration: LLM-in-the-Loop Scientific Model Discovery.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Experiment F configuration: LLM-in-the-Loop Scientific Model Discovery."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class TargetSpec:
    name: str
    description: str
    ground_truth_fn: Callable[[float], float]
    true_params: dict[str, float]
    difficulty: str
    expected_iterations: int


TARGETS = [
    TargetSpec(
        name="F1_exponential_decay",
        description="Exponential decay",
        ground_truth_fn=lambda x: 2.5 * math.exp(-0.8 * x),
        true_params={"a": 2.5, "b": 0.8},
        difficulty="easy",
        expected_iterations=1,
    ),
    TargetSpec(
        name="F2_damped_oscillation",
        description="Damped oscillation (product of sine and exponential)",
        ground_truth_fn=lambda x: 1.5 * math.sin(4.0 * x) * math.exp(-0.3 * x),
        true_params={"a": 1.5, "b": 4.0, "c": 0.3},
        difficulty="medium",
        expected_iterations=2,
    ),
    TargetSpec(
        name="F3_decay_plus_sine",
        description="Sum of exponential decay and sine wave",
        ground_truth_fn=lambda x: 2.0 * math.exp(-0.5 * x) + 1.0 * math.sin(3.0 * x),
        true_params={"a": 2.0, "b": 0.5, "c": 1.0, "d": 3.0},
        difficulty="medium",
        expected_iterations=3,
    ),
    TargetSpec(
        name="F4_logistic_growth",
        description="Logistic/sigmoid growth curve",
        ground_truth_fn=lambda x: 5.0 / (1.0 + math.exp(-2.0 * (x - 2.5))),
        true_params={"a": 5.0, "b": 2.0, "c": 2.5},
        difficulty="easy-medium",
        expected_iterations=1,
    ),
]

N_TARGETS = len(TARGETS)


@dataclass
class ExpFConfig:
    max_epochs: int = 2000
    lr: float = 0.01
    convergence_threshold: float = 1e-4
    early_stop_patience: int = 200
    n_data_points: int = 64
    x_lo: float = 0.0
    x_hi: float = 5.0
    noise_std: float = 0.02
    max_iterations: int = 5
    mse_threshold: float = 1e-3
    llm_temperature: float = 0.7
    llm_max_tokens: int = 16384
    llm_max_retries: int = 2
    n_seeds: int = 3
    param_names: list[str] = field(
        default_factory=lambda: ["a", "b", "c", "d"]
    )


DEFAULT = ExpFConfig()
