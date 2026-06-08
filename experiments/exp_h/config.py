############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# config.py: Experiment H configuration: Batched GPU Parallelization of DMCI.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Experiment H configuration: Batched GPU Parallelization of DMCI."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ExpHConfig:
    batch_sizes: list[int] = field(
        default_factory=lambda: [1, 8, 32, 128, 512, 1024, 4096]
    )
    population_sizes: list[int] = field(
        default_factory=lambda: [1, 10, 100, 1000]
    )
    n_data_points: int = 64
    x_range: tuple[float, float] = (0.1, 5.0)
    training_epochs: int = 500
    training_lr: float = 0.01
    warmup_iters: int = 5
    timing_iters: int = 20
    n_seeds: int = 3
    convergence_threshold: float = 1e-4

    batchable_models: list[str] = field(default_factory=lambda: [
        "M01_coulomb", "M02_beer_lambert", "M03_michaelis_menten",
        "M04_arrhenius", "M05_hookes_spring", "M06_logistic_growth",
        "M07_power_law", "M08_euler_ode", "M10_smooth_activation",
        "M11_recursive_filter", "M12_newton_sqrt",
        "M14_anomaly_scorer",
        # Excluded: M09 (recursive, needs list ops), M13 (closures/lambdas),
        #           M15 (list-based Horner's method)
    ])


DEFAULT = ExpHConfig()
