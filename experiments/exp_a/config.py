############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# config.py: Experiment A configuration: the program suite, seeds, optimizer, and baseline settings
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


@dataclass(frozen=True)
class ExpAConfig:
    n_seeds: int = 10
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    max_epochs: int = 3000
    convergence_threshold: float = 1e-3
    param_error_threshold: float = 0.01

    n_data_points: int = 8
    x_range: tuple[float, float] = (0.5, 3.0)

    lr: float = 0.05
    optimizer: str = "Adam"

    fd_epsilon: float = 1e-4
    fd_lr: float = 0.001

    es_sigma0: float = 0.5
    es_max_fevals: int = 10000
    es_pop_size: int = 20

    output_dir: str = "experiments/exp_a/results"

    methods: tuple[str, ...] = (
        "direct",
        "compiled_interp",
        "handcoded_interp",
        "finite_diff",
        "evolution_strategy",
    )


DEFAULT = ExpAConfig()
