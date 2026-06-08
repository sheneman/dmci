############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# config.py: Experiment D configuration.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Experiment D configuration."""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class ExpDConfig:
    # GP parameters
    pop_size: int = 50
    n_generations: int = 20
    tournament_size: int = 3
    crossover_rate: float = 0.8
    mutation_rate: float = 0.2
    max_tree_depth: int = 5
    min_tree_depth: int = 2

    # Inner constant-fitting loop
    inner_epochs: int = 20
    inner_lr: float = 0.05

    # Data generation
    n_data_points: int = 20
    x_range: tuple[float, float] = (0.5, 3.0)

    # Target function: a*sin(b*x) + c*x^2
    target_a: float = 2.0
    target_b: float = 3.0
    target_c: float = 0.5

    # Runs
    n_seeds: int = 5
    methods: list[str] = field(default_factory=lambda: [
        "gp_direct",
        "gp_dmci",
    ])


DEFAULT = ExpDConfig()
