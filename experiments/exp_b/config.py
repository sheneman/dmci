############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# config.py: Experiment B configuration.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Experiment B configuration."""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class ExpBConfig:
    max_epochs: int = 3000
    convergence_threshold: float = 1e-3
    lr: float = 0.05
    n_seeds: int = 5
    n_data_points: int = 20
    x_range: tuple[float, float] = (0.1, 5.0)

    methods: list[str] = field(default_factory=lambda: [
        "dmci",
        "direct_compiled",
        "handcoded_pytorch",
        "pure_mlp",
    ])


DEFAULT = ExpBConfig()
