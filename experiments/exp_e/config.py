############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# config.py: Experiment E.1 configuration: Operator Recovery via Soft-Dispatch.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Experiment E.1 configuration: Operator Recovery via Soft-Dispatch."""

from __future__ import annotations
from dataclasses import dataclass, field


OPERATORS = ["+", "-", "*", "/", "sin", "cos", "exp", "log"]
OP_LABELS = ["add", "sub", "mul", "div", "sin", "cos", "exp", "log"]
N_OPS = len(OPERATORS)

# 12 target tasks: (op1_idx, op2_idx, a_star, description)
# op1 is the outer operator, op2 is the inner operator.
# f*(x) = a* · op1(x, op2(x))
#   Binary ops (idx 0-3): op(x, inner)
#   Unary ops  (idx 4-7): op(inner)
# Inner ops: 0=2x, 1=-x, 2=x², 3=~1/x, 4=sin, 5=cos, 6=exp, 7=~log|x|
# Outer ops: 0=x+inner, 1=x-inner, 2=x*inner, 3=~x/inner, 4=sin, 5=cos, 6=exp, 7=~log|inner|
TASKS = [
    # --- a* = 0.5 ---
    (2, 0, 0.5, "0.5·x·2x = x²"),
    (2, 4, 0.5, "0.5·x·sin(x)"),
    (2, 6, 0.5, "0.5·x·exp(x)"),
    (0, 2, 0.5, "0.5·(x+x²)"),
    # --- a* = 1.0 ---
    (4, 2, 1.0, "sin(x²)"),
    (0, 5, 1.0, "x+cos(x)"),
    (1, 4, 1.0, "x−sin(x)"),
    (5, 2, 1.0, "cos(x²)"),
    # --- a* = 2.0 ---
    (0, 2, 2.0, "2·(x+x²)"),
    (5, 0, 2.0, "2·cos(2x)"),
    (4, 0, 2.0, "2·sin(2x)"),
    (2, 5, 2.0, "2·x·cos(x)"),
]

assert len(TASKS) == 12


@dataclass
class ExpE1Config:
    n_restarts: int = 20
    n_epochs: int = 3000
    lr: float = 0.05
    tau_start: float = 1.0
    tau_end: float = 0.1
    n_data: int = 64
    x_lo: float = 0.2
    x_hi: float = 3.0
    noise_std: float = 0.01
    deriv_weight: float = 1.0
    seed_offset: int = 0
    # Evolutionary algorithm
    ea_pop_size: int = 32
    ea_generations: int = 100
    ea_tournament_k: int = 3
    ea_mutation_op_prob: float = 0.3
    ea_mutation_a_std: float = 0.3
    ea_crossover_rate: float = 0.7


DEFAULT = ExpE1Config()
