############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# synth.py: Mechanism-labeled synthetic capacity-fade data. For each structure we generate `n_cells` capacity-vs-cycle...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Mechanism-labeled synthetic capacity-fade data.

For each structure we generate `n_cells` capacity-vs-cycle curves from its KNOWN ground-truth
parameters (curves.TRUE_PARAMS), with per-cell multiplicative parameter jitter and additive
coulometry-grade Gaussian noise. The labels (which mechanism generated each cell) are exactly
what lets the pilot grade STRUCTURE recovery -- the validation no real dataset can provide and
the reason a local synthetic de-risk is the right first gate (recon recommendation).

Determinism: a single seeded numpy Generator (no global Date/random), so the pilot is
reproducible and resumable.
"""

from __future__ import annotations

import numpy as np
import torch

from .config import T_CYCLES
from .curves import predict_curve, TRUE_PARAMS, JITTER

NOISE = 2e-3   # capacity SD; coulometry-grade (~0.2% of nominal)


def _jittered(name: str, rng: np.random.Generator) -> dict:
    out = {}
    for k, v in TRUE_PARAMS[name].items():
        j = JITTER.get(k, 0.0)
        out[k] = v * (1.0 + j * rng.standard_normal()) if j else v
    return out


def make_cells(name: str, n_cells: int, rng: np.random.Generator,
               t_cycles: int = T_CYCLES) -> tuple[torch.Tensor, list[dict]]:
    """Return obs `[n_cells, T, 1]` float32 and the per-cell ground-truth params."""
    k = np.arange(t_cycles)
    cells, truths = [], []
    for _ in range(n_cells):
        p = _jittered(name, rng)
        q = predict_curve(name, p, k)
        q = q + NOISE * rng.standard_normal(t_cycles)
        cells.append(q)
        truths.append(p)
    obs = torch.tensor(np.stack(cells)[:, :, None], dtype=torch.float32)  # [N, T, 1]
    return obs, truths


def make_dataset(n_cells: int = 8, seed: int = 0,
                 t_cycles: int = T_CYCLES) -> dict[str, tuple[torch.Tensor, list[dict]]]:
    """Mechanism-labeled dataset: {structure_name: (obs[N,T,1], truths)} for all structures."""
    rng = np.random.default_rng(seed)
    return {name: make_cells(name, n_cells, rng, t_cycles) for name in TRUE_PARAMS}
