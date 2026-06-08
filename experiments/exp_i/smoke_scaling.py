############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# smoke_scaling.py: High-d feasibility probe for the Exp I scaling sweep (run on HPC before the array). Times the compile, one...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""High-d feasibility probe for the Exp I scaling sweep (run on HPC before the array).

Times the compile, one batched DMCI forward+backward (an Adam epoch), and extrapolates
the cost of the non-budgeted Adam fit (max_epochs) at the largest parameter counts, so we
know d=96/126 will fit the SLURM wall-clock before launching 9 tasks.

    python -m experiments.exp_i.smoke_scaling
"""
from __future__ import annotations
import sys, time
sys.setrecursionlimit(5000)
import torch

from experiments.exp_i.config import ExpIConfig
from experiments.exp_i.models import build_static_model
from experiments.exp_i.harness import (_compile, _make_params, generate_data,
                                        _predict_batched)


def probe(n_pft: int, cfg: ExpIConfig):
    m = build_static_model(n_pft)
    t0 = time.perf_counter(); g = _compile(m, "dmci"); t_compile = time.perf_counter() - t0
    params = _make_params(m, 0)
    xs, ys = generate_data(m, cfg, 0)
    db = {d: torch.tensor([x[d] for x in xs], dtype=torch.float32) for d in m.input_names}
    yb = torch.stack([y if torch.is_tensor(y) else torch.tensor(float(y)) for y in ys])
    opt = torch.optim.Adam(list(params.values()), lr=cfg.lr)
    n = 3
    t0 = time.perf_counter()
    loss = None
    for _ in range(n):
        opt.zero_grad()
        pr = _predict_batched(g, db, params)
        loss = ((pr - yb) ** 2).sum()
        loss.backward(); opt.step()
    t_epoch = (time.perf_counter() - t0) / n
    print(f"n_pft={n_pft:2d} d={6*n_pft:3d}  compile={t_compile:6.1f}s  "
          f"per_epoch={t_epoch:6.2f}s  adam_{cfg.max_epochs}~{t_epoch*cfg.max_epochs/60:6.1f}min  "
          f"loss0={loss.item():.3e}", flush=True)


def main():
    cfg = ExpIConfig()
    for p in (16, 21):     # the untested large counts (d=96, 126)
        probe(p, cfg)


if __name__ == "__main__":
    main()
