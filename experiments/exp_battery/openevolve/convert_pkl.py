############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# convert_pkl.py: One-shot, USER-RUN converter + inspector for the vetted Severson capacity pkl. Run this yourself with the `!`...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""One-shot, USER-RUN converter + inspector for the vetted Severson capacity pkl.

Run this yourself with the `!` prefix so it executes under your authority (the agent's auto
classifier blocks pickle deserialization of an agent-downloaded file; running it as your own
command is the sanctioned path). It:
  1. deserializes severson_capacity.pkl with a numpy-ONLY restricted unpickler (find_class
     permits only numpy array reconstruction -> a malicious payload cannot import os/eval/etc.),
  2. re-saves it as a SAFE .npz (one named 1-D array per cell, loadable with allow_pickle=False),
  3. prints data-quality diagnostics + the grid-decision report for BOTH grid modes, so the agent
     can verify the curves and choose lifefrac vs absolute without another round trip.

    ! python3 -m experiments.exp_battery.openevolve.convert_pkl
"""

from __future__ import annotations

import collections
import os
import sys

import numpy as np

from experiments.exp_battery.openevolve.gen_target_real import (
    load_pkl, build_obs, report, KSPLIT_EARLY, KSPLIT_LATE, T_CYCLES,
)

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.abspath(os.path.join(HERE, "..", "data", "raw"))
PKL = os.path.join(RAW, "severson_capacity.pkl")
NPZ = os.path.join(RAW, "severson_capacity.npz")


def main():
    print(f"python {sys.version.split()[0]}  numpy {np.__version__}")
    try:
        import torch  # noqa: F401
        print(f"torch {torch.__version__} AVAILABLE (can build target.pt here)")
    except Exception as e:  # noqa: BLE001
        print(f"torch NOT available here ({type(e).__name__}) -- build target.pt elsewhere")

    curves = load_pkl(PKL)                       # restricted numpy-only unpickler
    np.savez(NPZ, **curves)                      # safe re-encode, no pickle to read back
    print(f"\n[convert] {len(curves)} cells  pkl -> {NPZ}")

    batches = collections.Counter(k[:2] for k in curves)
    print("batch prefixes:", dict(batches))
    lens = np.array([np.asarray(curves[k]).squeeze().size for k in curves])
    q0 = np.array([np.asarray(curves[k]).squeeze()[0] for k in curves])
    qe = np.array([np.asarray(curves[k]).squeeze()[-1] for k in curves])
    print(f"life(cycles): min={lens.min()} p10={np.percentile(lens,10):.0f} "
          f"median={np.median(lens):.0f} p90={np.percentile(lens,90):.0f} max={lens.max()}")
    print(f"Q0(Ah): min={q0.min():.4f} median={np.median(q0):.4f} max={q0.max():.4f}")
    print(f"Q_end(Ah): min={qe.min():.4f} median={np.median(qe):.4f} max={qe.max():.4f}")

    for grid in ("lifefrac", "absolute"):
        obs, kept, rej = build_obs(curves, T_CYCLES, grid, eol=0.80)
        print(f"\n########## GRID = {grid} ##########")
        report(obs, kept, rej, T_CYCLES, KSPLIT_LATE, KSPLIT_EARLY)


if __name__ == "__main__":
    main()
