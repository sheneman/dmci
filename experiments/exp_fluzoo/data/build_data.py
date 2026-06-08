############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# build_data.py: Orchestrate fetch -> preprocess -> write processed arrays + metadata provenance. python3 -m...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Orchestrate fetch -> preprocess -> write processed arrays + metadata provenance.

    python3 -m experiments.exp_fluzoo.data.build_data [--force]

Writes data/processed/{wili.npy [T,R], epiweeks.npy [T], seasons.npy [T],
mask_{train,val,test,pandemic}.npy [T]} (gitignored) and metadata.json (tracked).
Run locally; rsync data/processed to HPC before any sweep.
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

import numpy as np

from ..config import DEFAULT
from .fetch_ilinet import EPIDATA_URL, epiweek_window, fetch_all, source_sha256
from .preprocess_flu import preprocess

HERE = Path(__file__).resolve().parent
PROCESSED = HERE / "processed"


def load_processed(processed_dir: Path = PROCESSED) -> dict:
    """Load the binding-ready arrays for the runner/forecaster."""
    p = processed_dir
    out = {
        "wili": np.load(p / "wili.npy"),
        "epiweeks": np.load(p / "epiweeks.npy"),
        "seasons": np.load(p / "seasons.npy"),
        "masks": {n: np.load(p / f"mask_{n}.npy")
                  for n in ("train", "val", "test", "pandemic")},
        "metadata": json.loads((p / "metadata.json").read_text()),
    }
    return out


def build(force: bool = False, processed_dir: Path = PROCESSED) -> None:
    cfg = DEFAULT
    seasons = cfg.all_seasons
    by_region = fetch_all(cfg.regions, min(seasons), max(seasons), force=force)
    fd = preprocess(by_region, cfg.regions, cfg.train_seasons, cfg.val_seasons,
                    cfg.test_seasons, cfg.pandemic_seasons)

    processed_dir.mkdir(parents=True, exist_ok=True)
    np.save(processed_dir / "wili.npy", fd.wili)
    np.save(processed_dir / "epiweeks.npy", fd.epiweeks)
    np.save(processed_dir / "seasons.npy", fd.seasons)
    for name, m in fd.masks.items():
        np.save(processed_dir / f"mask_{name}.npy", m)

    ew_start, ew_end = epiweek_window(min(seasons), max(seasons))
    metadata = {
        "experiment": "exp_fluzoo",
        "source": "CDC ILINet via Delphi Epidata fluview endpoint",
        "source_url": EPIDATA_URL,
        "built_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "source_sha256": source_sha256(by_region),
        "epiweek_window": [int(ew_start), int(ew_end)],
        "regions": list(fd.regions),
        "units": "weighted %ILI as a PROPORTION (wili/100), matches model rho*I",
        "shape_wili": list(fd.wili.shape),
        "T": int(fd.T),
        "release_dates": fd.release_dates,
        "splits": {
            "train_seasons": list(cfg.train_seasons),
            "val_seasons": list(cfg.val_seasons),
            "test_seasons": list(cfg.test_seasons),
            "pandemic_seasons": list(cfg.pandemic_seasons),
            "weeks": {n: int(m.sum()) for n, m in fd.masks.items()},
        },
        "binding_contract": ("obs = wili[mask] is a [n_weeks, 11] float32 matrix bound via "
                             "as_matrix; (ref obs k) gathers week k as an 11-vector "
                             "(national, HHS-1..10 in column order)."),
        "revision_note": ("Latest-issue (finalised) wILI; revisions exist. release_dates "
                          "recorded per region for a pinned as_of snapshot if needed."),
        "wili_range": [float(np.nanmin(fd.wili)), float(np.nanmax(fd.wili))],
    }
    (processed_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"[build] wrote {processed_dir}/  T={fd.T}  shape={fd.wili.shape}  "
          f"range=[{metadata['wili_range'][0]:.4f}, {metadata['wili_range'][1]:.4f}]")
    print(f"[build] sha256={metadata['source_sha256'][:16]}...")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    build(force=args.force)
