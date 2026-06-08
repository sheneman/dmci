############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# build_data.py: Orchestrate the exp_lim_enso data pipeline: fetch -> preprocess (D_max once) -> write. Runs LOCALLY on the Mac...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Orchestrate the exp_lim_enso data pipeline: fetch -> preprocess (D_max once) -> write.

Runs LOCALLY on the Mac (python3 = miniconda; xarray + netCDF4 installed). Produces the
processed artifacts the experiment consumes, plus a self-describing metadata.json.

Usage (run from the repo root):

    python3 -m experiments.exp_lim_enso.data.build_data
    python3 -m experiments.exp_lim_enso.data.build_data --period 1950-01 2024-12 \
        --domain-lat -30 30 --domain-lon 30 290 --D-max 20
    python3 -m experiments.exp_lim_enso.data.build_data --force   # re-fetch + rebuild

OUTPUTS (under experiments/exp_lim_enso/data/processed/):
    pcs.npy     [T, D_max] float32   <- THE experiment input (bound via as_matrix(pcs[:,:D]))
    eofs.npy    [D_max, S] float32      spatial EOFs (weighted space)
    pc_std.npy  [D_max]    float32      std of each raw PC (un-normalise factor)
    lat.npy     [S]        float32      latitude  of each ocean column
    lon.npy     [S]        float32      longitude of each ocean column (0..360 E)
    mask.npy    [n_lat,n_lon] bool      ocean mask over the subset grid
    metadata.json                       full provenance + the [T,D] binding contract

Only experiments/exp_lim_enso/data/processed/ is rsync'd to HPC; raw/ stays local.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
RAW_DIR = HERE / "raw"
PROCESSED_DIR = HERE / "processed"
RAW_PATH = RAW_DIR / "sst.mnmean.nc"


# ----------------------------------------------------------------------------
def _check_dependencies() -> None:
    """Fail EARLY and ACTIONABLY if the preprocessing deps are missing, rather than
    crashing deep inside open_dataset."""
    missing = []
    try:
        import xarray  # noqa: F401
    except Exception:
        missing.append("xarray")
    try:
        import netCDF4  # noqa: F401
    except Exception:
        # h5netcdf is an acceptable alternative engine
        try:
            import h5netcdf  # noqa: F401
        except Exception:
            missing.append("netCDF4 (or h5netcdf)")
    if missing:
        raise SystemExit(
            "ERROR: missing required packages for preprocessing: "
            + ", ".join(missing) + "\n"
            "  Install them in the LOCAL miniconda env, e.g.:\n"
            "    python3 -m pip install xarray netCDF4\n"
            "  (or: conda install -c conda-forge xarray netcdf4)\n"
            "  The download step (fetch_ersst.py) is pure-stdlib and does not need these; "
            "only preprocessing does. This pipeline is meant to run locally on the Mac."
        )


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(HERE),
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def _file_sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ----------------------------------------------------------------------------
def build(period, domain_lat, domain_lon, d_max: int, force: bool) -> dict:
    _check_dependencies()
    # import here so the dependency check runs first with a clean message
    from .fetch_ersst import fetch, SOURCE_URL, EXPECTED_BYTES
    from . import preprocess_lim as pp

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # --- 1. fetch (resumable; skips if complete unless --force) ---
    print("=== [1/3] fetch ===", flush=True)
    raw_path = fetch(force=force, url=SOURCE_URL, dest=RAW_PATH)
    raw_sha = _file_sha256(raw_path)

    # --- 2. preprocess at D_max ONCE (nested basis; downstream slices pcs[:, :D]) ---
    print("\n=== [2/3] preprocess (float64) ===", flush=True)
    res = pp.preprocess(raw_path, domain_lat=tuple(domain_lat),
                        domain_lon=tuple(domain_lon), period=tuple(period),
                        d_max=int(d_max))
    T = int(res.pcs.shape[0])
    S = int(res.lat.shape[0])
    print(f"  T(time steps after running-mean trim) = {T}", flush=True)
    print(f"  S(ocean gridpoints)                   = {S}", flush=True)
    print(f"  grid = {res.mask.shape[0]} lat x {res.mask.shape[1]} lon "
          f"(ocean fraction {res.mask.mean():.2%})", flush=True)

    # --- 3. write float32 / bool artifacts + metadata ---
    print("\n=== [3/3] write processed artifacts ===", flush=True)
    pcs32 = res.pcs.astype(np.float32)
    eofs32 = res.eofs.astype(np.float32)
    pc_std32 = res.pc_std.astype(np.float32)
    lat32 = res.lat.astype(np.float32)
    lon32 = res.lon.astype(np.float32)
    mask_b = res.mask.astype(bool)

    np.save(PROCESSED_DIR / "pcs.npy", pcs32)
    np.save(PROCESSED_DIR / "eofs.npy", eofs32)
    np.save(PROCESSED_DIR / "pc_std.npy", pc_std32)
    np.save(PROCESSED_DIR / "lat.npy", lat32)
    np.save(PROCESSED_DIR / "lon.npy", lon32)
    np.save(PROCESSED_DIR / "mask.npy", mask_b)

    var_cumsum = np.cumsum(res.variance_explained).astype(float).tolist()
    dates = res.dates.astype("datetime64[M]")
    autocorr = pp.leading_pc_autocorr(res.pcs, lags=(1, 6, 12))

    metadata = {
        "source_url": SOURCE_URL,
        "source_expected_bytes": EXPECTED_BYTES,
        "source_sha256": raw_sha,
        "domain_lat": list(domain_lat),
        "domain_lon_deg_east": list(domain_lon),
        "period": list(period),
        "T": T,
        "S_ocean_gridpoints": S,
        "grid_shape_lat_lon": [int(res.mask.shape[0]), int(res.mask.shape[1])],
        "D_max": int(d_max),
        "variance_explained_per_pc": res.variance_explained.astype(float).tolist(),
        "variance_explained_cumsum": var_cumsum,
        "preprocessing_flags": res.flags,
        "dates_first": str(dates[0]),
        "dates_last": str(dates[-1]),
        "leading_pc_autocorr": {str(k): v for k, v in autocorr.items()},
        "git_commit": _git_commit(),
        "built_utc": datetime.now(timezone.utc).isoformat(),
        "built_by": "experiments/exp_lim_enso/data/build_data.py",
        "binding_contract": (
            "pcs.npy has shape [T, D_max], row = time step (month), column = PC index. "
            "For a fit at dimension D: slice pcs[:, :D] (nested basis) and bind it into the "
            "DMCI program with as_matrix(pcs[:, :D]); inside the Scheme, obs is that matrix and "
            "(ref obs k) gathers ROW k (the D-vector of PCs at month k). k is the loop counter "
            "(data-independent), as required. PCs are unit-variance normalised; multiply column "
            "j by pc_std[j] to recover the raw expansion coefficient."
        ),
        "artifacts": {
            "pcs.npy": {"shape": list(pcs32.shape), "dtype": "float32",
                        "desc": "unit-variance PCs [T, D_max] (THE input)"},
            "eofs.npy": {"shape": list(eofs32.shape), "dtype": "float32",
                         "desc": "spatial EOFs [D_max, S] in sqrt(cos lat)-weighted space"},
            "pc_std.npy": {"shape": list(pc_std32.shape), "dtype": "float32",
                           "desc": "std of each raw PC [D_max] (un-normalise factor)"},
            "lat.npy": {"shape": list(lat32.shape), "dtype": "float32",
                        "desc": "latitude of each ocean column [S]"},
            "lon.npy": {"shape": list(lon32.shape), "dtype": "float32",
                        "desc": "longitude (0..360 E) of each ocean column [S]"},
            "mask.npy": {"shape": list(mask_b.shape), "dtype": "bool",
                         "desc": "ocean mask over the subset grid [n_lat, n_lon]"},
        },
    }
    meta_path = PROCESSED_DIR / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))

    # --- summary ---
    print(f"  wrote pcs.npy     {pcs32.shape} float32")
    print(f"  wrote eofs.npy    {eofs32.shape} float32")
    print(f"  wrote pc_std.npy  {pc_std32.shape} float32")
    print(f"  wrote lat/lon.npy {lat32.shape} float32")
    print(f"  wrote mask.npy    {mask_b.shape} bool")
    print(f"  wrote metadata.json")
    _print_summary(T, res, var_cumsum, autocorr, dates, d_max)
    return metadata


def _print_summary(T, res, var_cumsum, autocorr, dates, d_max):
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  period   : {dates[0]} .. {dates[-1]}  (T = {T} months)")
    print(f"  D_max    : {d_max}   S(ocean) = {res.lat.shape[0]}")
    print("  cumulative variance explained:")
    for D in (6, 10, 15, 20):
        if D <= len(var_cumsum):
            print(f"      D={D:2d} -> {100*var_cumsum[D-1]:5.1f}%")
    print("  leading-PC (EOF1) autocorrelation — ENSO-like if positive at 6-12 mo:")
    for L in (1, 6, 12):
        if L in autocorr:
            print(f"      lag {L:2d} mo -> {autocorr[L]:+.3f}")
    enso_like = autocorr.get(6, 0.0) > 0.3 and autocorr.get(12, -1.0) > -0.2
    print(f"  EOF1 looks ENSO-like: {enso_like}  "
          f"(persistence at 6mo {'OK' if autocorr.get(6,0)>0.3 else 'weak'})")
    print("=" * 60)
    print("Next: rsync experiments/exp_lim_enso/data/processed/ to HPC; the fit harness")
    print("binds pcs[:, :D] via as_matrix and reads rows with (ref obs k).")


# ----------------------------------------------------------------------------
def _parse_period(s):
    # accept either two tokens or a single "YYYY-MM:YYYY-MM"
    if len(s) == 1 and ":" in s[0]:
        return tuple(s[0].split(":", 1))
    if len(s) == 2:
        return (s[0], s[1])
    raise argparse.ArgumentTypeError("period needs START END (e.g. 1950-01 2024-12)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Build exp_lim_enso PC inputs from ERSSTv5 (fetch+preprocess locally).")
    ap.add_argument("--period", nargs="+", default=["1950-01", "2024-12"],
                    metavar="YYYY-MM",
                    help="analysis period START END (default 1950-01 2024-12)")
    ap.add_argument("--domain-lat", nargs=2, type=float, default=[-30.0, 30.0],
                    metavar=("LO", "HI"), help="latitude domain (default -30 30)")
    ap.add_argument("--domain-lon", nargs=2, type=float, default=[30.0, 290.0],
                    metavar=("LO", "HI"),
                    help="longitude domain in degrees EAST 0..360 (default 30 290)")
    ap.add_argument("--D-max", type=int, default=20, help="number of PCs to retain (default 20)")
    ap.add_argument("--force", action="store_true",
                    help="re-fetch the raw file and rebuild from scratch")
    args = ap.parse_args(argv)

    period = _parse_period(args.period)
    try:
        build(period=period, domain_lat=args.domain_lat, domain_lon=args.domain_lon,
              d_max=args.D_max, force=args.force)
    except SystemExit:
        raise
    except FileNotFoundError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        return 1
    except Exception as e:  # surface a clean message, not a deep traceback
        import traceback
        traceback.print_exc()
        print(f"\nBUILD FAILED: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
