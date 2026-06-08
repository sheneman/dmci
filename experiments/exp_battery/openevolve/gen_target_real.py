############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# gen_target_real.py: Build the REAL Severson search target: per-cell discharge-capacity-vs-cycle resampled onto the common cycle...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Build the REAL Severson search target: per-cell discharge-capacity-vs-cycle resampled onto
the common cycle grid and normalized to SOH (Q/Q0), saved as results/target.pt in the SAME
contract as the synthetic gen_target.py (obs [N,T,1] float32, mechanism, truths) so the scorer
(oe_score.score_evolved), config (BCFG, N_SERIES=1, T_CYCLES, KSPLIT_*) and the OpenEvolve
harness run UNCHANGED -- the real-data path is a pure data swap.

Sources (auto-detected by extension; all reduce to {cell_id: capacity_vs_cycle 1-D float}):
  *.pkl   dict{cell_id: ndarray}     rg1990/knee-finder severson_capacity.pkl  (numpy-only,
          loaded via a RESTRICTED unpickler that permits ONLY numpy array reconstruction).
  *.csv   long CSV battery_id,cycle,QD   (Kaggle solitaryseeker summary CSV).
  *.mat   raw MATLAB v7.3 / HDF5 batch  (data.matr.io) -- summary 'QDischarge' via h5py.

Grid modes (--grid):
  lifefrac  (DEFAULT, recommended) -- resample each cell's life [first kept cycle .. EOL] onto T
            points, so the FULL flat-then-soft-rollover shape lands inside the window. Severson
            cells fade negligibly in the first ~100 cycles, so an absolute first-T window would be
            near-flat and structure-blind; life-fraction sampling is what makes smooth vs
            curvature-capable structures separable on the held-out tail.
  absolute  -- the cell's first T logged cycles (near-flat; provided for the contrast/ablation).

EOL: end-of-life cycle = first cycle whose SOH <= --eol-soh (default 0.80, i.e. Severson's 80%
of nominal); cells that never reach it use their last logged cycle as the life end.

This module is import-light: the report path (--report / --no-save) uses ONLY numpy (+optional
matplotlib) so it runs on the HPC login node (conda base) where torch will not load; torch is
imported lazily only when writing target.pt.
"""

from __future__ import annotations

import argparse
import io
import os
import pickle

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, "..", "results"))

# --- defaults mirror experiments/exp_battery/config.py (kept literal so this stays torch-free) --
T_CYCLES = 100
KSPLIT_LATE = 70
KSPLIT_EARLY = 45


# ----------------------------------------------------------------------------------------------
# Source loaders -> {cell_id: capacity_vs_cycle (1-D float, raw Ah)}
# ----------------------------------------------------------------------------------------------
class _NumpyOnlyUnpickler(pickle.Unpickler):
    """Unpickler that permits ONLY numpy array reconstruction globals -- refuses everything else,
    so a malicious payload cannot import os/subprocess/builtins to execute code. Verified against
    the rg1990 pkl, whose opcode stream contains exactly these three globals."""

    _ALLOWED = {
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
        ("numpy.core.multiarray", "scalar"),
    }

    def find_class(self, module, name):  # noqa: A003
        if (module, name) in self._ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked global {module}.{name} (numpy-only unpickler)")


def load_pkl(path: str) -> dict[str, np.ndarray]:
    with open(path, "rb") as fh:
        d = _NumpyOnlyUnpickler(io.BytesIO(fh.read())).load()
    return {str(k): np.asarray(v, dtype=np.float64).squeeze() for k, v in d.items()}


def load_npz(path: str) -> dict[str, np.ndarray]:
    """Load the safe .npz produced by convert_pkl.py (one named 1-D array per cell, no pickle)."""
    z = np.load(path, allow_pickle=False)
    return {k: np.asarray(z[k], dtype=np.float64).squeeze() for k in z.files}


def load_csv(path: str) -> dict[str, np.ndarray]:
    import pandas as pd

    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    bid, cyc, qd = cols["battery_id"], cols["cycle"], cols["qd"]
    out: dict[str, np.ndarray] = {}
    for b, g in df.groupby(bid):
        g = g.sort_values(cyc)
        out[str(b)] = np.column_stack([g[cyc].to_numpy(np.float64), g[qd].to_numpy(np.float64)])
    return out


def load_mat(paths: list[str]) -> dict[str, np.ndarray]:
    import h5py

    out: dict[str, np.ndarray] = {}
    for bi, path in enumerate(sorted(paths), start=1):
        with h5py.File(path, "r") as f:
            batch = f["batch"]
            n = batch["summary"].shape[0]
            for i in range(n):
                summ = f[batch["summary"][i, 0]]
                qd = np.hstack(summ["QDischarge"][0, :].tolist()).astype(np.float64)
                cyc = np.hstack(summ["cycle"][0, :].tolist()).astype(np.float64)
                out[f"b{bi}c{i}"] = np.column_stack([cyc, qd])
    return out


def load_source(src: str) -> dict[str, np.ndarray]:
    if os.path.isdir(src):
        mats = [os.path.join(src, f) for f in os.listdir(src) if f.endswith(".mat")]
        if not mats:
            raise SystemExit(f"no .mat files in {src}")
        return load_mat(mats)
    ext = os.path.splitext(src)[1].lower()
    if ext == ".pkl":
        return load_pkl(src)
    if ext == ".npz":
        return load_npz(src)
    if ext == ".csv":
        return load_csv(src)
    if ext == ".mat":
        return load_mat([src])
    raise SystemExit(f"unrecognized source extension: {ext}")


# ----------------------------------------------------------------------------------------------
# Per-cell cleanup -> SOH on the common grid
# ----------------------------------------------------------------------------------------------
def _eol_cycle(soh: np.ndarray, eol: float) -> int:
    """Index of the first cycle at/below the EOL SOH; len-1 if the cell never reaches it."""
    below = np.where(soh <= eol)[0]
    return int(below[0]) if below.size else int(soh.size - 1)


def _as_cycle_cap(arr: np.ndarray):
    """Coerce a cell's stored array to (cycle, capacity) 1-D pair.

    Accepts (N,2)=[cycle,cap] (rg1990 pkl/npz, Kaggle CSV, .mat), (2,N), or a bare 1-D capacity
    series (implicit consecutive cycle index)."""
    arr = np.squeeze(np.asarray(arr, dtype=np.float64))
    if arr.ndim == 2:
        if arr.shape[1] == 2:
            return arr[:, 0], arr[:, 1]
        if arr.shape[0] == 2:
            return arr[0], arr[1]
        return None, None
    if arr.ndim == 1:
        return np.arange(arr.size, dtype=np.float64), arr
    return None, None


def cell_to_grid(arr: np.ndarray, t: int, grid: str, eol: float,
                 drop_first: int = 0) -> tuple[np.ndarray, dict] | tuple[None, dict]:
    """One raw cell -> SOH on a length-`t` grid, plus per-cell diagnostics.

    Returns (None, info) when the cell is unusable (too short / non-finite / anomalous)."""
    cyc, cap = _as_cycle_cap(arr)
    if cyc is None:
        return None, {"reject": "bad_shape"}
    info = {"n_cycles": int(cap.size)}
    m = np.isfinite(cyc) & np.isfinite(cap) & (cap > 0)
    cyc, cap = cyc[m], cap[m]
    if cap.size < drop_first + 10:
        return None, {**info, "reject": "too_short"}
    order = np.argsort(cyc)
    cyc, cap = cyc[order], cap[order]
    if drop_first:                               # drop leading cycles (data-quality) if asked
        cyc, cap = cyc[drop_first:], cap[drop_first:]
    q0 = cap[0]
    if not np.isfinite(q0) or q0 <= 0:
        return None, {**info, "reject": "bad_q0"}
    soh = cap / q0                               # Q/Q0, starts ~1.0
    if soh.max() > 1.25 or soh.min() > 0.98:     # noisy cell, or no measurable fade at all
        return None, {**info, "reject": f"anomalous_soh[{soh.min():.3f},{soh.max():.3f}]"}
    eol_i = _eol_cycle(soh, eol)                 # index of life end (first SOH<=eol, else last)
    info.update({"q0": float(q0), "soh_end": float(soh[-1]), "life_cycles": int(cyc[-1]),
                 "eol_cycle": float(cyc[eol_i]), "reached_eol": bool(soh[eol_i] <= eol)})
    if grid == "absolute":                       # the cell's first t cycles (near-flat; contrast)
        end = cyc[0] + (t - 1)
        if cyc[-1] < end:
            return None, {**info, "reject": "shorter_than_T"}
        xi = cyc[0] + np.arange(t)
        return np.interp(xi, cyc, soh).astype(np.float64), info
    if eol_i < 5:                                # lifefrac: resample [cyc0 .. eol] onto t points
        return None, {**info, "reject": "eol_too_early"}
    xi = np.linspace(cyc[0], cyc[eol_i], t)      # captures the full flat->soft-rollover shape
    return np.interp(xi, cyc[: eol_i + 1], soh[: eol_i + 1]).astype(np.float64), info


def build_obs(curves: dict[str, np.ndarray], t: int, grid: str, eol: float):
    """All cells -> obs [N,t] (numpy), kept cell ids, and a rejection log."""
    kept_ids, rows, rejects = [], [], {}
    for cid, cap in curves.items():
        g, info = cell_to_grid(cap, t, grid, eol)
        if g is None:
            rejects[cid] = info.get("reject", "?")
            continue
        kept_ids.append(cid)
        rows.append(g)
    obs = np.stack(rows) if rows else np.zeros((0, t))
    return obs, kept_ids, rejects


# ----------------------------------------------------------------------------------------------
# Reporting (numpy-only; decide the grid empirically before the expensive sweep)
# ----------------------------------------------------------------------------------------------
def _sparkline(y: np.ndarray) -> str:
    blocks = "▁▂▃▄▅▆▇█"
    lo, hi = float(np.min(y)), float(np.max(y))
    if hi - lo < 1e-9:
        return blocks[0] * len(y)
    idx = np.clip(((y - lo) / (hi - lo) * (len(blocks) - 1)).round().astype(int), 0, len(blocks) - 1)
    return "".join(blocks[i] for i in idx)


def report(obs: np.ndarray, kept_ids: list[str], rejects: dict, t: int,
           ksplit_late: int, ksplit_early: int) -> None:
    n = obs.shape[0]
    print(f"\n=== REAL Severson target report (grid points T={t}) ===")
    print(f"kept cells: {n}   rejected: {len(rejects)}")
    if rejects:
        import collections
        rc = collections.Counter(rejects.values())
        print("  reject reasons:", dict(rc))
    if n == 0:
        return
    print(f"SOH range over all cells: [{obs.min():.4f}, {obs.max():.4f}]")
    print(f"SOH at start (grid 0):  mean={obs[:,0].mean():.4f}")
    print(f"SOH at end   (grid {t-1}): mean={obs[:,-1].mean():.4f}  min={obs[:,-1].min():.4f}")
    mean_curve = obs.mean(0)
    print(f"\nmean curve sparkline (grid 0..{t-1}):\n  {_sparkline(mean_curve)}")
    # rollover diagnostic: discrete 2nd difference of the mean curve -> where does fade accelerate?
    d2 = np.diff(mean_curve, 2)
    lo = int(0.2 * t)  # skip the early capacity-rise artifact real Severson cells show (~first 100 cyc)
    knee_grid = int(np.argmin(d2[lo:])) + lo + 1  # most-negative curvature = steepest acceleration
    after_early = knee_grid > ksplit_early
    pos = "AFTER" if after_early else "BEFORE"
    sees = ("flat-only fit, rollover held out [GOOD discriminator]" if after_early
            else "the rollover already inside the fit window")
    print(f"mean-curve steepest-acceleration grid index ~ {knee_grid}/{t}  "
          f"(ksplit_early={ksplit_early}, ksplit_late={ksplit_late})")
    print(f"  -> {pos} the EARLY split (early split sees {sees})")
    # decile table of the mean curve + drop fraction in held-out tails
    deciles = mean_curve[np.linspace(0, t - 1, 11).astype(int)]
    print("\nmean SOH at deciles 0..100%:", " ".join(f"{v:.3f}" for v in deciles))
    for ks, nm in [(ksplit_early, "EARLY"), (ksplit_late, "LATE")]:
        fit_drop = mean_curve[0] - mean_curve[ks - 1]
        tail_drop = mean_curve[ks - 1] - mean_curve[-1]
        print(f"  {nm} split @{ks}: fade in fit window={fit_drop:.4f}, "
              f"fade in held-out tail={tail_drop:.4f}  (tail/fit ratio={tail_drop/max(fit_drop,1e-6):.2f})")
    print("\nsample cell sparklines:")
    for cid in kept_ids[: min(8, n)]:
        row = obs[kept_ids.index(cid)]
        print(f"  {cid:8s} [{row[0]:.3f}->{row[-1]:.3f}]  {_sparkline(row)}")


# ----------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Build the real Severson SOH target.pt")
    ap.add_argument("--src", required=True, help="path to .pkl / .csv / .mat, or a dir of .mat")
    ap.add_argument("--grid", choices=["lifefrac", "absolute"], default="lifefrac")
    ap.add_argument("--t", type=int, default=T_CYCLES)
    ap.add_argument("--eol-soh", type=float, default=0.80)
    ap.add_argument("--ksplit-late", type=int, default=KSPLIT_LATE)
    ap.add_argument("--ksplit-early", type=int, default=KSPLIT_EARLY)
    ap.add_argument("--limit", type=int, default=0, help="cap #cells (0=all)")
    ap.add_argument("--out", default=os.path.join(RESULTS, "target.pt"))
    ap.add_argument("--no-save", action="store_true", help="report only; skip torch.save")
    args = ap.parse_args()

    curves = load_source(args.src)
    if args.limit:
        curves = dict(list(curves.items())[: args.limit])
    print(f"[gen_target_real] loaded {len(curves)} raw cells from {args.src}")
    obs, kept_ids, rejects = build_obs(curves, args.t, args.grid, args.eol_soh)
    report(obs, kept_ids, rejects, args.t, args.ksplit_late, args.ksplit_early)

    if args.no_save:
        np.savez(os.path.splitext(args.out)[0] + "_real.npz", obs=obs.astype(np.float32),
                 cell_ids=np.array(kept_ids))
        print(f"[gen_target_real] (no-save) wrote {os.path.splitext(args.out)[0]+'_real.npz'}")
        return
    import torch  # lazy: only needed to write target.pt (fails on the login node)

    obs_t = torch.tensor(obs[:, :, None], dtype=torch.float32)  # [N, T, 1]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"obs": obs_t, "mechanism": "real_severson", "truths": None,
                "cell_ids": kept_ids, "grid": args.grid, "eol_soh": args.eol_soh}, args.out)
    print(f"[gen_target_real] wrote {args.out}  obs={tuple(obs_t.shape)}  grid={args.grid}")


if __name__ == "__main__":
    main()
