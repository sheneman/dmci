############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# preprocess_lim.py: Canonical LIM/ENSO preprocessing of ERSSTv5 SST -> EOF principal components. The pipeline produces the PC time...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Canonical LIM/ENSO preprocessing of ERSSTv5 SST -> EOF principal components.

The pipeline produces the PC time series ``pcs[T, D_max]`` that is the INPUT to the
exp_lim_enso experiment: it is bound into the DMCI Kalman/LIM program via
``as_matrix(pcs[:, :D])`` and the loop reads observation row ``k`` with ``(ref obs k)``.

ALL computation here is FLOAT64 (climatology, detrend, area-weight, SVD); we cast to
float32 ONLY at the final write step (build_data.py). This matters: EOF/SVD on a tropical
SST field is conditioning-sensitive and float32 climatology removal leaks the annual cycle.

The standard LIM-ENSO recipe (Penland & Sardeshmukh 1995; Newman et al.) implemented:
  1. subset to the tropical Indo-Pacific domain and the analysis period;
  2. ocean mask: drop grid columns that are all-NaN (land/ice) over the period;
  3. remove the per-calendar-month climatology  -> monthly anomalies;
  4. per-gridpoint LINEAR DETREND of the anomalies  (AFTER climatology, BEFORE EOF —
     otherwise EOF1 captures the secular warming trend, not ENSO);
  5. 3-month running mean (trim the 1-month edges that the centred window can't fill);
  6. sqrt(cos lat) area-weighting of each column (equal-area in the EOF inner product);
  7. economy SVD of the [T, S] weighted anomaly matrix  -> EOFs (spatial) / PCs (temporal);
  8. retain D_max PCs; unit-variance normalise each PC (store pc_std for un-normalising).

EOF / SIGN / ORTHOGONALITY CONVENTIONS (documented so downstream code is unambiguous):
  - We compute  A_w = U S V^T  (economy SVD of the area-WEIGHTED, time-by-space anomaly
    matrix A_w, shape [T, S]). U is [T, k], S is [k], V is [S, k].
  - PCs are the temporal patterns  PC_j(t) = U[:, j] * S[j]  (a.k.a. the "expansion
    coefficients"); EOFs are the spatial patterns  EOF_j = V[:, j]  in the WEIGHTED space.
    Both PC columns are mutually orthogonal; EOF columns are orthonORMAL (V^T V = I).
  - PCs are then unit-variance normalised: PC_j <- PC_j / std(PC_j); we store
    pc_std[j] = std(PC_j) so the raw expansion coefficient is recoverable as
    pcs_norm * pc_std. (The Kalman/LIM fit only needs the *normalised* unit-variance PCs.)
  - SIGN: SVD sign is arbitrary. We FIX it deterministically so re-runs and the
    ENSO interpretation are stable: flip each (EOF_j, PC_j) pair so the spatial EOF has
    a POSITIVE area-weighted mean over the eastern-equatorial "Nino-ish" box
    (lat[-5,5], lon[190,270] = 170W-90W). This makes EOF1 a warm (El Nino) positive
    phase by convention; falls back to "largest-|loading| element positive" if the box
    is empty for the chosen domain.
  - VARIANCE EXPLAINED: frac_j = S[j]^2 / sum_i(S_i^2) over the FULL economy spectrum.

This module has NO torch / neural_compiler dependency and runs entirely on the Mac.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


# ----------------------------------------------------------------------------
# Defaults (the authoritative LIM-ENSO config).
DEFAULT_DOMAIN_LAT = (-30.0, 30.0)
DEFAULT_DOMAIN_LON = (30.0, 290.0)      # degrees East (0..360 convention)
DEFAULT_PERIOD = ("1950-01", "2024-12")
DEFAULT_D_MAX = 20
RUNNING_MEAN_WINDOW = 3                  # months (centred)
# Eastern-equatorial box used ONLY to fix the EOF sign (Nino-3-ish region).
SIGN_BOX_LAT = (-5.0, 5.0)
SIGN_BOX_LON = (190.0, 270.0)            # 170W .. 90W in 0..360 convention


@dataclass
class PreprocResult:
    """Everything build_data.py needs to write the processed artifacts + metadata."""
    pcs: np.ndarray            # [T, D_max] unit-variance normalised PCs (float64 here)
    eofs: np.ndarray           # [D_max, S] spatial EOFs in WEIGHTED space (float64)
    pc_std: np.ndarray         # [D_max] std of each raw expansion-coefficient PC
    lat: np.ndarray            # [S] latitude of each ocean column
    lon: np.ndarray            # [S] longitude of each ocean column
    mask: np.ndarray           # [n_lat, n_lon] bool ocean mask over the subset grid
    variance_explained: np.ndarray   # [D_max] fraction of total variance per PC
    dates: np.ndarray          # [T] datetime64[M] of each retained time step
    grid_lat: np.ndarray       # [n_lat] subset-grid latitudes (ascending)
    grid_lon: np.ndarray       # [n_lon] subset-grid longitudes (0..360, ascending)
    flags: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------
def _import_xarray():
    try:
        import xarray as xr  # noqa: F401
        return xr
    except Exception as e:  # pragma: no cover - exercised only when dep missing
        raise ImportError(
            "xarray is required for preprocessing but could not be imported "
            f"({e!r}).\n  Install it in the LOCAL miniconda env, e.g.:\n"
            "    python3 -m pip install xarray netCDF4   # (or: conda install -c conda-forge xarray netcdf4)\n"
            "  The download step (fetch_ersst.py) is pure-stdlib and does NOT need xarray; "
            "only this preprocessing step does."
        ) from e


def load_dataset(nc_path: Path | str):
    """Open the ERSSTv5 netCDF-4 file with xarray, trying the 'netcdf4' engine then
    'h5netcdf'. (scipy.io CANNOT read this HDF5/netCDF-4 file — do not try it.)"""
    xr = _import_xarray()
    nc_path = Path(nc_path)
    if not nc_path.exists():
        raise FileNotFoundError(
            f"Raw SST file not found: {nc_path}\n"
            f"  Run the fetch step first: python3 -m experiments.exp_lim_enso.data.fetch_ersst")
    last_err = None
    for engine in ("netcdf4", "h5netcdf"):
        try:
            return xr.open_dataset(nc_path, engine=engine)
        except Exception as e:  # try the next engine
            last_err = e
    raise RuntimeError(
        f"Could not open {nc_path} with either 'netcdf4' or 'h5netcdf'. "
        f"Last error: {last_err!r}.\n  Ensure netCDF4 or h5netcdf is installed locally."
    )


# ----------------------------------------------------------------------------
def _normalize_lon_to_0_360(lon: np.ndarray) -> np.ndarray:
    """Map any lon convention into [0, 360)."""
    return np.mod(lon, 360.0)


def _subset_domain(ds, domain_lat, domain_lon):
    """Subset to lat in domain_lat and lon in domain_lon (degrees East, 0..360).

    Robust to (a) ascending vs descending lat, and (b) the dataset using either a
    0..360 or a -180..180 longitude convention. Returns
        sst[time, lat, lon]  (np.float64),  grid_lat[ascending],  grid_lon[0..360 ascending],
        times[datetime64[M]].
    ERSSTv5 dims/coords are 'lat','lon','time','sst' (standard); we accept a few aliases.
    """
    # --- resolve coordinate / variable names ---
    def pick(cands):
        for c in cands:
            if c in ds.coords or c in ds.variables or c in ds.dims:
                return c
        raise KeyError(f"none of {cands} present in dataset (have {list(ds.variables)})")

    latn = pick(["lat", "latitude", "y"])
    lonn = pick(["lon", "longitude", "x"])
    timen = pick(["time", "T"])
    sstn = "sst" if "sst" in ds.variables else pick(["sst", "sstanom", "SST"])

    da = ds[sstn]

    lat_vals = np.asarray(ds[latn].values, dtype=np.float64)
    lon_vals = np.asarray(ds[lonn].values, dtype=np.float64)

    # --- ensure ascending latitude (so .sel/slice and cos-weighting are well-defined) ---
    if lat_vals[0] > lat_vals[-1]:
        da = da.isel({latn: slice(None, None, -1)})
        lat_vals = lat_vals[::-1]

    # --- normalise longitude to 0..360 and sort ascending ---
    lon360 = _normalize_lon_to_0_360(lon_vals)
    lon_order = np.argsort(lon360)
    da = da.isel({lonn: lon_order})
    lon360 = lon360[lon_order]
    # reflect the reordered/relabeled lon back onto the DataArray coord
    da = da.assign_coords({lonn: lon360})

    # --- latitude subset (inclusive) ---
    lat_lo, lat_hi = sorted(domain_lat)
    lat_keep = (lat_vals >= lat_lo) & (lat_vals <= lat_hi)
    da = da.isel({latn: np.where(lat_keep)[0]})
    grid_lat = lat_vals[lat_keep]

    # --- longitude subset (inclusive, contiguous in 0..360 since domain is 30..290) ---
    lon_lo, lon_hi = domain_lon  # caller passes already in 0..360 East
    if lon_lo <= lon_hi:
        lon_keep = (lon360 >= lon_lo) & (lon360 <= lon_hi)
    else:  # wrap-around domain (not used by default 30..290, but supported)
        lon_keep = (lon360 >= lon_lo) | (lon360 <= lon_hi)
    da = da.isel({lonn: np.where(lon_keep)[0]})
    grid_lon = lon360[lon_keep]

    # --- transpose to (time, lat, lon) and pull values as float64 ---
    da = da.transpose(timen, latn, lonn)
    sst = np.asarray(da.values, dtype=np.float64)  # [T0, nlat, nlon]
    times = np.asarray(ds[timen].values)           # datetime64
    return sst, grid_lat, grid_lon, times, (latn, lonn, timen, sstn)


def _subset_period(sst, times, period):
    """Keep time steps within [period_start, period_end] inclusive (monthly resolution)."""
    if period is None:
        return sst, times
    start, end = period
    t = times.astype("datetime64[M]")
    t0 = np.datetime64(start, "M")
    t1 = np.datetime64(end, "M")
    keep = (t >= t0) & (t <= t1)
    return sst[keep], t[keep]


# ----------------------------------------------------------------------------
def _ocean_mask_and_flatten(sst):
    """Build an ocean mask and flatten valid columns.

    sst: [T, nlat, nlon] with NaN over land/ice (and the _FillValue already decoded to NaN
    by xarray). A column (gridpoint) is OCEAN iff it is finite at EVERY time step over the
    period. Returns:
        flat[T, S]   anomaly-ready data over ocean columns (still raw SST here, float64),
        mask[nlat,nlon] bool,  col_latidx[S], col_lonidx[S]  (grid indices of each column).
    """
    T, nlat, nlon = sst.shape
    flat_all = sst.reshape(T, nlat * nlon)            # [T, nlat*nlon]
    finite_all_time = np.isfinite(flat_all).all(axis=0)   # [nlat*nlon] bool
    mask = finite_all_time.reshape(nlat, nlon)
    cols = np.where(finite_all_time)[0]
    flat = flat_all[:, cols]                          # [T, S]
    col_latidx = cols // nlon
    col_lonidx = cols % nlon
    return flat, mask, col_latidx, col_lonidx


def _remove_monthly_climatology(flat, dates):
    """Subtract the per-calendar-month mean (the seasonal cycle) from each column.

    flat: [T, S]; dates: [T] datetime64[M]. Returns anomalies [T, S] (float64) and the
    [12, S] climatology used (months 1..12)."""
    months = dates.astype("datetime64[M]").astype(int) % 12  # 0..11
    clim = np.zeros((12, flat.shape[1]), dtype=np.float64)
    for m in range(12):
        sel = months == m
        clim[m] = flat[sel].mean(axis=0)
    anom = flat - clim[months]
    return anom, clim


def _linear_detrend(anom):
    """Per-column least-squares linear detrend (remove the secular/warming trend).

    AFTER climatology removal, BEFORE EOF — otherwise EOF1 is the global-warming trend,
    not ENSO. Returns detrended anomalies [T, S] (float64) and the fitted slopes [S]
    (anom-units per time-step) for diagnostics."""
    T = anom.shape[0]
    t = np.arange(T, dtype=np.float64)
    t = t - t.mean()                       # centre for numerical stability
    denom = float((t * t).sum())
    slope = (t[:, None] * anom).sum(axis=0) / denom    # [S]
    intercept = anom.mean(axis=0)                      # since t is centred
    trend = intercept[None, :] + slope[None, :] * t[:, None]
    return anom - trend, slope


def _running_mean_3(anom, dates):
    """Centred 3-month running mean; trim the 1 step at each edge the window can't fill.

    Returns smoothed anomalies [T-2, S] (float64) and the correspondingly-trimmed dates."""
    if anom.shape[0] < RUNNING_MEAN_WINDOW:
        raise ValueError(f"need >= {RUNNING_MEAN_WINDOW} time steps for the running mean, "
                         f"have {anom.shape[0]}")
    kernel = np.ones(RUNNING_MEAN_WINDOW, dtype=np.float64) / RUNNING_MEAN_WINDOW
    # 'valid' convolution along time for every column
    sm = np.apply_along_axis(lambda c: np.convolve(c, kernel, mode="valid"), 0, anom)
    half = RUNNING_MEAN_WINDOW // 2
    trimmed_dates = dates[half: anom.shape[0] - half]
    return sm, trimmed_dates


def _area_weight(anom, col_lat):
    """Multiply each column by sqrt(cos(lat)) so the EOF inner product is area-fair.

    anom: [T, S]; col_lat: [S] latitudes (deg). Returns weighted anomalies [T, S] and the
    per-column weights [S]. (Clip cos at 0 to avoid sqrt of tiny negatives at |lat|~90.)"""
    w = np.sqrt(np.clip(np.cos(np.deg2rad(col_lat)), 0.0, None))   # [S]
    return anom * w[None, :], w


def _fix_eof_sign(eofs, pcs, col_lat, col_lon, weights):
    """Deterministically fix the arbitrary SVD sign of each (EOF, PC) pair.

    Convention: make the area-weighted mean of the (un-weighted) spatial loading over the
    eastern-equatorial Nino-ish box POSITIVE, so the leading EOF's positive phase is warm
    (El Nino). Falls back to 'largest-|loading| element is positive' if the box is empty.

    eofs: [k, S] (weighted space), pcs: [T, k]. Modifies/returns flipped copies."""
    eofs = eofs.copy(); pcs = pcs.copy()
    in_box = ((col_lat >= SIGN_BOX_LAT[0]) & (col_lat <= SIGN_BOX_LAT[1]) &
              (col_lon >= SIGN_BOX_LON[0]) & (col_lon <= SIGN_BOX_LON[1]))
    # un-weight the spatial loading (EOF is in weighted space; divide by weight) to judge
    # the physical sign; guard against divide-by-zero weights.
    safe_w = np.where(weights > 0, weights, 1.0)
    for j in range(eofs.shape[0]):
        phys = eofs[j] / safe_w
        if in_box.any():
            score = phys[in_box].mean()
        else:
            score = phys[np.argmax(np.abs(phys))]
        if score < 0:
            eofs[j] *= -1.0
            pcs[:, j] *= -1.0
    return eofs, pcs


# ----------------------------------------------------------------------------
def preprocess(nc_path: Path | str,
               domain_lat=DEFAULT_DOMAIN_LAT,
               domain_lon=DEFAULT_DOMAIN_LON,
               period=DEFAULT_PERIOD,
               d_max: int = DEFAULT_D_MAX) -> PreprocResult:
    """Run the full LIM/ENSO preprocessing in float64. Returns a PreprocResult.

    See the module docstring for the conventions. ``domain_lon`` is interpreted in
    degrees-East (0..360). Computes the economy SVD ONCE at ``d_max`` (nested basis), so
    downstream code slices ``pcs[:, :D]`` for any D <= d_max."""
    ds = load_dataset(nc_path)
    try:
        sst, grid_lat, grid_lon, times, names = _subset_domain(ds, domain_lat, domain_lon)
    finally:
        ds.close()

    sst, dates = _subset_period(sst, times, period)
    T_raw = sst.shape[0]
    if T_raw < 24:
        raise ValueError(f"only {T_raw} months after period subset — too short for EOFs.")

    flat, mask, col_latidx, col_lonidx = _ocean_mask_and_flatten(sst)
    S = flat.shape[1]
    if S < d_max:
        raise ValueError(f"only {S} ocean gridpoints in the domain (< d_max={d_max}).")
    col_lat = grid_lat[col_latidx]
    col_lon = grid_lon[col_lonidx]

    anom, _clim = _remove_monthly_climatology(flat, dates)
    anom, _slope = _linear_detrend(anom)
    anom, dates = _running_mean_3(anom, dates)
    anom_w, weights = _area_weight(anom, col_lat)

    # economy SVD of the weighted, time-by-space anomaly matrix A_w = U S V^T.
    # full_matrices=False -> U:[T,k], svals:[k], Vt:[k,S] with k=min(T,S).
    U, svals, Vt = np.linalg.svd(anom_w, full_matrices=False)
    total_var = float((svals ** 2).sum())
    var_frac_full = (svals ** 2) / total_var          # [k]

    k = int(d_max)
    pcs_raw = U[:, :k] * svals[:k]                     # [T, k] expansion coefficients
    eofs = Vt[:k, :]                                   # [k, S] spatial patterns (weighted)
    var_explained = var_frac_full[:k].astype(np.float64)

    eofs, pcs_raw = _fix_eof_sign(eofs, pcs_raw, col_lat, col_lon, weights)

    # unit-variance normalise each PC; store the std for un-normalising.
    pc_std = pcs_raw.std(axis=0, ddof=0)              # [k]
    pc_std_safe = np.where(pc_std > 0, pc_std, 1.0)
    pcs = pcs_raw / pc_std_safe[None, :]              # [T, k], unit variance

    flags = {
        "climatology_removed": True,
        "linear_detrend": True,
        "running_mean_window": RUNNING_MEAN_WINDOW,
        "area_weight": "sqrt(cos(lat))",
        "svd": "economy (full_matrices=False) of weighted time-by-space anomalies",
        "pc_normalization": "unit-variance per PC (ddof=0); pc_std stored for inverse",
        "eof_space": "weighted (EOF = right singular vectors V of A_w)",
        "sign_convention": (
            "flip so area-weighted mean loading over Nino box "
            f"lat{SIGN_BOX_LAT} lon{SIGN_BOX_LON} is positive (warm=positive); "
            "fallback largest-|loading| positive"),
        "lon_convention": "0..360 East",
        "lat_order": "ascending",
        "dtype_compute": "float64 (cast to float32 only at write)",
        "n_ocean_gridpoints_S": int(S),
        "svd_rank_k": int(min(U.shape[1], Vt.shape[0])),
        "total_variance": total_var,
    }

    return PreprocResult(
        pcs=pcs, eofs=eofs, pc_std=pc_std, lat=col_lat, lon=col_lon, mask=mask,
        variance_explained=var_explained, dates=dates.astype("datetime64[M]"),
        grid_lat=grid_lat, grid_lon=grid_lon, flags=flags,
    )


# ----------------------------------------------------------------------------
def leading_pc_autocorr(pcs: np.ndarray, lags=(1, 6, 12)) -> dict:
    """Autocorrelation of the leading PC at the given lags (months). ENSO-like PC1 has
    strong positive autocorr at 6-12 months (ENSO's ~2-7yr quasi-periodicity / persistence
    barrier), unlike white noise. Used by build_data.py for the ENSO sanity print."""
    x = np.asarray(pcs[:, 0], dtype=np.float64)
    x = x - x.mean()
    var = float((x * x).sum())
    out = {}
    for L in lags:
        if L < len(x):
            out[int(L)] = float((x[:-L] * x[L:]).sum() / var) if var > 0 else 0.0
    return out
