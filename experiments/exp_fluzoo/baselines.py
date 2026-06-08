############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# baselines.py: Non-DMCI forecast baselines, scored on the SAME test origins/horizons. These calibrate the "does the program...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Non-DMCI forecast baselines, scored on the SAME test origins/horizons.

These calibrate the "does the program search buy anything" question:
  * persistence       -- yhat(o+h) = last observed value y(o)
  * damped_persistence-- yhat(o+h) = y(o) * damp^h
  * seasonal_naive    -- yhat(o+h) = training-season climatology at that within-season week
  * ar                -- per-region AR(p) least-squares, rolled h steps from the observed window

The classical-mechanistic baseline (a hand-written SEIR) and the random-grammar+DMCI and
direct-compile baselines are scored through the SAME model path (forecast.filter_then_forecast)
using, respectively, programs.reference_program("seir_regional"), a random-grammar program,
and the direct compiler -- so they need no separate scorer here.
"""

from __future__ import annotations

import numpy as np

from .config import DEFAULT
from .forecast import season_matrix, forecast_metrics, origins


def _climatology(data: dict, cfg=DEFAULT) -> np.ndarray:
    """Mean wILI per (within-season week index, region) over the TRAINING seasons -> [W, R]."""
    mats = [season_matrix(data, s) for s in cfg.train_seasons]
    W = min(m.shape[0] for m in mats)
    stack = np.stack([m[:W] for m in mats], axis=0)   # [n_seasons, W, R]
    return stack.mean(axis=0)                          # [W, R]


def _ar_fit(series: np.ndarray, p: int) -> np.ndarray:
    """Least-squares AR(p) coefficients (with intercept) for a 1-D series."""
    if len(series) <= p + 1:
        return np.zeros(p + 1)
    X = np.column_stack([series[p - 1 - i: len(series) - 1 - i] for i in range(p)]
                        + [np.ones(len(series) - p)])
    y = series[p:]
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef


def _ar_forecast(coef: np.ndarray, hist: np.ndarray, h: int, p: int) -> float:
    buf = list(hist[-p:]) if len(hist) >= p else [hist[-1]] * p
    for _ in range(h):
        x = np.array(buf[-p:][::-1] + [1.0])
        nxt = float(np.dot(coef, x))
        buf.append(max(nxt, 0.0))
    return buf[-1]


def _score(predict_fn, data, cfg, test_seasons, stride) -> dict:
    bucket = {h: {"pred": [], "true": []} for h in cfg.horizons}
    for season in test_seasons:
        S = season_matrix(data, season)
        L = S.shape[0]
        if L < 12:
            continue
        for o in origins(L, cfg.horizons, stride):
            for h in cfg.horizons:
                w = o + h
                if w >= L:
                    continue
                bucket[h]["pred"].append(np.asarray(predict_fn(S, o, h)))
                bucket[h]["true"].append(S[w])
    metrics = {f"h{h}": forecast_metrics(bucket[h]["pred"], bucket[h]["true"])
               for h in cfg.horizons}
    metrics["mean_rmse"] = float(np.nanmean([metrics[f"h{h}"]["rmse"] for h in cfg.horizons]))
    return metrics


def score_baselines(data: dict, cfg=DEFAULT, test_seasons=None, stride: int = 3,
                    ar_p: int = 3, damp: float = 0.9) -> dict:
    """Return {baseline_name: per-horizon metrics} on the held-out test seasons."""
    test_seasons = cfg.test_seasons if test_seasons is None else test_seasons
    clim = _climatology(data, cfg)                       # [W, R]
    Wc = clim.shape[0]
    # Per-region AR coefficients fit on concatenated training seasons.
    train = np.concatenate([season_matrix(data, s) for s in cfg.train_seasons], axis=0)
    ar_coef = [_ar_fit(train[:, r], ar_p) for r in range(cfg.n_regions)]

    def persistence(S, o, h):
        return S[o]

    def damped(S, o, h):
        return S[o] * (damp ** h)

    def seasonal_naive(S, o, h):
        w = min(o + h, Wc - 1)
        return clim[w]

    def ar(S, o, h):
        return np.array([_ar_forecast(ar_coef[r], S[:o + 1, r], h, ar_p)
                         for r in range(cfg.n_regions)])

    return {
        "persistence": _score(persistence, data, cfg, test_seasons, stride),
        "damped_persistence": _score(damped, data, cfg, test_seasons, stride),
        "seasonal_naive": _score(seasonal_naive, data, cfg, test_seasons, stride),
        f"ar{ar_p}": _score(ar, data, cfg, test_seasons, stride),
    }
