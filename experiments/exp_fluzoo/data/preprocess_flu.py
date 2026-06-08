############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# preprocess_flu.py: Build the [T, 11] weighted-%ILI observation matrix and season-aware splits. Turns the per-region Delphi Epidata...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Build the [T, 11] weighted-%ILI observation matrix and season-aware splits.

Turns the per-region Delphi Epidata rows into the binding contract every FluZoo
program consumes: obs[T, R] where row = epiweek (chronological), column = region in
config.REGIONS order (national, HHS-1..10), value = weighted %ILI expressed as a
PROPORTION (wili / 100, i.e. in [0, ~0.08]) so it matches the model observable
rho * I (a fraction of the population). Season masks (train / val / test / pandemic)
are derived from the MMWR season convention: season Y = epiweek Yw40 .. (Y+1)w39.
No torch dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def epiweek_season(epiweek: int) -> int:
    """MMWR influenza-season start year for an epiweek (week>=40 -> that year)."""
    year, week = divmod(int(epiweek), 100)
    return year if week >= 40 else year - 1


@dataclass
class FluData:
    epiweeks: np.ndarray          # [T] int  (chronological)
    seasons: np.ndarray           # [T] int  (season start year per week)
    regions: tuple[str, ...]      # length R, column order
    wili: np.ndarray              # [T, R] float32, PROPORTION (wili/100)
    masks: dict[str, np.ndarray]  # split name -> [T] bool
    release_dates: dict[str, str] = field(default_factory=dict)  # provenance per region

    @property
    def T(self) -> int:
        return self.wili.shape[0]

    def split(self, name: str) -> np.ndarray:
        """The [n_weeks, R] sub-matrix for a named split (rows in chronological order)."""
        return self.wili[self.masks[name]]


def preprocess(by_region: dict[str, list[dict]], regions: tuple[str, ...],
               train_seasons, val_seasons, test_seasons, pandemic_seasons) -> FluData:
    # National defines the canonical timeline; every region is aligned onto it.
    ref = regions[0]
    epiweeks = np.array(sorted(r["epiweek"] for r in by_region[ref]), dtype=np.int64)
    T = len(epiweeks)
    idx = {int(ew): i for i, ew in enumerate(epiweeks)}

    wili = np.full((T, len(regions)), np.nan, dtype=np.float64)
    release_dates: dict[str, str] = {}
    for c, region in enumerate(regions):
        rows = by_region[region]
        if rows:
            release_dates[region] = max(r.get("release_date", "") for r in rows)
        for row in rows:
            i = idx.get(int(row["epiweek"]))
            if i is not None and row.get("wili") is not None:
                wili[i, c] = float(row["wili"]) / 100.0   # percent -> proportion

    n_nan = int(np.isnan(wili).sum())
    if n_nan:
        # Linear-interpolate the rare missing cell per region; report it.
        print(f"[preprocess] WARNING: {n_nan} missing (region,week) cells; interpolating")
        for c in range(wili.shape[1]):
            col = wili[:, c]
            m = np.isnan(col)
            if m.any():
                col[m] = np.interp(np.flatnonzero(m), np.flatnonzero(~m), col[~m])
    wili = wili.astype(np.float32)

    seasons = np.array([epiweek_season(int(ew)) for ew in epiweeks], dtype=np.int64)

    def mask_for(season_set) -> np.ndarray:
        s = set(int(x) for x in season_set)
        return np.array([int(se) in s for se in seasons], dtype=bool)

    masks = {
        "train": mask_for(train_seasons),
        "val": mask_for(val_seasons),
        "test": mask_for(test_seasons),
        "pandemic": mask_for(pandemic_seasons),
    }
    for name, m in masks.items():
        nweeks = int(m.sum())
        ssn = sorted(set(int(s) for s in seasons[m]))
        print(f"[preprocess] {name:8s}: {nweeks:4d} weeks  seasons={ssn}")

    return FluData(epiweeks=epiweeks, seasons=seasons, regions=tuple(regions),
                   wili=wili, masks=masks, release_dates=release_dates)
