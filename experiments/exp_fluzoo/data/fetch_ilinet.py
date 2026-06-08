############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# fetch_ilinet.py: Fetch CDC ILINet weighted %ILI from the public Delphi Epidata `fluview` endpoint. Pure standard library (urllib...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Fetch CDC ILINet weighted %ILI from the public Delphi Epidata `fluview` endpoint.

Pure standard library (urllib + json), resumable, sha256-verified -- the same
no-hard-dependency idiom as exp_lim_enso/data/fetch_ersst.py. One raw JSON file per
region is cached under data/raw/ (gitignored); only the processed arrays + a
metadata.json provenance record are kept. Delphi Epidata exposes ILINet data sourced
from CDC's ILINet dashboard; by default it returns the latest issue (finalised values)
for each epiweek. Revisions exist -- we record release_date/issue per row so a pinned
`as_of` snapshot can be reconstructed for strict reproducibility.

Run LOCALLY (the API is public, not campus-gated), then rsync data/processed to HPC.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

EPIDATA_URL = "https://api.delphi.cmu.edu/epidata/fluview/"

HERE = Path(__file__).resolve().parent
RAW_DIR = HERE / "raw"


def epiweek_window(min_season: int, max_season: int) -> tuple[int, int]:
    """Season Y spans epiweek Yw40 .. (Y+1)w39; return the inclusive epiweek window."""
    return (min_season * 100 + 40, (max_season + 1) * 100 + 39)


def _get(region: str, ew_start: int, ew_end: int, retries: int = 4) -> list[dict]:
    params = {"regions": region, "epiweeks": f"{ew_start}-{ew_end}"}
    url = EPIDATA_URL + "?" + urllib.parse.urlencode(params)
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                payload = json.load(r)
            if payload.get("result") != 1:
                raise RuntimeError(f"epidata result={payload.get('result')} "
                                   f"message={payload.get('message')!r}")
            return payload.get("epidata", [])
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"fetch failed for {region} {ew_start}-{ew_end}: {last}")


def fetch_region(region: str, ew_start: int, ew_end: int, *,
                 raw_dir: Path = RAW_DIR, force: bool = False) -> list[dict]:
    """Fetch (and cache) all fluview rows for one region over an epiweek window."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache = raw_dir / f"fluview_{region}_{ew_start}_{ew_end}.json"
    if cache.exists() and not force:
        return json.loads(cache.read_text())
    rows = _get(region, ew_start, ew_end)
    rows.sort(key=lambda r: r["epiweek"])
    cache.write_text(json.dumps(rows))
    return rows


def fetch_all(regions: tuple[str, ...], min_season: int, max_season: int, *,
              raw_dir: Path = RAW_DIR, force: bool = False) -> dict[str, list[dict]]:
    ew_start, ew_end = epiweek_window(min_season, max_season)
    out: dict[str, list[dict]] = {}
    for region in regions:
        rows = fetch_region(region, ew_start, ew_end, raw_dir=raw_dir, force=force)
        out[region] = rows
        print(f"[fetch] {region:6s} {ew_start}-{ew_end}: {len(rows)} weeks "
              f"(wILI {rows[0]['wili']:.3f} .. {rows[-1]['wili']:.3f})")
    return out


def source_sha256(by_region: dict[str, list[dict]]) -> str:
    """Stable content hash over the (region, epiweek, wili) triples for provenance."""
    h = hashlib.sha256()
    for region in sorted(by_region):
        for row in by_region[region]:
            h.update(f"{region}:{row['epiweek']}:{row['wili']}\n".encode())
    return h.hexdigest()


if __name__ == "__main__":
    import argparse

    from ..config import DEFAULT

    ap = argparse.ArgumentParser(description="Fetch ILINet wILI into data/raw/")
    ap.add_argument("--force", action="store_true", help="ignore cache and re-download")
    args = ap.parse_args()

    seasons = DEFAULT.all_seasons
    data = fetch_all(DEFAULT.regions, min(seasons), max(seasons), force=args.force)
    print(f"[fetch] sha256={source_sha256(data)}")
