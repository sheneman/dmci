############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# fetch_ersst.py: Resumable streaming download of NOAA ERSSTv5 monthly-mean SST into data/raw/. The source file (verified) is...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Resumable streaming download of NOAA ERSSTv5 monthly-mean SST into data/raw/.

The source file (verified) is HDF5/netCDF-4, exactly 152_446_809 bytes, served with
``accept-ranges: bytes`` so we support HTTP Range resume of a partial download.

This module is PURE STDLIB (urllib) — the fetch must not introduce a hard non-stdlib
dependency. (xarray/netCDF4 are only needed by the *preprocess* step.) ``requests`` is
used opportunistically if importable, but is never required.

Run locally on the Mac (network here is not guaranteed inside the agent sandbox):

    python3 -m experiments.exp_lim_enso.data.fetch_ersst            # resume / skip if complete
    python3 -m experiments.exp_lim_enso.data.fetch_ersst --force    # re-download from scratch
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# --- Verified contract (do NOT change without re-verifying upstream) ---------
SOURCE_URL = "https://downloads.psl.noaa.gov/Datasets/noaa.ersst.v5/sst.mnmean.nc"
EXPECTED_BYTES = 152_446_809
FILENAME = "sst.mnmean.nc"

# Directory layout: .../data/raw/sst.mnmean.nc (raw/ is gitignored).
RAW_DIR = Path(__file__).resolve().parent / "raw"
RAW_PATH = RAW_DIR / FILENAME

_CHUNK = 1 << 20  # 1 MiB streaming chunks
_USER_AGENT = "nncompile-exp_lim_enso-fetch/1.0 (+research; stdlib-urllib)"


# ----------------------------------------------------------------------------
def _sha256(path: Path, chunk: int = _CHUNK) -> str:
    """Streaming sha256 of a file (does not load it into memory)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _fmt_mb(n: int) -> str:
    return f"{n / 1e6:.1f} MB"


def _server_supports_ranges_and_size(url: str) -> tuple[bool, int | None]:
    """HEAD-ish probe: report whether the server advertises byte ranges and the
    Content-Length it reports. Best-effort — returns (False, None) on any error so
    the caller can fall back to a fresh GET."""
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            accepts = (resp.headers.get("Accept-Ranges", "").lower() == "bytes")
            clen = resp.headers.get("Content-Length")
            return accepts, (int(clen) if clen is not None else None)
    except Exception:
        return False, None


def _stream_to_file(url: str, dest: Path, resume_from: int = 0) -> None:
    """Stream ``url`` into ``dest``. If ``resume_from`` > 0, issue a Range request
    and append; otherwise truncate-write from the start."""
    headers = {"User-Agent": _USER_AGENT}
    mode = "wb"
    if resume_from > 0:
        headers["Range"] = f"bytes={resume_from}-"
        mode = "ab"

    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=120) as resp:
        status = getattr(resp, "status", None) or resp.getcode()
        # If we asked to resume but the server ignored Range (200 not 206), restart
        # cleanly to avoid corrupting the file by appending a full body to a partial.
        if resume_from > 0 and status != 206:
            print(f"  server ignored Range (HTTP {status}); restarting from 0", flush=True)
            resp.close()
            return _stream_to_file(url, dest, resume_from=0)

        total = resume_from
        target = EXPECTED_BYTES
        t0 = time.time()
        last_print = 0.0
        with open(dest, mode) as f:
            while True:
                block = resp.read(_CHUNK)
                if not block:
                    break
                f.write(block)
                total += len(block)
                now = time.time()
                if now - last_print > 1.0:
                    pct = 100.0 * total / target if target else 0.0
                    rate = (total - resume_from) / max(now - t0, 1e-9) / 1e6
                    print(f"  {_fmt_mb(total)} / {_fmt_mb(target)} "
                          f"({pct:5.1f}%)  {rate:5.1f} MB/s", flush=True)
                    last_print = now
    print(f"  wrote {_fmt_mb(total)} total", flush=True)


def fetch(force: bool = False, url: str = SOURCE_URL, dest: Path = RAW_PATH) -> Path:
    """Download (or resume / skip) the ERSSTv5 file. Returns the path on success.

    Steps:
      1. If a complete, correctly-sized file already exists and ``not force`` -> skip
         (after logging its sha256).
      2. Otherwise, if a partial exists and the server supports ranges -> resume.
      3. Else download fresh.
    Always asserts the final size == EXPECTED_BYTES and logs the sha256.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    if force and dest.exists():
        print(f"--force: removing existing {dest}", flush=True)
        dest.unlink()

    if dest.exists() and dest.stat().st_size == EXPECTED_BYTES and not force:
        print(f"Already complete: {dest} ({_fmt_mb(EXPECTED_BYTES)})", flush=True)
        digest = _sha256(dest)
        print(f"sha256: {digest}", flush=True)
        return dest

    accepts_ranges, server_size = _server_supports_ranges_and_size(url)
    if server_size is not None and server_size != EXPECTED_BYTES:
        print(f"WARNING: server Content-Length {server_size} != expected "
              f"{EXPECTED_BYTES}. Proceeding but will assert on final size.", flush=True)

    have = dest.stat().st_size if dest.exists() else 0
    if 0 < have < EXPECTED_BYTES and accepts_ranges:
        print(f"Resuming from {_fmt_mb(have)} (server supports byte ranges)", flush=True)
        _stream_to_file(url, dest, resume_from=have)
    else:
        if have >= EXPECTED_BYTES:
            # Oversized/garbage partial — start clean.
            print(f"Existing file size {have} unusable; restarting.", flush=True)
            dest.unlink()
        elif have > 0 and not accepts_ranges:
            print("Partial present but server does not support ranges; restarting.",
                  flush=True)
            dest.unlink()
        print(f"Downloading {url}", flush=True)
        _stream_to_file(url, dest, resume_from=0)

    actual = dest.stat().st_size
    assert actual == EXPECTED_BYTES, (
        f"Download size mismatch: got {actual} bytes, expected {EXPECTED_BYTES}. "
        f"Re-run (it will resume) or pass --force to restart."
    )
    digest = _sha256(dest)
    print(f"OK: {dest}", flush=True)
    print(f"size:   {actual} bytes ({_fmt_mb(actual)})", flush=True)
    print(f"sha256: {digest}", flush=True)
    return dest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Resumable download of NOAA ERSSTv5 sst.mnmean.nc into data/raw/.")
    ap.add_argument("--force", action="store_true",
                    help="ignore any existing file and re-download from scratch")
    ap.add_argument("--url", default=SOURCE_URL, help="source URL (default: verified PSL URL)")
    ap.add_argument("--dest", type=Path, default=RAW_PATH, help="destination path")
    args = ap.parse_args(argv)
    try:
        fetch(force=args.force, url=args.url, dest=args.dest)
    except AssertionError as e:
        print(f"FETCH FAILED: {e}", file=sys.stderr)
        return 1
    except urllib.error.URLError as e:
        print(f"NETWORK ERROR: {e}\n"
              f"  The agent sandbox has no guaranteed network; run this locally on the Mac.",
              file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
