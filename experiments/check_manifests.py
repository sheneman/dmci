############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# check_manifests.py: Per-results-directory completeness manifests for the committed raw experiment data. Each...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Per-results-directory completeness manifests for the committed raw experiment data.

Each experiments/<exp>/results[_llm]/ directory holds the per-run raw outputs the paper's
tables and figures are derived from (committed via Git LFS; see .gitattributes). To let a
reviewer confirm completeness from a committed inventory -- rather than by counting files
against the manuscript by hand -- this script maintains a `MANIFEST.txt` in each such
directory listing the expected file basenames and count, and verifies the committed data
against it.

Usage:
    python3 -m experiments.check_manifests            # check each MANIFEST.txt vs git ls-files
    python3 -m experiments.check_manifests --write    # (re)generate the MANIFEST.txt files

On a fresh clone the actual file *contents* are LFS pointers until `git lfs pull`, but the
file *inventory* (names/count) is intact immediately, so this completeness check works
without pulling the large objects. The aggregate_table.py scripts verify the contents.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = "MANIFEST.txt"

# Result directories whose committed raw outputs back the paper's tables/figures.
RESULT_DIRS = [
    "experiments/exp_a/results",
    "experiments/exp_b/results",
    "experiments/exp_b/results_llm",
    "experiments/exp_c/results",
    "experiments/exp_d/results",
    "experiments/exp_h/results",
]


def tracked_basenames(reldir: str) -> list[str]:
    """Git-tracked file basenames in `reldir` (excluding the MANIFEST itself), sorted."""
    out = subprocess.run(
        ["git", "ls-files", reldir], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.split()
    names = sorted(Path(p).name for p in out if Path(p).name != MANIFEST)
    return names


def manifest_text(reldir: str, names: list[str]) -> str:
    return (
        f"# Inventory of {reldir} -- committed raw outputs (Git LFS).\n"
        f"# {len(names)} files. Regenerate/verify with: python3 -m experiments.check_manifests\n"
        + "\n".join(names) + "\n"
    )


def read_manifest(path: Path) -> list[str]:
    if not path.exists():
        return []
    return sorted(
        line.strip() for line in path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="(re)generate the MANIFEST.txt files")
    args = ap.parse_args()

    ok = True
    for reldir in RESULT_DIRS:
        names = tracked_basenames(reldir)
        mpath = REPO_ROOT / reldir / MANIFEST
        if args.write:
            mpath.write_text(manifest_text(reldir, names))
            print(f"wrote {reldir}/{MANIFEST} ({len(names)} files)")
            continue
        expected = read_manifest(mpath)
        if not expected:
            print(f"  MISSING  {reldir}/{MANIFEST}")
            ok = False
            continue
        missing = sorted(set(expected) - set(names))
        extra = sorted(set(names) - set(expected))
        if missing or extra:
            ok = False
            print(f"  MISMATCH {reldir}: committed={len(names)} manifest={len(expected)}"
                  f"{'  missing=' + str(missing[:3]) if missing else ''}"
                  f"{'  extra=' + str(extra[:3]) if extra else ''}")
        else:
            print(f"  OK       {reldir}: {len(names)} files match manifest")
    if not args.write:
        print("\nMANIFEST CHECK: " + ("all directories match" if ok else "MISMATCHES FOUND"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
