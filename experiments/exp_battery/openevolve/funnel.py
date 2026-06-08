############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# funnel.py: Canonical-AST distinct-structure funnel over an OpenEvolve battery archive. Turns "the LLM searches the...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Canonical-AST distinct-structure funnel over an OpenEvolve battery archive.

Turns "the LLM searches the discrete space of programs" from an assertion into a measured count,
and tests recovery-vs-novelty. For each island it loads EVERY evaluated program, structurally
fingerprints it with FluZoo's canonical_hash (positional symbol renaming + numeric-literal
bucketing on the loop body, so programs differing only in parameter names/constants collapse to
one hash), and reports: evaluated, parse-valid, DISTINCT structures (ASTs), how many match one of
the five hand-written reference families, and how many are NOVEL (outside the reference set).

    python3 -m experiments.exp_battery.openevolve.funnel --prefix bat_island        # synthetic
    python3 -m experiments.exp_battery.openevolve.funnel --prefix bat_real_island   # real Severson
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

sys.setrecursionlimit(20000)

from experiments.exp_fluzoo.programs import parse_program
from experiments.exp_fluzoo.validity import canonical_hash
from experiments.exp_battery.structures import STRUCTURES

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, "..", "results"))


def model_body(code: str):
    m = re.search(r'BATTERY_MODEL\s*=\s*r?"""(.*?)"""', code or "", re.DOTALL)
    return m.group(1) if m else None


def safe_hash(src):
    try:
        return canonical_hash(parse_program(src, name="c"))
    except Exception:  # noqa: BLE001
        return None


def latest_programs(island_dir):
    """All evaluated programs from the island's last checkpoint (the archive accumulates them)."""
    cps = sorted(glob.glob(os.path.join(island_dir, "checkpoints", "checkpoint_*")),
                 key=lambda p: int(p.rsplit("_", 1)[1]))
    out = []
    if not cps:
        return out
    for f in glob.glob(os.path.join(cps[-1], "programs", "*.json")):
        try:
            d = json.load(open(f))
        except Exception:  # noqa: BLE001
            continue
        body = model_body(d.get("code", ""))
        if body is not None:
            out.append(body)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="bat_island")
    args = ap.parse_args()

    ref_hashes = {n: safe_hash(s) for n, s in STRUCTURES.items()}
    ref_set = {h for h in ref_hashes.values() if h}
    print("reference families (canonical hashes):")
    for n, h in ref_hashes.items():
        print(f"  {n:18s} {h}")

    islands = sorted(glob.glob(os.path.join(RESULTS, f"{args.prefix}_*")))
    rep = {"prefix": args.prefix, "reference_hashes": ref_hashes, "islands": {}}
    pooled, pooled_novel = set(), set()
    for idir in islands:
        name = os.path.basename(idir)
        progs = latest_programs(idir)
        valid = [h for h in (safe_hash(b) for b in progs) if h]
        distinct = set(valid)
        novel = distinct - ref_set
        pooled |= distinct
        pooled_novel |= novel
        rep["islands"][name] = {"evaluated": len(progs), "parse_valid": len(valid),
                                "distinct_structures": len(distinct),
                                "distinct_matching_reference": len(distinct & ref_set),
                                "distinct_novel": len(novel)}
        print(f"\n{name}: {len(progs)} evaluated | {len(valid)} parse-valid | "
              f"{len(distinct)} DISTINCT structures "
              f"({len(distinct & ref_set)} match a reference, {len(novel)} novel)")
    rep["pooled"] = {"distinct_structures": len(pooled), "distinct_novel": len(pooled_novel),
                     "distinct_matching_reference": len(pooled & ref_set),
                     "reference_families": len(ref_set)}
    print(f"\nPOOLED: {len(pooled)} distinct structures across islands | "
          f"{len(pooled & ref_set)} match a reference family | {len(pooled_novel)} novel "
          f"(outside the {len(ref_set)} hand-written references)")
    out = os.path.join(RESULTS, f"funnel_{args.prefix}.json")
    json.dump(rep, open(out, "w"), indent=2, default=str)
    print(f"WROTE {out}")


if __name__ == "__main__":
    main()
