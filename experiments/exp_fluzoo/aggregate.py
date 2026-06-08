############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# aggregate.py: Build the FluZoo manuscript tables and figure data from the sweep outputs. Reads results/run_all_summary.json,...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Build the FluZoo manuscript tables and figure data from the sweep outputs.

Reads results/run_all_summary.json, results/<name>.json, and llm_cache/<key>.json; writes
results/agg/:
  T1_funnel.csv          -- discrete-search funnel: proposals -> accepted -> unique structures
  T2_forecast.csv        -- selected program vs. baselines, test RMSE by horizon + peak error
  T3_scaling.csv/.dat    -- best held-out skill vs. number of programs searched (the headline)
  diversity.csv          -- accepted-program family histogram and parameter-count distribution
  agg_summary.json       -- machine-readable roll-up
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from .config import DEFAULT
from .validity import STAGES

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
CACHE_DIR = HERE / "llm_cache"


def _zoo_records() -> list[dict]:
    out = []
    for p in sorted(CACHE_DIR.glob("*.json")):
        try:
            out.append(json.loads(p.read_text()))
        except Exception:  # noqa: BLE001
            pass
    return out


def funnel_table(records: list[dict]) -> list[tuple]:
    """Counts reaching each funnel stage (+ accepted, unique structures)."""
    n = len(records)
    rows = [("proposed", n, 1.0)]
    for i, stage in enumerate(STAGES[1:], start=1):
        c = 0
        for r in records:
            st = r.get("stage", "proposed")
            reached = STAGES.index(st) if st in STAGES else 0
            if r.get("status") == "accepted" or i < reached:
                if i <= reached:
                    c += 1
        rows.append((stage, c, c / n if n else 0.0))
    accepted = [r for r in records if r.get("status") == "accepted"]
    rows.append(("accepted", len(accepted), len(accepted) / n if n else 0.0))
    uniq = len({r.get("canonical") for r in accepted if r.get("canonical")})
    rows.append(("unique_structures", uniq, uniq / n if n else 0.0))
    return rows


def _write_csv(path: Path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def aggregate(output_dir: Path = RESULTS, cfg=DEFAULT) -> dict:
    agg = output_dir / "agg"
    agg.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "run_all_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

    # ---- T1: funnel ------------------------------------------------------
    zoo = _zoo_records()
    if zoo:
        frows = funnel_table(zoo)
        _write_csv(agg / "T1_funnel.csv", ["stage", "count", "fraction"], frows)

    # ---- T2: forecast skill (selected program vs baselines) --------------
    sel = summary.get("selected")
    base_test = summary.get("baselines", {}).get("test", {})
    t2 = []
    if sel:
        hb = sel.get("test_by_horizon", {})
        row = ["FluZoo (selected)"] + [round(hb.get(f"h{h}", {}).get("rmse", float("nan")), 5)
                                       for h in cfg.horizons] + [round(sel["val_mean_rmse"], 5)]
        t2.append(row)
    # baselines: per-horizon RMSE from results (recompute mean only available -> read full)
    for name, mean_rmse in base_test.items():
        t2.append([name] + ["" for _ in cfg.horizons] + [round(mean_rmse, 5)])
    _write_csv(agg / "T2_forecast.csv",
               ["method"] + [f"test_rmse_{h}w" for h in cfg.horizons] + ["val_mean_rmse"], t2)

    # ---- T3: skill vs programs searched (headline figure) ----------------
    curve = summary.get("skill_curve", [])
    if curve:
        _write_csv(agg / "T3_scaling.csv", ["k", "best_val_rmse", "selected_test_rmse"],
                   [(c["k"], round(c["best_val_rmse"], 6), round(c["selected_test_rmse"], 6))
                    for c in curve])
        # pgfplots .dat (whitespace, header comment)
        dat = agg / "T3_scaling.dat"
        with dat.open("w") as f:
            f.write("k best_val_rmse selected_test_rmse\n")
            for c in curve:
                f.write(f"{c['k']} {c['best_val_rmse']:.6f} {c['selected_test_rmse']:.6f}\n")

    # ---- diversity: family histogram + param counts ----------------------
    accepted = [r for r in zoo if r.get("status") == "accepted"]
    fam = Counter()
    for r in accepted:
        rc = r.get("recipe", {})
        if isinstance(rc, dict):
            fam[rc.get("compartmental", "?")] += 1
    _write_csv(agg / "diversity.csv", ["compartmental_family", "count"], sorted(fam.items()))

    out = {
        "n_zoo": len(zoo),
        "n_accepted": len(accepted),
        "n_unique": len({r.get("canonical") for r in accepted if r.get("canonical")}),
        "selected": sel,
        "baselines_test": base_test,
        "tables": [str(p.relative_to(output_dir)) for p in sorted(agg.glob("*"))],
    }
    (agg / "agg_summary.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"[aggregate] wrote {agg}/ : {', '.join(p.name for p in sorted(agg.glob('*')))}")
    return out


if __name__ == "__main__":
    aggregate()
