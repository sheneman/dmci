############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# aggregate.py: Aggregate the LIM-ENSO per-run records into the manuscript tables + pgfplots data. Globs ``results/*.json``...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Aggregate the LIM-ENSO per-run records into the manuscript tables + pgfplots data.

Globs ``results/*.json`` (one per fit, written by ``runner.run_single``) and builds:

  T1  nll_by_solver        -- mean +/- std train-window NLL by solver x D (the fit objective;
                              reported for completeness, NOT the selection metric).
  T2  robustness_by_solver -- fraction of seeds within ``eps`` of the best NLL at each (D),
                              i.e. how reliably a solver reaches the best basin (exp_i robustness).
  T3  forecast_skill       -- held-out Nino-3.4 ACC / RMSE by lead for the fitted-LIM vs the
                              three references (persistence, damped persistence, Green-function).
                              This is the SCIENTIFIC headline; the Green-function operator is the
                              reference, never an NLL competitor.
  T4  scaling_by_D         -- NLL, ENSO timescales (period/decay), held-out ACC, per_step_ms,
                              and min_detS vs D (the scaling story; mirrors exp_i's d-scaling).

Outputs (mirroring exp_i.aggregate_scaling style):
  * printed tables to stdout,
  * CSV tables ``results/agg/T{1..4}_*.csv``,
  * pgfplots ``.dat`` files ``results/agg/*.dat`` (one row per D / lead, one col per series),
  * ``results/agg/scaling_gate_summary.json`` (machine-readable go/no-go + portfolio).

The portfolio winner is taken from each (structure, D) on held-out forecast skill (the
SAME criterion run_all uses), NEVER train NLL -- the exp_i overfit lesson.

    python3 -m experiments.exp_lim_enso.aggregate
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from .config import DEFAULT
from . import params


HERE = Path(__file__).parent
DEFAULT_RESULTS = HERE / "results"

# Forecast leads + baseline methods we report skill for.
LEADS = (3, 6, 9, 12)
SKILL_METHODS = ["fitted_lim", "persistence", "damped_persistence", "green_lim"]
# Within-eps-of-best-NLL robustness threshold (relative), mirroring exp_i robustness.
ROBUST_EPS_REL = 0.02
HEADLINE_LEAD = 6


# ===========================================================================
# Load.
# ===========================================================================

def _load_records(results_dir: Path) -> list[dict]:
    """Glob ``results/*.json`` (skip the summary) -> list of per-run records."""
    recs = []
    for f in sorted(results_dir.glob("*.json")):
        if f.name in ("run_all_summary.json",):
            continue
        try:
            r = json.loads(f.read_text())
        except Exception:  # noqa: BLE001
            continue
        if isinstance(r, dict) and "method" in r and "D" in r:
            recs.append(r)
    return recs


def _finite(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _ms(vals):
    """(mean, std) over the finite entries of ``vals``; (nan, nan) if none."""
    v = [x for x in (_finite(a) for a in vals) if x is not None]
    if not v:
        return float("nan"), float("nan")
    return float(np.mean(v)), float(np.std(v))


def _get_lead(d: dict, h: int):
    """Look up a per-lead value keyed by int OR str (JSON round-trips int keys to str)."""
    if not isinstance(d, dict):
        return None
    return d.get(h, d.get(str(h)))


# ===========================================================================
# T1 + T2: NLL + robustness by solver x D.
# ===========================================================================

def table1_nll(records, methods, D_list, out_dir: Path, structure="S0"):
    """Mean +/- std train-window NLL by solver x D, for the S0 dense operator only.

    Restricted to ``structure`` (S0, the dense LIM operator): this is the canonical
    solver-vs-solver scaling comparison. Alternate F-structures (S1/S3/S4/S5) exist only at
    the selection dimension and are reported separately in T5; pooling them into the (D,method)
    cells here would blend operators of very different skill (e.g. contaminating the D=15 cell).
    CSV + printed."""
    cell = defaultdict(list)   # (D, method) -> [nll, ...]
    for r in records:
        if r.get("structure") != structure:
            continue
        cell[(r["D"], r["method"])].append(r.get("final_nll"))
    rows = ["D,method,nll_mean,nll_std,n"]
    print("\n=== T1: train-window NLL by solver x D (mean +/- std) ===")
    print(f"{'D':>4s} | " + " | ".join(f"{m:>20s}" for m in methods))
    for D in D_list:
        line = []
        for m in methods:
            vals = cell.get((D, m), [])
            mu, sd = _ms(vals)
            line.append(f"{mu:>11.3f}+-{sd:<7.3f}" if not math.isnan(mu) else f"{'-':>20s}")
            rows.append(f"{D},{m},{mu:.6f},{sd:.6f},{len(vals)}")
        print(f"{D:>4d} | " + " | ".join(line))
    (out_dir / "T1_nll_by_solver.csv").write_text("\n".join(rows) + "\n")


def table2_robustness(records, methods, D_list, out_dir: Path, eps_rel=ROBUST_EPS_REL,
                      structure="S0"):
    """Fraction of seeds whose NLL is within ``eps_rel`` of the best NLL at each (D).

    Best NLL at a (D) is the minimum finite NLL across ALL solvers/seeds; a solver's
    robustness is the fraction of its seeds within ``eps_rel*max(1,|best|)`` of that best
    (how reliably the solver reaches the best basin). Mirrors exp_i's robustness fraction.
    Restricted to ``structure`` (S0 dense operator), matching T1: alternate F-structures live
    only at the selection D and would otherwise contaminate the per-(D,method) seed pools."""
    recs = [r for r in records if r.get("structure") == structure]
    by_D_best = {}
    for D in D_list:
        all_nll = [_finite(r.get("final_nll")) for r in recs if r["D"] == D]
        all_nll = [x for x in all_nll if x is not None]
        by_D_best[D] = min(all_nll) if all_nll else None

    seeds_by = defaultdict(list)   # (D, method) -> [nll,...]
    for r in recs:
        v = _finite(r.get("final_nll"))
        if v is not None:
            seeds_by[(r["D"], r["method"])].append(v)

    rows = ["D,method,robust_frac,n_seeds,best_nll"]
    print(f"\n=== T2: robustness (frac seeds within {eps_rel:.0%} of best NLL) by solver x D ===")
    print(f"{'D':>4s} | " + " | ".join(f"{m:>20s}" for m in methods))
    for D in D_list:
        best = by_D_best[D]
        line = []
        for m in methods:
            vals = seeds_by.get((D, m), [])
            if not vals or best is None:
                line.append(f"{'-':>20s}")
                rows.append(f"{D},{m},nan,0,{'' if best is None else best:.6f}")
                continue
            tol = eps_rel * max(1.0, abs(best))
            frac = sum(1 for v in vals if (v - best) <= tol) / len(vals)
            line.append(f"{frac:>20.2f}")
            rows.append(f"{D},{m},{frac:.4f},{len(vals)},{best:.6f}")
        print(f"{D:>4d} | " + " | ".join(line))
    (out_dir / "T2_robustness_by_solver.csv").write_text("\n".join(rows) + "\n")
    return by_D_best


# ===========================================================================
# T3: forecast skill (ACC / RMSE) by lead, fitted-LIM vs references.
# ===========================================================================

def table3_forecast(records, D_list, out_dir: Path, primary="dmci_adam",
                    structure="S0", headline_D=10):
    """Held-out Nino-3.4 ACC/RMSE by lead for fitted-LIM vs persistence/damped/Green, at the
    HEADLINE (structure, D) only: the S0 dense operator at the headline dimension, 3 seeds.
    Mixing structures (S1/S3/S4/S5) or dimensions would blend operators of very different skill,
    so we restrict to one cell. Skill is read from each record's nested ``forecast`` block, so all
    four methods are compared on the same held-out window."""
    # collect skill series: (lead, method) -> [ACC...], [RMSE...]   for the S0 headline-D fits.
    acc = defaultdict(list)
    rmse = defaultdict(list)
    for r in records:
        if (r.get("method") != primary or r.get("structure") != structure
                or int(r.get("D")) != headline_D):
            continue
        nino = (r.get("forecast") or {}).get("nino34") or {}
        for h in LEADS:
            cell = _get_lead(nino, h)
            if not cell:
                continue
            for meth in SKILL_METHODS:
                if meth in cell:
                    a = _finite(cell[meth].get("ACC"))
                    e = _finite(cell[meth].get("RMSE"))
                    if a is not None:
                        acc[(h, meth)].append(a)
                    if e is not None:
                        rmse[(h, meth)].append(e)

    rows = ["lead,method,ACC_mean,ACC_std,RMSE_mean,RMSE_std,n"]
    print(f"\n=== T3: held-out Nino-3.4 skill by lead ({primary}) -- fitted-LIM vs references ===")
    print(f"{'lead':>5s} | " + " | ".join(f"{m:>20s}" for m in SKILL_METHODS))
    print("  -- ACC (higher better) --")
    for h in LEADS:
        line = []
        for meth in SKILL_METHODS:
            amu, asd = _ms(acc.get((h, meth), []))
            rmu, rsd = _ms(rmse.get((h, meth), []))
            line.append(f"{amu:>20.3f}" if not math.isnan(amu) else f"{'-':>20s}")
            rows.append(f"{h},{meth},{amu:.6f},{asd:.6f},{rmu:.6f},{rsd:.6f},"
                        f"{len(acc.get((h, meth), []))}")
        print(f"{h:>5d} | " + " | ".join(line))
    print("  -- RMSE (lower better) --")
    for h in LEADS:
        line = []
        for meth in SKILL_METHODS:
            rmu, _ = _ms(rmse.get((h, meth), []))
            line.append(f"{rmu:>20.3f}" if not math.isnan(rmu) else f"{'-':>20s}")
        print(f"{h:>5d} | " + " | ".join(line))
    (out_dir / "T3_forecast_skill.csv").write_text("\n".join(rows) + "\n")

    # pgfplots .dat: ACC vs lead, one column per method (S0 dense, headline D, 3-seed mean).
    lines = ["lead " + " ".join(SKILL_METHODS)]
    for h in LEADS:
        cells = []
        for meth in SKILL_METHODS:
            amu, _ = _ms(acc.get((h, meth), []))
            cells.append(f"{amu:.6e}" if not math.isnan(amu) else "nan")
        lines.append(f"{h} " + " ".join(cells))
    (out_dir / "forecast_acc_by_lead.dat").write_text("\n".join(lines) + "\n")
    return acc, rmse


# ===========================================================================
# T4: scaling by D (NLL, ENSO timescales, ACC, per_step_ms, min_detS).
# ===========================================================================

def table4_scaling(records, D_list, out_dir: Path, primary="dmci_adam",
                   headline_lead=HEADLINE_LEAD, structure="S0"):
    """NLL / ENSO timescales / held-out ACC / per_step_ms / min_detS vs D for the PRIMARY solver.

    The scaling story (mirrors exp_i d-scaling): how the DMCI exact-gradient fit's likelihood,
    recovered ENSO mode, forecast skill, interpreter cost, and PD-margin evolve with the state
    dimension D. Restricted to ``structure`` (S0 dense operator) so the D=15 cell is the dense
    operator, not a blend with the S1/S3/S4/S5 selection runs. CSV + pgfplots .dat + printed."""
    agg = {}
    for D in D_list:
        cell = [r for r in records if r["D"] == D and r.get("method") == primary
                and r.get("structure") == structure]
        if not cell:
            continue
        nll_mu, nll_sd = _ms([r.get("final_nll") for r in cell])
        per_mu, _ = _ms([r.get("per_step_ms") for r in cell])
        detS_mu, _ = _ms([r.get("min_detS") for r in cell])
        condS_mu, _ = _ms([r.get("cond_S") for r in cell])
        rho_mu, _ = _ms([r.get("rho_F") for r in cell])
        period_mu, _ = _ms([r.get("enso_period_mo") for r in cell])
        decay_mu, _ = _ms([r.get("enso_decay_mo") for r in cell])
        acc_mu, _ = _ms([_get_lead(r.get("heldout_acc") or {}, headline_lead) for r in cell])
        agg[D] = {
            "nll_mean": nll_mu, "nll_std": nll_sd,
            "per_step_ms": per_mu, "min_detS": detS_mu, "cond_S": condS_mu,
            "rho_F": rho_mu, "enso_period_mo": period_mu, "enso_decay_mo": decay_mu,
            f"heldout_acc_lead{headline_lead}": acc_mu, "n_seeds": len(cell),
        }

    cols = ["nll_mean", f"heldout_acc_lead{headline_lead}", "enso_period_mo",
            "enso_decay_mo", "rho_F", "per_step_ms", "min_detS", "cond_S"]
    rows = ["D," + ",".join(cols) + ",n_seeds"]
    print(f"\n=== T4: scaling by D ({primary}) ===")
    print(f"{'D':>4s} | " + " | ".join(f"{c:>16s}" for c in cols))
    for D in D_list:
        if D not in agg:
            continue
        a = agg[D]
        print(f"{D:>4d} | " + " | ".join(f"{a[c]:>16.4g}" for c in cols))
        rows.append(f"{D}," + ",".join(f"{a[c]:.6g}" for c in cols) + f",{a['n_seeds']}")
    (out_dir / "T4_scaling_by_D.csv").write_text("\n".join(rows) + "\n")

    # pgfplots .dat: the headline scaling curves vs D.
    lines = ["D " + " ".join(cols)]
    for D in D_list:
        if D not in agg:
            continue
        a = agg[D]
        lines.append(f"{D} " + " ".join(f"{a[c]:.6e}" for c in cols))
    (out_dir / "scaling_by_D.dat").write_text("\n".join(lines) + "\n")
    return agg


# ===========================================================================
# Portfolio + go/no-go summary.
# ===========================================================================

def table5_structure_selection(records, structures, D_list, out_dir: Path,
                               primary="dmci_adam", headline_lead=HEADLINE_LEAD):
    """T5: model selection across F-structures (the program-as-data demo). Per (structure, D),
    the mean train NLL, the free-parameter count k (params.param_count), AIC = 2k + 2*NLL,
    BIC = k*ln(N) + 2*NLL with N = T_train, and held-out Nino-3.4 ACC at the headline lead.
    The ``recompiled_engine`` column is ``no`` for EVERY row: the interpreter, the autograd
    path, and the MLE driver are byte-identical across structures, only the bound F-assembly
    changes (the headline of the experiment). Lower AIC/BIC is better; higher ACC is better."""
    N = DEFAULT.T_train
    rows = ["D,structure,k,nll_mean,aic,bic,heldout_acc,recompiled_engine"]
    by_D = defaultdict(list)
    for D in D_list:
        for structure in structures:
            cell = [r for r in records
                    if r.get("structure") == structure and int(r.get("D")) == D
                    and r.get("method") == primary]
            if not cell:
                continue
            nll_mu, _ = _ms([r.get("final_nll") for r in cell])
            acc_mu, _ = _ms([_get_lead(r.get("heldout_acc") or {}, headline_lead) for r in cell])
            try:
                k = int(params.param_count(D, structure, DEFAULT.lowrank_rank)["total"])
            except Exception:  # noqa: BLE001  (unknown structure -> skip k-based scores)
                k = None
            aic = (2 * k + 2 * nll_mu) if (k is not None and math.isfinite(nll_mu)) else float("nan")
            bic = (k * math.log(N) + 2 * nll_mu) if (k is not None and math.isfinite(nll_mu)) else float("nan")
            rows.append(f"{D},{structure},{k if k is not None else ''},{nll_mu:.4f},"
                        f"{aic:.4f},{bic:.4f},{acc_mu:.6f},no")
            by_D[D].append((structure, aic, bic, acc_mu, k))
    (out_dir / "T5_structure_selection.csv").write_text("\n".join(rows) + "\n")
    for D in sorted(by_D):
        cand = [c for c in by_D[D] if math.isfinite(c[1])]
        if not cand:
            continue
        best_aic = min(cand, key=lambda c: c[1])
        best_bic = min(cand, key=lambda c: c[2])
        accs = [c for c in by_D[D] if c[3] == c[3]]   # finite ACC
        best_acc = max(accs, key=lambda c: c[3]) if accs else (None,)
        print(f"        D={D} structure-selection: AIC->{best_aic[0]} (k={best_aic[4]}, "
              f"aic={best_aic[1]:.1f}); BIC->{best_bic[0]}; held-out ACC->{best_acc[0]}"
              + (f" (acc={best_acc[3]:.3f})" if accs else ""))
    return by_D


def _portfolio(records, structures, D_list, lead):
    """Per (structure, D): winning solver on held-out Nino-3.4 ACC @ lead (NOT train NLL)."""
    out = {}
    for structure in structures:
        for D in D_list:
            cell = [r for r in records
                    if r.get("structure") == structure and r.get("D") == D]
            if not cell:
                continue
            by_method = defaultdict(list)
            for r in cell:
                v = _get_lead(r.get("heldout_acc") or {}, lead)
                v = _finite(v)
                if v is not None:
                    by_method[r["method"]].append(v)
            mean_acc = {m: float(np.mean(v)) for m, v in by_method.items() if v}
            if not mean_acc:
                continue
            winner = max(mean_acc, key=mean_acc.get)
            out[f"{structure}_D{D}"] = {"winner": winner, "val_acc": mean_acc[winner],
                                        "by_method": mean_acc}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate LIM-ENSO per-run records")
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument("--primary", default="dmci_adam",
                    help="solver whose forecast-skill / scaling curves are the headline")
    ap.add_argument("--headline-lead", type=int, default=HEADLINE_LEAD)
    args = ap.parse_args()

    records = _load_records(args.results_dir)
    if not records:
        print(f"No per-run records in {args.results_dir}/ (results/*.json). "
              f"Run run_all first.")
        return

    out_dir = args.results_dir / "agg"
    out_dir.mkdir(parents=True, exist_ok=True)

    methods = sorted({r["method"] for r in records},
                     key=lambda m: (m != "dmci_adam", m))   # primary first
    D_list = sorted({int(r["D"]) for r in records})
    structures = sorted({r.get("structure", "S0") for r in records})

    print(f"[aggregate] {len(records)} records  methods={methods}  D={D_list}  "
          f"structures={structures}")

    table1_nll(records, methods, D_list, out_dir)
    by_D_best = table2_robustness(records, methods, D_list, out_dir)
    table3_forecast(records, D_list, out_dir, primary=args.primary)
    scaling = table4_scaling(records, D_list, out_dir, primary=args.primary,
                             headline_lead=args.headline_lead)
    table5_structure_selection(records, structures, D_list, out_dir,
                               primary=args.primary, headline_lead=args.headline_lead)
    portfolio = _portfolio(records, structures, D_list, args.headline_lead)

    # --- go/no-go: a (structure,D) is GO when the held-out skill beats persistence and the
    #     PD margin holds at the fitted operator (min_detS > floor). Reported, never aborts. ---
    go_no_go = {}
    for structure in structures:
        for D in D_list:
            cell = [r for r in records
                    if r.get("structure") == structure and r.get("D") == D
                    and r.get("method") == args.primary]
            if not cell:
                continue
            beats_persist = []
            pd_ok = []
            for r in cell:
                nino = (r.get("forecast") or {}).get("nino34") or {}
                c = _get_lead(nino, args.headline_lead)
                if c and "fitted_lim" in c and "persistence" in c:
                    a_fit = _finite(c["fitted_lim"].get("ACC"))
                    a_per = _finite(c["persistence"].get("ACC"))
                    if a_fit is not None and a_per is not None:
                        beats_persist.append(a_fit >= a_per)
                d = _finite(r.get("min_detS"))
                pd_ok.append(d is not None and d > DEFAULT.jitter_eps * 0)  # finite & >0
            go_no_go[f"{structure}_D{D}"] = {
                "beats_persistence_frac": (float(np.mean(beats_persist))
                                           if beats_persist else None),
                "pd_ok_frac": float(np.mean(pd_ok)) if pd_ok else None,
                "GO": bool(beats_persist and np.mean(beats_persist) > 0.5
                           and pd_ok and np.mean(pd_ok) > 0.5),
            }

    summary = {
        "n_records": len(records),
        "methods": methods, "D_list": D_list, "structures": structures,
        "primary": args.primary, "headline_lead": args.headline_lead,
        "best_nll_by_D": {str(k): v for k, v in by_D_best.items()},
        "scaling_by_D": {str(k): v for k, v in scaling.items()},
        "portfolio_winner_by_structure_D": portfolio,
        "go_no_go_by_structure_D": go_no_go,
    }
    (out_dir / "scaling_gate_summary.json").write_text(json.dumps(summary, indent=2,
                                                                  default=str))

    print(f"\n=== portfolio winner (held-out Nino-3.4 ACC @ {args.headline_lead}mo) ===")
    for key in sorted(portfolio):
        w = portfolio[key]
        print(f"  {key:>12s}: {w['winner']:>18s} (acc={w['val_acc']:.3f})")
    print(f"\nWrote tables + .dat + scaling_gate_summary.json to {out_dir}/")


if __name__ == "__main__":
    main()
