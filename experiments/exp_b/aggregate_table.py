############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# aggregate_table.py: Aggregate Experiment B per-run results into Table 6 and verify manuscript values. Canonical source of truth:...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Aggregate Experiment B per-run results into Table 6 and verify manuscript values.

Canonical source of truth: experiments/exp_b/results/{method}_{model}_{seed:02d}.json
Each file is emitted by runner.py and carries: converged, convergence_epoch,
final_loss, total_wall_time (plus param errors / tier / domain).

This script reconstructs every cell of Table 6 (tab:exp_b_results) -- including the
Avg Epoch column that exp_b_summary.json does not store -- plus the aggregate
wall-clock figures, the cross-method convergence-epoch-match counts, and the
mean-final-loss coordinates of Figure fig:exp_b_loss (fig_b_loss_comparison.tex),
and checks them against the values printed in the manuscript so future drift is caught.

Usage:
    python3 -m experiments.exp_b.aggregate_table                 # print table + all self-checks
    python3 -m experiments.exp_b.aggregate_table --write-summary # refresh exp_b_summary.json (now WITH epochs)
    python3 -m experiments.exp_b.aggregate_table --emit-fig-coords  # regenerate Figure coordinate blocks

Data provenance / reproducibility:
- The committed per-run JSONs (final_loss, convergence_epoch, total_wall_time) are
  sufficient to regenerate every committed Table 6 cell AND every Figure fig:exp_b_loss
  coordinate. No external .dat file is needed; the figure values are verified here.
- runner.py also writes a per-run CSV (full per-epoch loss/grad-norm/wall-time/param
  trajectory) next to each JSON. Those CSVs (~17 MB) are NOT referenced by any committed
  table or figure and are therefore not tracked; the raw trajectories live on the cluster
  (HPC: ~/src/nncompile/experiments/exp_b/results/*.csv) and are regenerable via run_all.

If the per-run results are missing locally, regenerate them on the cluster with:
    python3 -m experiments.exp_b.run_all --use-llm-cache --output-dir experiments/exp_b/results_llm
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev

from .models import ALL_MODELS, MODEL_BY_NAME

RESULTS_DIR = Path(__file__).parent / "results_llm"
SUMMARY_PATH = (Path(__file__).parent.parent.parent
                / "paper2" / "figures" / "data" / "exp_b_summary.json")
METHODS = ["dmci", "direct_compiled", "handcoded_pytorch", "pure_mlp"]

# Display label used in Table 6 (paper) keyed by model id.
LABEL = {
    "M01_coulomb": "M01 Coulomb", "M02_beer_lambert": "M02 Beer-Lambert",
    "M03_michaelis_menten": "M03 Michaelis-Menten", "M04_arrhenius": "M04 Arrhenius",
    "M05_hookes_spring": "M05 Damped oscillator", "M06_logistic_growth": "M06 Logistic growth",
    "M07_power_law": "M07 Power law", "M08_euler_ode": "M08 Euler ODE",
    "M09_taylor_exp": "M09 Taylor e^ax", "M10_smooth_activation": "M10 SiLU",
    "M11_recursive_filter": "M11 Recursive filter", "M12_newton_sqrt": "M12 Newton sqrt",
    "M13_composed_transforms": "M13 Composed transforms", "M14_anomaly_scorer": "M14 Anomaly scorer",
    "M15_horner_eval": "M15 Horner polynomial",
}

# Values printed in paper-dmci/paper2.tex Table 6 (tab:exp_b_results), for drift detection.
# (avg_epoch, avg_loss_4dp, avg_time_s, mlp_conv). Convergence epochs and final losses are
# identical between the reference run and the LLM re-run; only the wall times and M05's MLP
# convergence count differ -- these reflect the committed results_llm/ that the paper reports.
PAPER_TABLE6 = {
    "M01_coulomb": (556, 0.0000, 222, 0), "M02_beer_lambert": (66, 0.0016, 21, 5),
    "M03_michaelis_menten": (1753, 0.0007, 378, 5), "M04_arrhenius": (308, 0.0001, 130, 5),
    "M05_hookes_spring": (135, 0.0000, 103, 1), "M06_logistic_growth": (388, 0.0001, 231, 5),
    "M07_power_law": (752, 0.0003, 232, 4), "M08_euler_ode": (23, 0.0030, 241, 5),
    "M09_taylor_exp": (165, 0.4803, 3127, 5), "M10_smooth_activation": (248, 0.0004, 160, 5),
    "M11_recursive_filter": (17, 0.0002, 257, 5), "M12_newton_sqrt": (0, 0.0000, 203, 5),
    "M13_composed_transforms": (356, 0.0003, 215, 5), "M14_anomaly_scorer": (100, 0.0002, 182, 5),
    "M15_horner_eval": (2128, 0.0010, 1992, 5),
}
# Aggregate prose claims (paper-dmci/paper2.tex line ~439).
PAPER_AGG = {"dmci_time": 513, "direct_time": 7.0, "handcoded_time": 2.1,
             "ratio_direct": 73, "ratio_handcoded": 248, "mlp_conv_total": 65}

# Mean-final-loss values hardcoded as pgfplots coordinates in
# paper-dmci/figures/fig_b_loss_comparison.tex (fig:exp_b_loss), per method, in MODEL order.
# Verified against the per-run results (rounded to 3 significant figures in the figure).
PAPER_FIG_LOSS = {
    "handcoded_pytorch": [4.37e-05, 1.64e-03, 6.67e-04, 8.69e-05, 5.00e-06, 1.48e-04, 3.40e-04,
                          3.04e-03, 4.80e-01, 4.20e-04, 1.34e-04, 1.48e-12, 2.80e-04, 2.04e-04, 9.73e-04],
    "direct_compiled":   [4.39e-05, 1.64e-03, 6.67e-04, 8.69e-05, 5.00e-06, 1.48e-04, 3.40e-04,
                          3.04e-03, 4.80e-01, 4.20e-04, 1.52e-04, 1.48e-12, 2.80e-04, 2.04e-04, 9.73e-04],
    "dmci":              [4.39e-05, 1.64e-03, 6.67e-04, 8.69e-05, 5.00e-06, 1.48e-04, 3.40e-04,
                          3.04e-03, 4.80e-01, 4.20e-04, 1.52e-04, 1.48e-12, 2.80e-04, 2.04e-04, 9.73e-04],
    "pure_mlp":          [2.37e+02, 8.25e-02, 1.80e-01, 5.60e-03, 2.23e-01, 3.70e-01, 2.82e-01,
                          5.35e-04, 1.10e+00, 2.77e-04, 6.71e-04, 4.93e-02, 7.14e-02, 8.29e-01, 5.39e-04],
}


def load_runs():
    """runs[method][model] -> list of per-seed dicts (sorted by seed)."""
    runs: dict[str, dict[str, list]] = {m: {} for m in METHODS}
    files = sorted(RESULTS_DIR.glob("*.json"))
    if not files:
        raise SystemExit(
            f"No per-run results in {RESULTS_DIR}. Regenerate with:\n"
            "  python3 -m experiments.exp_b.run_all --use-llm-cache --output-dir experiments/exp_b/results_llm")
    for f in files:
        d = json.loads(f.read_text())
        runs.setdefault(d["method"], {}).setdefault(d["model_name"], []).append(d)
    for m in runs:
        for mod in runs[m]:
            runs[m][mod].sort(key=lambda r: r["seed"])
    return runs


def agg(runs, method, model):
    rs = runs[method][model]
    conv = [r for r in rs if r["converged"]]
    epochs = [r["convergence_epoch"] for r in conv if r["convergence_epoch"] is not None]
    return {
        "n_seeds": len(rs), "converged": len(conv),
        "mean_epoch": mean(epochs) if epochs else None,
        "mean_final_loss": mean(r["final_loss"] for r in rs),
        "mean_wall_time": mean(r["total_wall_time"] for r in rs),
        "final_losses": [r["final_loss"] for r in rs],
        "convergence_epochs": [r["convergence_epoch"] for r in rs],
    }


def build_summary(runs):
    out = {}
    for method in METHODS:
        for model in sorted(runs[method]):
            rs = runs[method][model]
            a = agg(runs, method, model)
            out[f"{method}_{model}"] = {
                "method": method, "model": model, "n_seeds": a["n_seeds"],
                "converged": a["converged"],
                "convergence_epochs": a["convergence_epochs"],
                "mean_epoch": a["mean_epoch"],
                "final_losses": a["final_losses"],
                "mean_final_loss": a["mean_final_loss"],
                "std_final_loss": pstdev(a["final_losses"]) if len(a["final_losses"]) > 1 else 0.0,
                "wall_times": [r["total_wall_time"] for r in rs],
                "mean_wall_time": a["mean_wall_time"],
                "std_wall_time": pstdev([r["total_wall_time"] for r in rs]) if len(rs) > 1 else 0.0,
            }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-summary", action="store_true",
                    help=f"(re)write {SUMMARY_PATH} including the epoch fields")
    ap.add_argument("--emit-fig-coords", action="store_true",
                    help="print pgfplots coordinate blocks for fig_b_loss_comparison from the data")
    args = ap.parse_args()

    runs = load_runs()
    order = [m.name for m in ALL_MODELS]
    checks = []

    print(f"{'Model':24} {'P':>2} {'AvgEpoch':>8} {'AvgLoss':>9} {'AvgTime':>8} {'MLPConv':>7}")
    print("-" * 64)
    for model in order:
        d = agg(runs, "dmci", model)
        mlp_conv = agg(runs, "pure_mlp", model)["converged"]
        params = len(MODEL_BY_NAME[model].param_names)
        epoch = round(d["mean_epoch"])
        loss = round(d["mean_final_loss"], 4)
        t = round(d["mean_wall_time"])
        print(f"{LABEL[model]:24} {params:>2} {epoch:>8} {loss:>9.4f} {t:>8} {mlp_conv:>5}/5")
        pe, pl, pt, pm = PAPER_TABLE6[model]
        checks.append((f"{model} epoch", epoch, pe))
        checks.append((f"{model} loss", loss, pl))
        # wall times carry sub-second rounding noise; accept within 1 s of the printed value
        checks.append((f"{model} time", pt if abs(t - pt) <= 1 else t, pt))
        checks.append((f"{model} mlpconv", mlp_conv, pm))

    # ---- aggregates ----
    def method_mean_time(meth):
        return mean(agg(runs, meth, m)["mean_wall_time"] for m in order)
    dmci_t, direct_t, hand_t = (method_mean_time(m) for m in
                                ["dmci", "direct_compiled", "handcoded_pytorch"])
    mlp_total = sum(agg(runs, "pure_mlp", m)["converged"] for m in order)
    print(f"\nDMCI mean time={dmci_t:.2f}s  direct={direct_t:.2f}s  handcoded={hand_t:.2f}s")
    print(f"DMCI/direct={dmci_t/direct_t:.1f}x  DMCI/handcoded={dmci_t/hand_t:.1f}x  "
          f"MLP converged={mlp_total}/75")
    checks += [("agg dmci_time", round(dmci_t), PAPER_AGG["dmci_time"]),
               ("agg direct_time", round(direct_t, 1), PAPER_AGG["direct_time"]),
               ("agg ratio_direct", round(dmci_t/direct_t), PAPER_AGG["ratio_direct"]),
               ("agg ratio_handcoded", round(dmci_t/hand_t), PAPER_AGG["ratio_handcoded"]),
               ("agg mlp_conv_total", mlp_total, PAPER_AGG["mlp_conv_total"])]

    # ---- convergence-epoch match across methods (75 model-seed pairs) ----
    def epoch_match(other):
        match, mism = 0, []
        for model in order:
            for s in range(5):
                de = runs["dmci"][model][s]["convergence_epoch"]
                oe = runs[other][model][s]["convergence_epoch"]
                if de == oe:
                    match += 1
                else:
                    mism.append((model, s, de, oe))
        return match, mism
    dm, dmis = epoch_match("direct_compiled")
    hm, hmis = epoch_match("handcoded_pytorch")
    print(f"\nConvergence-epoch match (of 75 pairs): DMCI=direct {dm}/75 ; DMCI=handcoded {hm}/75")
    if hmis:
        print("  handcoded epoch mismatches:",
              [(LABEL[m].split()[0], s, de, oe) for m, s, de, oe in hmis])
    # final-loss identity dmci vs direct
    max_loss_diff = max(abs(runs["dmci"][m][s]["final_loss"]
                            - runs["direct_compiled"][m][s]["final_loss"])
                        for m in order for s in range(5))
    print(f"DMCI vs direct max |final_loss diff| over 75 pairs = {max_loss_diff:.2e}")

    # ---- figure (fig:exp_b_loss) coordinate verification ----
    # The figure hardcodes mean final loss per method, rounded to 3 sig figs.
    fig_match, fig_total = 0, 0
    for meth, vals in PAPER_FIG_LOSS.items():
        for i, model in enumerate(order):
            got = agg(runs, meth, model)["mean_final_loss"]
            want = vals[i]
            fig_total += 1
            if want != 0 and abs(got - want) / want <= 0.02:
                fig_match += 1
            elif want == 0 and got == 0:
                fig_match += 1
            else:
                print(f"  FIG MISMATCH {meth:18} {model:22} fig={want:.3e} data={got:.3e}")
    print(f"FIGURE self-check (fig:exp_b_loss): {fig_match}/{fig_total} coordinates match (<=2%)")

    if args.emit_fig_coords:
        short = [m.split("_")[0] for m in order]  # M01..M15
        print("\n% pgfplots coordinates regenerated from per-run results:")
        for meth in ["handcoded_pytorch", "direct_compiled", "dmci", "pure_mlp"]:
            coords = " ".join(f"({short[i]}, {agg(runs, meth, m)['mean_final_loss']:.2e})"
                              for i, m in enumerate(order))
            print(f"% {meth}\n        {coords}")

    # ---- self-check ----
    fails = [(n, got, exp) for n, got, exp in checks if got != exp]
    print(f"\nSELF-CHECK vs manuscript: {len(checks)-len(fails)}/{len(checks)} cells match")
    for n, got, exp in fails:
        print(f"  MISMATCH {n}: computed={got} paper={exp}")

    if args.write_summary:
        SUMMARY_PATH.write_text(json.dumps(build_summary(runs), indent=0))
        print(f"\nWrote {SUMMARY_PATH} ({SUMMARY_PATH.stat().st_size} bytes, now includes epoch fields)")

    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
