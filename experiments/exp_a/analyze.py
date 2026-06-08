############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# analyze.py: Statistical analysis of Experiment A results. Loads per-run CSVs, computes aggregate statistics, and generates...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Statistical analysis of Experiment A results.

Loads per-run CSVs, computes aggregate statistics, and generates LaTeX tables.
"""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import ExpAConfig, DEFAULT
from .programs import ALL_PROGRAMS, PROGRAMS_BY_NAME


def load_summaries(results_dir: str) -> list[dict]:
    summaries = []
    for f in sorted(Path(results_dir).glob("*.json")):
        if f.name == "ablations.json":
            continue
        with open(f) as fh:
            summaries.append(json.load(fh))
    return summaries


def load_loss_histories(results_dir: str) -> dict[tuple, list[float]]:
    """Load per-epoch loss for each (method, program, seed)."""
    histories = {}
    for f in sorted(Path(results_dir).glob("*.csv")):
        parts = f.stem.rsplit("_", 1)
        seed = int(parts[-1])
        method_prog = parts[0]
        # Parse method and program from filename
        for prog in [p.name for p in ALL_PROGRAMS]:
            if method_prog.endswith(prog):
                method = method_prog[: -len(prog) - 1]
                with open(f) as fh:
                    reader = csv.DictReader(fh)
                    losses = [float(row["loss"]) for row in reader]
                histories[(method, prog, seed)] = losses
                break
    return histories


@dataclass
class AggStats:
    method: str
    program: str
    n_seeds: int
    n_converged: int
    convergence_rate: float
    mean_conv_epoch: float
    std_conv_epoch: float
    median_conv_epoch: float
    mean_final_loss: float
    std_final_loss: float
    mean_wall_time: float
    std_wall_time: float
    mean_param_errors: dict[str, float]
    std_param_errors: dict[str, float]


def compute_stats(summaries: list[dict]) -> list[AggStats]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for s in summaries:
        groups[(s["method"], s["program"])].append(s)

    stats = []
    for (method, program), runs in sorted(groups.items()):
        conv_epochs = [r["convergence_epoch"] for r in runs
                       if r["converged"]]
        final_losses = [r["final_loss"] for r in runs]
        wall_times = [r["total_wall_time"] for r in runs]

        # Aggregate param errors
        param_names = list(runs[0]["final_param_errors"].keys())
        param_errs = {
            pn: [r["final_param_errors"][pn] for r in runs]
            for pn in param_names
        }

        stats.append(AggStats(
            method=method,
            program=program,
            n_seeds=len(runs),
            n_converged=len(conv_epochs),
            convergence_rate=len(conv_epochs) / len(runs),
            mean_conv_epoch=float(np.mean(conv_epochs)) if conv_epochs else float("nan"),
            std_conv_epoch=float(np.std(conv_epochs)) if conv_epochs else float("nan"),
            median_conv_epoch=float(np.median(conv_epochs)) if conv_epochs else float("nan"),
            mean_final_loss=float(np.mean(final_losses)),
            std_final_loss=float(np.std(final_losses)),
            mean_wall_time=float(np.mean(wall_times)),
            std_wall_time=float(np.std(wall_times)),
            mean_param_errors={pn: float(np.mean(param_errs[pn]))
                               for pn in param_names},
            std_param_errors={pn: float(np.std(param_errs[pn]))
                              for pn in param_names},
        ))

    return stats


def mean_loss_curves(histories: dict[tuple, list[float]],
                     max_epochs: int | None = None) -> dict[tuple, tuple]:
    """Compute mean and std loss curves per (method, program).

    Returns dict of (method, program) -> (mean_array, std_array).
    """
    groups: dict[tuple, list[list[float]]] = defaultdict(list)
    for (method, program, seed), losses in histories.items():
        groups[(method, program)].append(losses)

    curves = {}
    for key, all_losses in groups.items():
        min_len = min(len(l) for l in all_losses)
        if max_epochs is not None:
            min_len = min(min_len, max_epochs)
        arr = np.array([l[:min_len] for l in all_losses])
        curves[key] = (arr.mean(axis=0), arr.std(axis=0))

    return curves


def generate_convergence_table(stats: list[AggStats]) -> str:
    """Generate LaTeX table: epochs to convergence per method x program."""
    methods_order = ["direct", "compiled_interp", "handcoded_interp",
                     "finite_diff", "evolution_strategy"]
    method_labels = {
        "direct": "Direct",
        "compiled_interp": "Compiled Interp.",
        "handcoded_interp": "Hand-coded Interp.",
        "finite_diff": "Finite Diff.",
        "evolution_strategy": "Evol. Strategy",
    }
    programs_order = [p.name for p in ALL_PROGRAMS]
    prog_labels = {p.name: p.name.split("_", 1)[0] for p in ALL_PROGRAMS}

    stats_map = {(s.method, s.program): s for s in stats}

    lines = []
    lines.append(r"\begin{tabular}{l" + "c" * len(methods_order) + "}")
    lines.append(r"\toprule")
    header = "Program & " + " & ".join(method_labels[m] for m in methods_order) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    for prog in programs_order:
        row = [prog_labels[prog]]
        for method in methods_order:
            s = stats_map.get((method, prog))
            if s is None:
                row.append("--")
            elif s.n_converged == 0:
                row.append("--")
            else:
                row.append(f"${s.mean_conv_epoch:.0f} \\pm {s.std_conv_epoch:.0f}$")
        lines.append(" & ".join(row) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def generate_param_error_table(stats: list[AggStats]) -> str:
    """Generate LaTeX table: final parameter errors per method x program."""
    methods_order = ["direct", "compiled_interp", "handcoded_interp"]
    method_labels = {
        "direct": "Direct",
        "compiled_interp": "Compiled Interp.",
        "handcoded_interp": "Hand-coded Interp.",
    }
    programs_order = [p.name for p in ALL_PROGRAMS]

    stats_map = {(s.method, s.program): s for s in stats}

    lines = []
    lines.append(r"\begin{tabular}{llc" + "c" * len(methods_order) + "}")
    lines.append(r"\toprule")
    header = "Program & Param & Target & " + " & ".join(
        method_labels[m] for m in methods_order) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    for prog in programs_order:
        spec = PROGRAMS_BY_NAME[prog]
        for j, pn in enumerate(spec.param_names):
            label = spec.name.split("_", 1)[0] if j == 0 else ""
            target = spec.target_values[pn]
            row = [label, f"${pn}$", f"${target:.1f}$"]
            for method in methods_order:
                s = stats_map.get((method, prog))
                if s is None:
                    row.append("--")
                else:
                    me = s.mean_param_errors[pn]
                    se = s.std_param_errors[pn]
                    row.append(f"${me:.4f} \\pm {se:.4f}$")
            lines.append(" & ".join(row) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def print_summary(stats: list[AggStats]):
    print(f"\n{'Method':20s} {'Program':20s} {'Conv':>5s} {'Epoch':>10s} "
          f"{'Final Loss':>12s} {'Time (s)':>10s}")
    print("-" * 80)
    for s in stats:
        conv = f"{s.n_converged}/{s.n_seeds}"
        epoch = f"{s.mean_conv_epoch:.0f}+/-{s.std_conv_epoch:.0f}" if s.n_converged else "--"
        loss = f"{s.mean_final_loss:.2e}+/-{s.std_final_loss:.2e}"
        wt = f"{s.mean_wall_time:.1f}+/-{s.std_wall_time:.1f}"
        print(f"{s.method:20s} {s.program:20s} {conv:>5s} {epoch:>10s} "
              f"{loss:>12s} {wt:>10s}")


def main(results_dir: str = DEFAULT.output_dir):
    print(f"Loading results from {results_dir}...")
    summaries = load_summaries(results_dir)
    if not summaries:
        print("No results found.")
        return

    print(f"Loaded {len(summaries)} runs.")
    stats = compute_stats(summaries)
    print_summary(stats)

    output_dir = str(Path(results_dir) / "tables")
    os.makedirs(output_dir, exist_ok=True)

    table1 = generate_convergence_table(stats)
    with open(os.path.join(output_dir, "table_convergence.tex"), "w") as f:
        f.write(table1)
    print(f"\nConvergence table saved to {output_dir}/table_convergence.tex")

    table2 = generate_param_error_table(stats)
    with open(os.path.join(output_dir, "table_param_errors.tex"), "w") as f:
        f.write(table2)
    print(f"Param error table saved to {output_dir}/table_param_errors.tex")


if __name__ == "__main__":
    import sys
    results_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT.output_dir
    main(results_dir)
