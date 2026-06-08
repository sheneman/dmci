############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# plot.py: Paper-ready figures for Experiment A.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Paper-ready figures for Experiment A."""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import ExpAConfig, DEFAULT
from .programs import ALL_PROGRAMS, PROGRAMS_BY_NAME
from .analyze import (
    load_summaries, load_loss_histories, compute_stats,
    mean_loss_curves, AggStats,
)

# Paper-ready styling
plt.rcParams.update({
    "font.size": 10,
    "font.family": "serif",
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "lines.linewidth": 1.5,
})

METHOD_STYLES = {
    "direct":            {"color": "#2196F3", "label": "Direct compilation", "ls": "-"},
    "compiled_interp":   {"color": "#F44336", "label": "Compiled interpreter", "ls": "-"},
    "handcoded_interp":  {"color": "#4CAF50", "label": "Hand-coded interpreter", "ls": "--"},
    "finite_diff":       {"color": "#FF9800", "label": "Finite differences", "ls": "-."},
    "evolution_strategy": {"color": "#9C27B0", "label": "Evolution strategy", "ls": ":"},
}

PROGRAM_LABELS = {
    "P1_single_const": "P1: Single constant",
    "P2_multi_const": "P2: Multi constant",
    "P3_recursive": "P3: Recursive",
    "P4_higher_order": "P4: Higher-order",
    "P5_multi_function": "P5: Multi-function",
    "P6_composed": "P6: Composed",
}


def plot_convergence_curves(results_dir: str, output_dir: str,
                            max_epochs: int | None = None):
    """Figure 1: Loss vs epoch, 2x3 grid, all methods per subplot."""
    histories = load_loss_histories(results_dir)
    curves = mean_loss_curves(histories, max_epochs=max_epochs)

    programs = [p.name for p in ALL_PROGRAMS]
    methods = list(METHOD_STYLES.keys())

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for i, prog in enumerate(programs):
        ax = axes[i]
        for method in methods:
            key = (method, prog)
            if key not in curves:
                continue
            mean, std = curves[key]
            style = METHOD_STYLES[method]
            epochs = np.arange(len(mean))
            ax.semilogy(epochs, mean, color=style["color"],
                        ls=style["ls"], label=style["label"])
            ax.fill_between(epochs, np.maximum(mean - std, 1e-10),
                            mean + std, color=style["color"], alpha=0.15)

        ax.set_title(PROGRAM_LABELS.get(prog, prog))
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (MSE)")
        ax.set_ylim(bottom=1e-8)
        if i == 0:
            ax.legend(loc="upper right", framealpha=0.9)

    fig.suptitle("Experiment A: Convergence Curves", fontsize=14, y=1.02)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "fig1_convergence.pdf"))
    fig.savefig(os.path.join(output_dir, "fig1_convergence.png"))
    plt.close(fig)
    print(f"Figure 1 saved to {output_dir}/fig1_convergence.pdf")


def plot_param_recovery(results_dir: str, output_dir: str):
    """Figure 2: Final parameter error bar chart (autograd methods only)."""
    summaries = load_summaries(results_dir)
    stats = compute_stats(summaries)
    stats_map = {(s.method, s.program): s for s in stats}

    methods = ["direct", "compiled_interp", "handcoded_interp"]
    programs = [p.name for p in ALL_PROGRAMS]

    # Collect all (program, param) pairs
    prog_params = []
    for prog in programs:
        spec = PROGRAMS_BY_NAME[prog]
        for pn in spec.param_names:
            prog_params.append((prog, pn))

    x = np.arange(len(prog_params))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 5))
    for j, method in enumerate(methods):
        means = []
        stds = []
        for prog, pn in prog_params:
            s = stats_map.get((method, prog))
            if s:
                means.append(s.mean_param_errors[pn])
                stds.append(s.std_param_errors[pn])
            else:
                means.append(0)
                stds.append(0)
        style = METHOD_STYLES[method]
        ax.bar(x + j * width, means, width, yerr=stds,
               color=style["color"], label=style["label"],
               capsize=3, alpha=0.85)

    labels = [f"{PROGRAM_LABELS[prog].split(':')[0]}: {pn}"
              for prog, pn in prog_params]
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Parameter Error |learned - target|")
    ax.set_title("Parameter Recovery Accuracy (Autograd Methods)")
    ax.legend()
    ax.set_yscale("log")

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "fig2_param_recovery.pdf"))
    fig.savefig(os.path.join(output_dir, "fig2_param_recovery.png"))
    plt.close(fig)
    print(f"Figure 2 saved to {output_dir}/fig2_param_recovery.pdf")


def plot_wall_time(results_dir: str, output_dir: str):
    """Figure 3: Wall-clock time per method x program."""
    summaries = load_summaries(results_dir)
    stats = compute_stats(summaries)
    stats_map = {(s.method, s.program): s for s in stats}

    methods = list(METHOD_STYLES.keys())
    programs = [p.name for p in ALL_PROGRAMS]

    x = np.arange(len(programs))
    width = 0.15

    fig, ax = plt.subplots(figsize=(14, 5))
    for j, method in enumerate(methods):
        means = []
        stds = []
        for prog in programs:
            s = stats_map.get((method, prog))
            if s:
                means.append(s.mean_wall_time)
                stds.append(s.std_wall_time)
            else:
                means.append(0)
                stds.append(0)
        style = METHOD_STYLES[method]
        ax.bar(x + j * width, means, width, yerr=stds,
               color=style["color"], label=style["label"],
               capsize=2, alpha=0.85)

    ax.set_xticks(x + width * 2)
    ax.set_xticklabels([PROGRAM_LABELS[p] for p in programs],
                       rotation=30, ha="right")
    ax.set_ylabel("Total Wall Time (s)")
    ax.set_title("Computation Cost by Method and Program")
    ax.legend(loc="upper left")

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "fig3_wall_time.pdf"))
    fig.savefig(os.path.join(output_dir, "fig3_wall_time.png"))
    plt.close(fig)
    print(f"Figure 3 saved to {output_dir}/fig3_wall_time.pdf")


def plot_grad_norms(results_dir: str, output_dir: str,
                    max_epochs: int | None = None):
    """Figure 4: Gradient norm over training (autograd methods only), 2x3 grid."""
    import csv
    from collections import defaultdict

    programs = [p.name for p in ALL_PROGRAMS]
    methods = ["direct", "compiled_interp", "handcoded_interp"]

    # Load gradient norm histories
    grad_groups: dict[tuple, list[list[float]]] = defaultdict(list)
    for f in sorted(Path(results_dir).glob("*.csv")):
        parts = f.stem.rsplit("_", 1)
        seed = int(parts[-1])
        method_prog = parts[0]
        for prog in programs:
            if method_prog.endswith(prog):
                method = method_prog[: -len(prog) - 1]
                if method not in methods:
                    break
                with open(f) as fh:
                    reader = csv.DictReader(fh)
                    norms = [float(row["grad_norm"]) for row in reader]
                grad_groups[(method, prog)].append(norms)
                break

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for i, prog in enumerate(programs):
        ax = axes[i]
        for method in methods:
            key = (method, prog)
            if key not in grad_groups:
                continue
            all_norms = grad_groups[key]
            min_len = min(len(n) for n in all_norms)
            if max_epochs is not None:
                min_len = min(min_len, max_epochs)
            arr = np.array([n[:min_len] for n in all_norms])
            mean = arr.mean(axis=0)
            std = arr.std(axis=0)
            epochs = np.arange(min_len)
            style = METHOD_STYLES[method]
            ax.semilogy(epochs, mean, color=style["color"],
                        ls=style["ls"], label=style["label"])
            ax.fill_between(epochs, np.maximum(mean - std, 1e-10),
                            mean + std, color=style["color"], alpha=0.15)

        ax.set_title(PROGRAM_LABELS.get(prog, prog))
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Gradient L2 Norm")
        if i == 0:
            ax.legend(loc="upper right", framealpha=0.9)

    fig.suptitle("Gradient Norm Over Training", fontsize=14, y=1.02)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "fig4_grad_norms.pdf"))
    fig.savefig(os.path.join(output_dir, "fig4_grad_norms.png"))
    plt.close(fig)
    print(f"Figure 4 saved to {output_dir}/fig4_grad_norms.pdf")


def plot_ablation_grad_path(results_dir: str, output_dir: str):
    """Figure 5c: Gradient path length (autograd nodes) per program."""
    ablation_path = Path(results_dir) / "ablations" / "ablations.json"
    if not ablation_path.exists():
        print(f"No ablation results at {ablation_path}")
        return

    with open(ablation_path) as f:
        data = json.load(f)

    grad_data = data.get("grad_path", [])
    if not grad_data:
        print("No grad_path data in ablations.")
        return

    programs = []
    direct_nodes = []
    interp_nodes = []
    for entry in grad_data:
        if entry["method"] == "direct":
            programs.append(entry["program"])
            direct_nodes.append(entry["autograd_nodes"])
        elif entry["method"] == "compiled_interp":
            interp_nodes.append(entry["autograd_nodes"])

    x = np.arange(len(programs))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, direct_nodes, width,
           color=METHOD_STYLES["direct"]["color"],
           label="Direct compilation", alpha=0.85)
    ax.bar(x + width / 2, interp_nodes, width,
           color=METHOD_STYLES["compiled_interp"]["color"],
           label="Compiled interpreter", alpha=0.85)

    labels = [PROGRAM_LABELS.get(p, p) for p in programs]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Autograd Graph Nodes")
    ax.set_title("Gradient Path Length: Direct vs. Compiled Interpreter")
    ax.legend()

    for i, (d, c) in enumerate(zip(direct_nodes, interp_nodes)):
        ax.text(i - width / 2, d + 0.3, str(d), ha="center", fontsize=8)
        ax.text(i + width / 2, c + 0.3, str(c), ha="center", fontsize=8)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "fig5_grad_path.pdf"))
    fig.savefig(os.path.join(output_dir, "fig5_grad_path.png"))
    plt.close(fig)
    print(f"Figure 5 saved to {output_dir}/fig5_grad_path.pdf")


def generate_all_figures(results_dir: str = DEFAULT.output_dir,
                         output_dir: str | None = None,
                         max_epochs: int | None = None):
    if output_dir is None:
        output_dir = str(Path(results_dir) / "figures")

    print(f"Generating figures from {results_dir}...")
    plot_convergence_curves(results_dir, output_dir, max_epochs=max_epochs)
    plot_param_recovery(results_dir, output_dir)
    plot_wall_time(results_dir, output_dir)
    plot_grad_norms(results_dir, output_dir, max_epochs=max_epochs)
    plot_ablation_grad_path(results_dir, output_dir)
    print(f"\nAll figures saved to {output_dir}/")


if __name__ == "__main__":
    import sys
    results_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT.output_dir
    generate_all_figures(results_dir)
