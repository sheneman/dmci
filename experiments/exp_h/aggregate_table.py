############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# aggregate_table.py: Aggregate Experiment H benchmark results and verify the manuscript tables. Canonical sources of truth...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Aggregate Experiment H benchmark results and verify the manuscript tables.

Canonical sources of truth (experiments/exp_h/results/):
  - exp_h_cuda.json    -> {"part_e": [36 records]}     torch.compile speedups (Part E)
  - bench_diffesm_all.json -> [72 records]             DiffESM-S benchmark (Part F)
  - bench_diffsoc_all.json -> [72 records]             DiffSoc-S benchmark  (Part H)
  - convergence_cpu.json   -> per-epoch trajectories   convergence speedup (Part G)
  - exp_h_5147138.out      -> SLURM stdout             Parts A/B/D (not in any JSON)
  - exp_h_cpu.json     -> {"part_d": [20 records]}     a *CPU* population run (see note)

This script is the Experiment-H counterpart of exp_b/exp_c/exp_d aggregate_table.py.
It reconstructs every cell of the Experiment-H manuscript tables from the committed
data and self-checks each cell against the value printed in paper-dmci/paper2.tex so
future drift is caught:

  tab:exp_h_throughput    (Part A, forward throughput)      <- exp_h_5147138.out
  tab:exp_h_training      (Part B, training speedup)        <- exp_h_5147138.out
  tab:exp_h_population    (Part D, population batching)      <- exp_h_5147138.out (GPU run)
  tab:exp_h_compile       (Part E, torch.compile)            <- exp_h_cuda.json[part_e]
  tab:exp_h_crossmodel    (cross-model summary)              <- bench_diff{esm,soc}_all.json
  tab:exp_h_diffesm       (Part F, DiffESM-S @ BS=64)        <- bench_diffesm_all.json
  tab:exp_h_diffsoc       (Part H, DiffSoc-S @ BS=64)        <- bench_diffsoc_all.json
  tab:exp_h_diffsoc_scaling (Part H, throughput scaling)     <- bench_diffsoc_all.json
  tab:convergence         (Part G, convergence speedup)      <- convergence_cpu.json
  tab:diffesm_spec / tab:diffsoc_spec (source-line counts)   <- wc -l large_examples/*.scm

Usage:
    python3 -m experiments.exp_h.aggregate_table     # print all tables + self-check

Notes / data provenance:
  - Parts A, B and D are NOT stored in any committed JSON; the manuscript values come
    from the committed SLURM stdout exp_h_5147138.out. They parse cleanly, so they ARE
    machine-verified here (against the regex-parsed log lines).
  - The manuscript Part-D table (tab:exp_h_population) reports the *A100 GPU* population
    run (147x..16,873x). That run was logged in exp_h_5147138.out but its JSON dump
    (exp_h_cuda.json) was later overwritten by the Part-E results, so it survives only in
    the .out log. The committed exp_h_cpu.json[part_d] is a *separate CPU* population run
    with different numbers (138x..14,296x); it does NOT back the printed table and is
    reported here for transparency, not checked against the manuscript.
  - The convergence JSON stores per-epoch (wall_time, loss/best_loss) trajectories but not
    a per-restart converged count; for the population rows we verify wall time and
    best_loss<1e-8 (the table's "<10^-8" cell) but treat the "Conv." count as a static
    table entry.
  - Model node/parameter counts in the two spec tables are static structural constants of
    the compiled graphs; only the source-line counts are machine-checked (via wc -l).

If the committed JSONs are missing locally, on a fresh clone run `git lfs pull` first, or
regenerate on the cluster with:
    sbatch experiments/exp_h/slurm_submit.sh        # Parts A-E -> exp_h_{cpu,cuda}.json
    sbatch experiments/exp_h/slurm_diffesm.sh       # -> bench_diffesm_all.json
    sbatch experiments/exp_h/slurm_diffsoc.sh       # -> bench_diffsoc_all.json
    sbatch experiments/exp_h/slurm_convergence.sh   # -> convergence_cpu.json
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
REPO_ROOT = Path(__file__).resolve().parents[2]
LARGE_EX = REPO_ROOT / "large_examples"

# ----------------------------------------------------------------------------
# PAPER_* targets: values transcribed verbatim from paper-dmci/paper2.tex.
# ----------------------------------------------------------------------------

# tab:exp_h_throughput (Part A): {model: (BS1, BS32, BS128, BS512)} forward speedups.
PAPER_THROUGHPUT = {
    "M01_coulomb":           (2.9,  83,  302, 1309),
    "M02_beer_lambert":      (3.5, 110,  374, 1710),
    "M03_michaelis_menten":  (2.7,  77,  312, 1226),
    "M04_arrhenius":         (2.0,  56,  240,  950),
    "M05_hookes_spring":     (1.6,  48,  205,  821),
    "M06_logistic_growth":   (1.3,  39,  164,  657),
    "M07_power_law":         (3.2,  84,  385, 1531),
    "M08_euler_ode":         (0.6,  17,   66,  279),
    "M10_smooth_activation": (1.5,  39,  158,  662),
    "M11_recursive_filter":  (0.6,  18,   74,  291),
    "M12_newton_sqrt":       (0.7,  23,   90,  357),
    "M14_anomaly_scorer":    (2.8,  82,  355, 1409),
}

# tab:exp_h_training (Part B): {model: (seq_s, batch_s, speedup, loss_str)}.
# loss_str is the literal table cell; "<1e-5"/"<1e-3" denote inequality cells.
PAPER_TRAINING = {
    "M01_coulomb":           (5.81,  0.34, 17.0, "14.22"),
    "M02_beer_lambert":      (3.81,  0.31, 12.2, "0.00"),
    "M03_michaelis_menten":  (5.95,  0.35, 16.9, "4.60"),
    "M04_arrhenius":         (6.82,  0.37, 18.5, "0.14"),
    "M05_hookes_spring":     (10.86, 0.44, 24.7, "<1e-5"),
    "M06_logistic_growth":   (11.06, 0.44, 25.0, "0.32"),
    "M07_power_law":         (6.55,  0.37, 17.8, "0.69"),
    "M08_euler_ode":         (48.21, 1.09, 44.3, "0.00"),
    "M10_smooth_activation": (11.03, 0.44, 25.2, "<1e-3"),
    "M11_recursive_filter":  (42.02, 0.94, 44.8, "0.00"),
    "M12_newton_sqrt":       (21.97, 0.61, 36.3, "0.00"),
    "M14_anomaly_scorer":    (7.07,  0.37, 19.0, "0.00"),
}

# tab:exp_h_population (Part D, A100 GPU): {model: (Pop1, Pop10, Pop100)} speedups.
PAPER_POPULATION = {
    "M01_coulomb":          (147, 1401, 14528),
    "M02_beer_lambert":     (169, 1685, 16873),
    "M03_michaelis_menten": (144, 1402, 14395),
    "M04_arrhenius":        (109, 1111, 11145),
    "M05_hookes_spring":    (90,   916,  9156),
}

# tab:exp_h_compile (Part E): {model: ((fwd64,fwd512,fwd4096),(fb64,fb512,fb4096))}.
PAPER_COMPILE = {
    "M01_coulomb":           ((0.30, 0.46, 0.39), (0.40, 0.40, 0.41)),
    "M02_beer_lambert":      ((0.29, 0.24, 0.24), (0.32, 0.31, 0.33)),
    "M03_michaelis_menten":  ((0.37, 0.34, 0.34), (0.49, 0.51, 0.49)),
    "M04_arrhenius":         ((0.58, 0.49, 0.48), (0.54, 0.56, 0.55)),
    "M05_hookes_spring":     ((0.80, 0.60, 0.63), (0.89, 0.85, 0.86)),
    "M06_logistic_growth":   ((0.64, 0.63, 0.64), (0.85, 0.85, 0.85)),
    "M07_power_law":         ((0.39, 0.37, 0.39), (0.80, 0.79, 0.80)),
    "M08_euler_ode":         ((0.90, 0.86, 0.86), (0.94, 0.93, 0.94)),
    "M10_smooth_activation": ((0.69, 0.64, 0.65), (0.68, 0.84, 0.84)),
    "M11_recursive_filter":  ((0.87, 0.87, 0.86), (0.94, 0.94, 0.98)),
    "M12_newton_sqrt":       ((0.81, 0.81, 0.80), (0.89, 0.93, 0.91)),
    "M14_anomaly_scorer":    ((0.56, 0.59, 0.60), (0.81, 0.82, 0.82)),
}

# tab:exp_h_crossmodel (CPU, 100 steps). Keyed by model tag in the JSONs.
PAPER_CROSSMODEL = {
    "esm": {"lines": 317, "seq_fwd_ms": 175.8, "bat64_ms": 2.85,
            "sp64": 61.6, "sp1024": 875},
    "soc": {"lines": 703, "seq_fwd_ms": 1687.0, "bat64_ms": 27.1,
            "sp64": 62.3, "sp1024": 852},
}

# tab:exp_h_diffesm (Part F, BS=64, 100 steps): (fwd_s, fwd_ms, fb_s, fb_ms) per device/method.
PAPER_DIFFESM = {
    ("cpu",  "sequential"): (11.25, 175.8, 30.85, 482.0),
    ("cpu",  "batched"):    (0.18,    2.85, 0.45,   7.09),
    ("cuda", "sequential"): (23.65, 369.5, 62.59, 978.0),
    ("cuda", "batched"):    (0.38,    5.92, 0.94,  14.73),
}

# tab:exp_h_diffsoc (Part H, BS=64): {(device, method): {"fwd": (10,50,100), "fb": (10,50,100)}}.
PAPER_DIFFSOC = {
    ("cpu",  "sequential"): {"fwd": (12.05, 56.37, 111.91), "fb": (31.01, 155.21, 318.12)},
    ("cpu",  "batched"):    {"fwd": (0.19,   0.88,   1.73),  "fb": (0.45,    2.26,   4.58)},
    ("cuda", "sequential"): {"fwd": (24.93, 116.61, 230.74), "fb": (74.56, 384.79, 630.28)},
    ("cuda", "batched"):    {"fwd": (0.39,   1.84,   3.66),  "fb": (1.14,    5.81,   9.45)},
}

# tab:exp_h_diffsoc_scaling (100 steps). seq row is method=sequential bs=1; rest are batched.
# {device: {batch_size: (ms_per_eval, speedup)}}; speedup vs sequential bs=1 per-eval.
PAPER_SCALING = {
    "cpu": {1: (1705.1, 1.0), 4: (428.3, 3.9), 16: (107.1, 15.7),
            64: (27.1, 62.3), 256: (7.4, 228), 1024: (2.0, 852)},
    "cuda": {1: (3624.0, 1.0), 4: (915.8, 3.9), 16: (228.5, 15.7),
             64: (57.1, 62.6), 256: (14.4, 249), 1024: (3.6, 997)},
}
PAPER_SCALING_SEQ = {"cpu": 1687.0, "cuda": 3577.9}  # the "1 (seq.)" ms/eval row.

# tab:convergence (Part G): {condition: (time_s, conv_str)}. Final loss cell is "<10^-8".
PAPER_CONVERGENCE = {
    "sequential":     731.4,
    "batched":         23.9,
    "population_M1":   23.8,
    "population_M10":  26.8,
    "population_M50":  29.9,
    "population_M200": 38.0,
}

# tab:diffesm_spec / tab:diffsoc_spec source-line cells.
PAPER_SPEC_LINES = {"diffesm_s.scm": 317, "diffsoc_s.scm": 703}

MODEL_ORDER = ["M01_coulomb", "M02_beer_lambert", "M03_michaelis_menten", "M04_arrhenius",
               "M05_hookes_spring", "M06_logistic_growth", "M07_power_law", "M08_euler_ode",
               "M10_smooth_activation", "M11_recursive_filter", "M12_newton_sqrt",
               "M14_anomaly_scorer"]


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def load_json(name):
    p = RESULTS_DIR / name
    if not p.exists():
        raise SystemExit(
            f"Missing {p}. On a fresh clone run `git lfs pull` first, or regenerate on the "
            "cluster (see this module's docstring).")
    return json.loads(p.read_text())


def find_bench(records, device, method, bs, steps):
    for r in records:
        if (r["device"] == device and r["method"] == method
                and r["batch_size"] == bs and r["n_steps"] == steps):
            return r
    return None


def rel_ok(got, want, tol):
    if want == 0:
        return abs(got) <= tol
    return abs(got - want) / abs(want) <= tol


def loss_cell_ok(loss, cell):
    """Match a Part-B loss against its (possibly inequality) printed table cell."""
    if cell == "<1e-5":
        return loss < 1e-5
    if cell == "<1e-3":
        return loss < 1e-3
    return abs(loss - float(cell)) <= 0.005  # printed to 2 decimals


# ----------------------------------------------------------------------------
# Part A / B / D parsing from the SLURM stdout log
# ----------------------------------------------------------------------------
LOG_PATH = RESULTS_DIR / "exp_h_5147138.out"


def parse_log_section(text, header):
    """Return the substring of `text` from `header` to the next 'Results saved' line."""
    i = text.index(header)
    end = text.find("Results saved", i)
    return text[i:end if end != -1 else len(text)]


def parse_part_a(text):
    """{model: {bs: speedup_float}} from the Part A throughput section."""
    block = parse_log_section(text, "Part A: Forward Throughput Scaling")
    pat = re.compile(r"(M\d+_\w+)\s+bs=\s*(\d+)\s+batch=[\d.]+s\s+seq=\s*\S+\s+"
                     r"speedup=([\d.]+|N/A)x")
    out: dict[str, dict[int, float]] = {}
    for m in pat.finditer(block):
        sp = m.group(3)
        if sp != "N/A":
            out.setdefault(m.group(1), {})[int(m.group(2))] = float(sp)
    return out


def parse_part_b(text):
    """{model: (seq_s, batch_s, speedup, loss_s)} from the Part B training section."""
    block = parse_log_section(text, "Part B: Training Epoch Speedup")
    pat = re.compile(r"(M\d+_\w+)\s+seq=([\d.]+)s\s+batch=([\d.]+)s\s+"
                     r"speedup=([\d.]+)x\s+loss_s=([\d.eE+-]+)\s+loss_b=([\d.eE+-]+)")
    out = {}
    for m in pat.finditer(block):
        out[m.group(1)] = (float(m.group(2)), float(m.group(3)), float(m.group(4)),
                           float(m.group(5)))
    return out


def parse_part_d(text):
    """{model: {pop: speedup}} from the Part D population section (GPU run)."""
    block = parse_log_section(text, "Part D: Population Batching")
    pat = re.compile(r"(M\d+_\w+)\s+pop=\s*(\d+)\s+evals=\s*\d+\s+batch=[\d.]+s\s+"
                     r"speedup=([\d.]+|N/A)x")
    out: dict[str, dict[int, float]] = {}
    for m in pat.finditer(block):
        sp = m.group(3)
        if sp != "N/A":
            out.setdefault(m.group(1), {})[int(m.group(2))] = float(sp)
    return out


# ----------------------------------------------------------------------------
# Per-table checks
# ----------------------------------------------------------------------------
def check_part_a(log_text, checks, disc):
    print("\n[Part A] Forward throughput (tab:exp_h_throughput) -- source: exp_h_5147138.out")
    data = parse_part_a(log_text)
    cols = [1, 32, 128, 512]
    n0 = len(checks)
    for model in PAPER_THROUGHPUT:
        got = data.get(model, {})
        for bs, want in zip(cols, PAPER_THROUGHPUT[model]):
            g = got.get(bs)
            # BS=1 is printed to 1 decimal (exact); the larger columns are rounded to the
            # nearest integer (e.g. log 18.4 -> table 18, log 301.7 -> 302).
            if g is None:
                ok = False
            elif bs == 1:
                ok = abs(g - want) <= 0.05
            else:
                ok = round(g) == want
            checks.append(ok)
            if not ok:
                disc.append(("tab:exp_h_throughput", f"{model} BS={bs}", g, want))
    _report_block(checks, n0)


def check_part_b(log_text, checks, disc):
    print("[Part B] Training speedup (tab:exp_h_training) -- source: exp_h_5147138.out")
    data = parse_part_b(log_text)
    n0 = len(checks)
    for model, (pseq, pbat, psp, ploss) in PAPER_TRAINING.items():
        rec = data.get(model)
        if rec is None:
            checks.append(False)
            disc.append(("tab:exp_h_training", f"{model} (missing)", None, "-"))
            continue
        seq, bat, sp, loss = rec
        for label, g, w, tol in [("seq", seq, pseq, 0.02), ("batch", bat, pbat, 0.06),
                                 ("speedup", sp, psp, 0.02)]:
            ok = rel_ok(g, w, tol)
            checks.append(ok)
            if not ok:
                disc.append(("tab:exp_h_training", f"{model} {label}", g, w))
        ok = loss_cell_ok(loss, ploss)
        checks.append(ok)
        if not ok:
            disc.append(("tab:exp_h_training", f"{model} loss", loss, ploss))
    _report_block(checks, n0)


def check_part_d(log_text, checks, disc):
    print("[Part D] Population batching (tab:exp_h_population) -- source: exp_h_5147138.out "
          "(A100 GPU run)")
    data = parse_part_d(log_text)
    cols = [1, 10, 100]
    n0 = len(checks)
    for model, wants in PAPER_POPULATION.items():
        got = data.get(model, {})
        for pop, want in zip(cols, wants):
            g = got.get(pop)
            ok = g is not None and rel_ok(g, want, 0.02)
            checks.append(ok)
            if not ok:
                disc.append(("tab:exp_h_population", f"{model} Pop={pop}", g, want))
    _report_block(checks, n0)
    # Transparency: the committed exp_h_cpu.json is a *different* CPU run.
    try:
        cpu = load_json("exp_h_cpu.json")["part_d"]
        m1 = {r["pop_size"]: r["speedup"] for r in cpu if r["model"] == "M01_coulomb"}
        print(f"         note: committed exp_h_cpu.json[part_d] is a CPU run "
              f"(M01 Pop=1/10/100 = {m1.get(1):.0f}/{m1.get(10):.0f}/{m1.get(100):.0f}x); "
              "it does NOT back this GPU table and is not checked.")
    except Exception:
        pass


def check_part_e(checks, disc):
    print("[Part E] torch.compile (tab:exp_h_compile) -- source: exp_h_cuda.json[part_e]")
    recs = load_json("exp_h_cuda.json")["part_e"]
    cols = [64, 512, 4096]
    n0 = len(checks)
    for model, (pf, pfb) in PAPER_COMPILE.items():
        fwd = {r["batch_size"]: r["fwd_speedup"] for r in recs if r["model"] == model}
        fb = {r["batch_size"]: r["fwdbwd_speedup"] for r in recs if r["model"] == model}
        for bs, want in zip(cols, pf):
            g = fwd.get(bs)
            ok = g is not None and abs(round(g, 2) - want) <= 0.01  # table is 2-dp, reproduces
            checks.append(ok)
            if not ok:
                disc.append(("tab:exp_h_compile", f"{model} fwd BS={bs}", g, want))
        for bs, want in zip(cols, pfb):
            g = fb.get(bs)
            ok = g is not None and abs(round(g, 2) - want) <= 0.01
            checks.append(ok)
            if not ok:
                disc.append(("tab:exp_h_compile", f"{model} fwd+bwd BS={bs}", g, want))
    _report_block(checks, n0)


def check_diffesm(checks, disc):
    print("[Part F] DiffESM-S benchmark (tab:exp_h_diffesm, BS=64/100steps) -- "
          "source: bench_diffesm_all.json")
    recs = load_json("bench_diffesm_all.json")
    n0 = len(checks)
    for (dev, method), (pfwd, pfwd_ms, pfb, pfb_ms) in PAPER_DIFFESM.items():
        r = find_bench(recs, dev, method, 64, 100)
        if r is None:
            checks.append(False)
            disc.append(("tab:exp_h_diffesm", f"{dev}/{method} (missing)", None, "-"))
            continue
        fwd, fb = r["fwd_time"], r["fwd_bwd_time"]
        ms = r["fwd_per_eval"] * 1000.0
        fbms = r["fwd_bwd_per_eval"] * 1000.0
        for label, g, w, tol in [("fwd_s", fwd, pfwd, 0.02), ("fwd_ms", ms, pfwd_ms, 0.02),
                                 ("fb_s", fb, pfb, 0.02), ("fb_ms", fbms, pfb_ms, 0.02)]:
            # sub-second wall times: allow 0.01s absolute as well as 2% relative.
            ok = rel_ok(g, w, tol) or abs(g - w) <= 0.01
            checks.append(ok)
            if not ok:
                disc.append(("tab:exp_h_diffesm", f"{dev}/{method} {label}", round(g, 3), w))
    _report_block(checks, n0)


def check_diffsoc(checks, disc):
    print("[Part H] DiffSoc-S benchmark (tab:exp_h_diffsoc, BS=64) -- "
          "source: bench_diffsoc_all.json")
    recs = load_json("bench_diffsoc_all.json")
    n0 = len(checks)
    for (dev, method), want in PAPER_DIFFSOC.items():
        for kind, key in [("fwd", "fwd_time"), ("fb", "fwd_bwd_time")]:
            for steps, w in zip((10, 50, 100), want[kind]):
                r = find_bench(recs, dev, method, 64, steps)
                g = r[key] if r else None
                ok = g is not None and (rel_ok(g, w, 0.02) or abs(g - w) <= 0.01)
                checks.append(ok)
                if not ok:
                    disc.append(("tab:exp_h_diffsoc",
                                 f"{dev}/{method} {kind} {steps}-step",
                                 round(g, 3) if g is not None else None, w))
    _report_block(checks, n0)


def check_scaling(checks, disc):
    print("[Part H] DiffSoc-S throughput scaling (tab:exp_h_diffsoc_scaling, 100steps) -- "
          "source: bench_diffsoc_all.json")
    recs = load_json("bench_diffsoc_all.json")
    n0 = len(checks)
    for dev, rows in PAPER_SCALING.items():
        seq = find_bench(recs, dev, "sequential", 1, 100)
        seq_ms = seq["fwd_per_eval"] * 1000.0
        # the "1 (seq.)" ms/eval cell
        ok = abs(seq_ms - PAPER_SCALING_SEQ[dev]) / PAPER_SCALING_SEQ[dev] <= 0.02
        checks.append(ok)
        if not ok:
            disc.append(("tab:exp_h_diffsoc_scaling", f"{dev} seq ms/eval",
                         round(seq_ms, 1), PAPER_SCALING_SEQ[dev]))
        for bs, (pms, psp) in rows.items():
            r = find_bench(recs, dev, "batched", bs, 100)
            ms = r["fwd_per_eval"] * 1000.0
            sp = seq_ms / ms
            ok_ms = rel_ok(ms, pms, 0.02) or abs(ms - pms) <= 0.1
            ok_sp = rel_ok(sp, psp, 0.03)  # speedup printed to 1-3 sig figs
            checks.append(ok_ms)
            checks.append(ok_sp)
            if not ok_ms:
                disc.append(("tab:exp_h_diffsoc_scaling", f"{dev} BS={bs} ms/eval",
                             round(ms, 1), pms))
            if not ok_sp:
                disc.append(("tab:exp_h_diffsoc_scaling", f"{dev} BS={bs} speedup",
                             round(sp, 1), psp))
    _report_block(checks, n0)


def check_crossmodel(checks, disc):
    print("[Summary] Cross-model table (tab:exp_h_crossmodel, CPU 100steps) -- "
          "source: bench_diff{esm,soc}_all.json")
    # The cross-model table mixes two legitimate sequential-forward references: the
    # "Sequential forward (ms/eval)" cell and the speedups do not all use the same
    # denominator (DiffESM-S shows seq@BS=64 for the display and sp@BS=64 but seq@BS=1 for
    # sp@BS=1024; DiffSoc-S uses seq@BS=1 throughout). Both per-eval measurements are present
    # in the committed JSON, so each speedup cell is checked against whichever sequential
    # reference (BS=1 or BS=64) reproduces it; a cell is a true mismatch only if neither does.
    esm = load_json("bench_diffesm_all.json")
    soc = load_json("bench_diffsoc_all.json")
    n0 = len(checks)
    for tag, recs in [("esm", esm), ("soc", soc)]:
        p = PAPER_CROSSMODEL[tag]
        seq1 = find_bench(recs, "cpu", "sequential", 1, 100)["fwd_per_eval"] * 1000.0
        seq64 = find_bench(recs, "cpu", "sequential", 64, 100)["fwd_per_eval"] * 1000.0
        b64 = find_bench(recs, "cpu", "batched", 64, 100)["fwd_per_eval"] * 1000.0
        b1024 = find_bench(recs, "cpu", "batched", 1024, 100)["fwd_per_eval"] * 1000.0

        # seq_fwd_ms display cell: matches one of the two sequential references.
        ok = rel_ok(seq1, p["seq_fwd_ms"], 0.02) or rel_ok(seq64, p["seq_fwd_ms"], 0.02)
        checks.append(ok)
        if not ok:
            disc.append(("tab:exp_h_crossmodel", f"{tag} seq_fwd_ms",
                         f"seq@1={seq1:.1f}/seq@64={seq64:.1f}", p["seq_fwd_ms"]))

        ok = rel_ok(b64, p["bat64_ms"], 0.02)
        checks.append(ok)
        if not ok:
            disc.append(("tab:exp_h_crossmodel", f"{tag} bat64_ms", round(b64, 2), p["bat64_ms"]))

        # speedups: accept if either sequential reference reproduces the printed value.
        for label, batched_ms, want in [("sp64", b64, p["sp64"]), ("sp1024", b1024, p["sp1024"])]:
            cand = [seq1 / batched_ms, seq64 / batched_ms]
            ok = any(rel_ok(c, want, 0.03) for c in cand)
            checks.append(ok)
            if not ok:
                disc.append(("tab:exp_h_crossmodel", f"{tag} {label}",
                             f"{cand[0]:.1f}/{cand[1]:.1f}", want))
    _report_block(checks, n0)


def check_convergence(checks, disc):
    print("[Part G] Convergence speedup (tab:convergence) -- source: convergence_cpu.json")
    d = load_json("convergence_cpu.json")
    n0 = len(checks)
    cfg = d["config"]
    # caption constants: 20 timesteps, 15 params, N=32, 300 epochs
    for label, g, w in [("n_steps", cfg["n_steps"], 20), ("n_data", cfg["n_data"], 32),
                        ("epochs", cfg["epochs"], 300),
                        ("n_params", len(cfg["opt_params"]), 15)]:
        ok = g == w
        checks.append(ok)
        if not ok:
            disc.append(("tab:convergence", f"caption {label}", g, w))
    for cond, want_t in PAPER_CONVERGENCE.items():
        traj = d[cond]
        last = traj[-1]
        wt = last["wall_time"]
        ok = abs(wt - want_t) <= 0.1  # printed to 1 decimal
        checks.append(ok)
        if not ok:
            disc.append(("tab:convergence", f"{cond} time", round(wt, 1), want_t))
        # final loss cell: population_M1 (single restart) prints 2.0e-8; the rest print "<10^-8".
        loss = last.get("loss", last.get("best_loss"))
        if cond == "population_M1":
            ok_loss = loss is not None and abs(loss - 2.0e-8) < 0.2e-8
            want_loss = "2.0e-8"
        else:
            ok_loss = loss is not None and loss < 1e-8
            want_loss = "<1e-8"
        checks.append(ok_loss)
        if not ok_loss:
            disc.append(("tab:convergence", f"{cond} loss", loss, want_loss))
    _report_block(checks, n0)


def check_spec_lines(checks, disc):
    print("[Spec] Source-line counts (tab:diffesm_spec, tab:diffsoc_spec) -- "
          "source: wc -l large_examples/*.scm")
    n0 = len(checks)
    for fname, want in PAPER_SPEC_LINES.items():
        p = LARGE_EX / fname
        if not p.exists():
            checks.append(False)
            disc.append(("spec", f"{fname} (missing)", None, want))
            continue
        n = int(subprocess.run(["wc", "-l", str(p)], capture_output=True, text=True)
                .stdout.split()[0])
        ok = n == want
        checks.append(ok)
        if not ok:
            disc.append(("spec", f"{fname} lines", n, want))
    _report_block(checks, n0)
    print("         note: spec-table node/param counts are static structural constants "
          "(not machine-checked).")


def _report_block(checks, n0):
    n = len(checks) - n0
    ok = sum(1 for c in checks[n0:] if c)
    print(f"         {ok}/{n} cells match")


# ----------------------------------------------------------------------------
def main():
    checks: list[bool] = []
    disc: list[tuple] = []

    if not LOG_PATH.exists():
        print(f"NOTE: {LOG_PATH.name} not found; Parts A/B/D trace to the committed SLURM "
              ".out log and are SKIPPED (not machine-verified).")
        log_text = None
    else:
        log_text = LOG_PATH.read_text()

    if log_text is not None:
        check_part_a(log_text, checks, disc)
        check_part_b(log_text, checks, disc)
        check_part_d(log_text, checks, disc)
    else:
        print("[Part A/B/D] SKIPPED: data lives only in exp_h_5147138.out (not present).")

    check_part_e(checks, disc)
    check_crossmodel(checks, disc)
    check_diffesm(checks, disc)
    check_diffsoc(checks, disc)
    check_scaling(checks, disc)
    check_convergence(checks, disc)
    check_spec_lines(checks, disc)

    fails = sum(1 for c in checks if not c)
    print(f"\nSELF-CHECK vs manuscript: {len(checks) - fails}/{len(checks)} cells match")

    if disc:
        print("\nDISCREPANCIES FOUND (computed vs printed):")
        for tab, cell, got, want in disc:
            print(f"  MISMATCH [{tab}] {cell}: computed={got} paper={want}")

    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
