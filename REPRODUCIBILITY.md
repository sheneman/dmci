# Reproducibility Guide

This document is the single entry point for reproducing the results in
"Compile Once, Differentiate Everywhere" (the Differentiable
Meta-Circular Interpreter, DMCI). It covers environment setup, a one-command quick
verification, per-experiment instructions with expected numbers and data provenance, an
artifact-to-table/figure map, seeds, and hardware.

Reproducibility is a core value of this work. Every source file carries an authorship
header; every result-bearing directory has a `MANIFEST.txt` mapping artifacts to the exact
manuscript claim they back; every real dataset has a SHA256-pinned `metadata.json`. Where an
artifact must be regenerated from a public dataset or a long HPC run rather than shipped, this
guide says so explicitly.

---

## 1. Environment

- Python >= 3.11
- PyTorch >= 2.0 (CPU is sufficient for the quick path and all CPU experiments)
- NumPy >= 1.24, SciPy >= 1.10, pytest >= 7.0
- Optional: JAX (the `jax.vmap` baseline in Experiment H; lambdify comparators in
  Appendices I and J), CuPy (forward-only GPU backend), `openai` (only to re-run live LLM
  generation).

The environment is pinned by `pyproject.toml` and `requirements.txt`. The exact cluster
build used to produce the reported numbers is recorded in `requirements.txt`:
PyTorch 2.12.0+cu130, NumPy 2.4.6, SciPy 1.17.1, JAX 0.10.1, Python 3.11.10 (CUDA 13.0).

```bash
python -m venv .venv && source .venv/bin/activate    # or `uv venv`
pip install -e .
```

The HPC `.venv` is self-contained; no module system is required.

---

## 2. One-command quick verification (minutes, CPU)

```bash
./reproduce.sh
```

Installs the package editable and runs the full test suite (**1,225 tests, ~37 s on a laptop
CPU**). It verifies the load-bearing claims that need no cluster:

- the self-hosted Scheme evaluator (`bootstrap/compiler.scm`) compiles and runs;
- gradients flow through the interpreter and **DMCI == direct compilation** (to ~1e-6,
  bit-identical loss histories on several recursive programs);
- **batched evaluation matches sequential** evaluation;
- `.ncg` serialization round-trips.

Integrity of the committed result artifacts:

```bash
python experiments/check_manifests.py
```

The full cluster experiment commands are printed by `./reproduce.sh full`.

---

## 3. Hardware

- Experiment H (batched throughput, population, large-model benchmarks): a single
  NVIDIA A100, plus CPU runs.
- All other experiments: university HPC CPU cores (the `eight` partition) unless noted;
  the LLM co-search islands (battery, FluZoo) run on the GPU partition for the LLM mutator.
- Quick verification and all aggregators run on a laptop CPU.

Timing numbers (ms-level compile/hot-swap, wall-clock per fit) are hardware-dependent; only
relative ratios and method-equivalence claims are portable.

---

## 4. Per-experiment reproduction

Each experiment directory has its own `README.md`. Seeds and hyperparameters are in each
`config.py` and embedded in every result file. Self-checking aggregators
(`aggregate_table.py`) rebuild the manuscript tables and assert each cell.

### Experiment A: gradient fidelity / trajectory equivalence (synthetic), §4.1, App. A
- Data: fully synthetic, seed-regenerable (8-point grid on [0.5, 3.0]).
- Run: `python -m experiments.exp_a.run_all` (SLURM: `slurm_submit.sh`, plus
  `slurm_noise.sh`, `slurm_gradient_delta.sh`, `slurm_s3_branch.sh`, `slurm_profile.sh`).
- Expected: trajectory equivalence |DMCI - direct| < 7e-7; gradient relerr 0.0, cosine min
  0.99999982 over 300 samples; Table P1..P6 as printed.

### Experiment B: LLM-generated differentiable programs (synthetic + LLM cache), §4.2, App. B
- Run (the reported run): `slurm_array_llm.sh` -> `results_llm/`. The hand-authored reference
  run (`slurm_submit.sh` -> `results/`, path A) agrees except on M15. **The manuscript numbers
  come from `results_llm/`.**
- Verify: `python -m experiments.exp_b.aggregate_table` (65/65 cells; DMCI 513 s / 73x / 248x).
- The 15 committed LLM programs live in `llm_cache/`; live regeneration needs an LLM key and is
  non-deterministic.

### Experiment C: recursive scientific models (synthetic), §4.3, App.
- Run: `slurm_submit.sh` (full DMCI C01/C06 ~7-9.6 h/seed, HPC). Results are Git LFS
  (`git lfs pull`).
- Verify: `python -m experiments.exp_c.aggregate_table` (43/43 cells).

### LIM-ENSO: real-data Kalman MLE (ERSSTv5), §4.3, App. `app:exp_lim_enso_ext`
- Data: **NOAA ERSSTv5** `sst.mnmean.nc`; `python -m experiments.exp_lim_enso.data.build_data`
  (SHA256-verified). The tracked `data/processed/metadata.json` is the binding contract;
  raw `.nc` / processed `.npy` are regenerated, not committed.
- Run: gate `slurm_gate.sh`, sweep `slurm_parallel.sh` (per-fit 4-8 h CPU, HPC). Local smoke:
  `python -m experiments.exp_lim_enso.smoke`.
- Verify: `python -m experiments.exp_lim_enso.aggregate` rebuilds T1-T5 from the committed
  per-run records (`results/*.json`, ~3.5 MB plain git), so the corrected aggregates are
  regenerable from the repo without the cluster. T1/T2/T4 restrict to the S0 dense operator
  (the canonical scaling/robustness comparison); the alternate F-structures are reported only
  in T5. Re-running reproduces the headline D=15 cell (dmci_adam train NLL -3675.6, n=3).

### Experiment D: structural-search cost (synthetic), App. `app:exp_d`
- Run: `slurm_submit.sh` (DMCI ~57-79 min/seed, HPC). Verify:
  `python -m experiments.exp_d.aggregate_table` (10/10; 25.3x crossover).

### Experiment E: Gumbel-Softmax operator recovery (synthetic), App. `app:exp_e`
- Run: `sbatch experiments/exp_e/slurm_submit_e1.sh` (note the `_e1` suffix; ~46 CPU-hours).
- Expected: DMCI 26/240 (10.8%), Exhaustive 240/240, Evolutionary 199/240 (82.9%),
  Random 5/240 (2.1%).

### Experiment F: LLM-in-the-loop discovery (synthetic + live LLM), App. `app:exp_f`
- Run: baseline `slurm_submit.sh`; thinking-mode `slurm_f_thinking.sh`. Needs LLM egress;
  the committed JSONs are authoritative (LLM output is non-deterministic).
- Expected (Table 6 rows): 6/12, 6/12, 9/12, 11/12, and F3-only multistart 3/3.

### Experiment G: runtime compositional modeling (synthetic), App. `app:exp_g`
- Run: `slurm_submit.sh` (CPU). The table is a per-label mean over 5 seeds of `results/*.json`
  (logic in the README).
- Expected: G1 0.942 / 0.772 / 20.0 ms; G2 0.0004 / 0.108 / 35.2 ms; G3 0.0004 / 0.0007 / 27.2 ms.

### Experiment H: batched parallelization (synthetic; A100), §4.5, App. `app:exp_h_ext`
- Run: `slurm_submit.sh` then `slurm_part_e.sh` **in order** (Part E overwrites
  `exp_h_cuda.json`; the Part-D population table survives in the committed SLURM `.out`), plus
  `slurm_diffesm.sh`, `slurm_diffsoc.sh`, `slurm_convergence.sh`, `slurm_vmap.sh`. Local smoke:
  `python -m experiments.exp_h.exp_h --part C --device cpu` (12/12 PASS, pred_diff 0.0).
- Verify: `python -m experiments.exp_h.aggregate_table` (**275/275 cells**, including the
  DiffESM-S/DiffSoc-S source-line counts 317 / 703 against `wc -l`).

### Experiment I: d-scaling sweep (synthetic), App. `app:exp_i`
- Run: `slurm_scaling.sh` -> `python -m experiments.exp_i.aggregate_scaling` ->
  `exp_i_scaling.dat` under the experiment's results directory.

### Experiment J: program-space calibration vs compile-each (synthetic + LLM subset), App. `app:exp_j`
- Run: `slurm_expj.sh` (synthetic) and `slurm_expj_llm.sh` (LLM subset). The 260 generated
  programs are preserved inside the committed `expj_llm_results.json` (auditable from the
  in-JSON program strings).

### Experiment L: battery capacity-fade co-search (synthetic + real Severson), §4.4, App. `app:exp_battery_ext`
- Data: synthetic leg fully seed-regenerable (`gen_target.py`). Real leg: **Severson et al.**
  (DOI `10.1038/s41560-019-0356-8`, `data.matr.io/1/`); `data/raw/severson_capacity.{pkl,npz}`
  are gitignored and SHA256-pinned in `data/metadata.json`. Built target `results/target_real.pt`
  (117 cells) is committed.
- Run: OpenEvolve islands `openevolve/run_real_islands.sh` -> re-scores
  `rescore.py` / `rescore_real.py` / `rescore_knee.py`, ablation `ablate_inner.py`, funnel
  `funnel.py` (via `openevolve/slurm_rescore.sh`).
- Committed (see `results/MANIFEST.txt`): `battery_rescore.json`,
  `battery_rescore_real_ks{70,45}.json`, `battery_rescore_real_knee_ks{70,45}.json`,
  `battery_ablate_inner_ks{70,45}.json`, `funnel_bat_island.json`, `funnel_bat_real_island.json`,
  `target.pt`, `target_real.pt`, and the six `bat_island_*/best/` + `bat_real_island_*/best/`
  winners.
- Expected: synthetic recovery (iters=300) island0 0.0111 / island1 0.0204 / island2 0.0135 vs
  smooth seed 0.0301; real held-out (LATE) evolved island2 0.0512 / island1 0.0267 beat the best
  hand baseline (sigmoidal 0.0507); inner-fitter ablation (gradient-free DE) 9-27x worse;
  pooled distinct structures 119 (synthetic) / 160 (real). Full 200-program island archives are
  not committed (the funnel JSON summarizes them).

### FluZoo (Experiment K): influenza co-search stress test (real ILINet), §4.5, App. `app:exp_fluzoo_ext`
- Data: **CDC ILINet** via Delphi Epidata `fluview`; `data/build_data.py` rebuilds the processed
  arrays + the tracked `data/processed/metadata.json` (URL + SHA256 + epiweek window).
- Run: OpenEvolve `openevolve/slurm_oe_islands.sh`; rigorous re-score
  `openevolve/baseline_test.py`.
- Committed (see `results/MANIFEST.txt`): `baseline_test.json` (the 300-step re-score),
  `_pool_124_125/oe_pool_summary.json` + `oe_island_{0,1,2}/best/` (the OpenEvolve campaign).
- Expected (the fitness-fidelity finding): evolved val 0.01477 / test 0.01812 ties the hand SEIR
  seed 0.01488 / 0.01805 and an SEIRS variant 0.01473 / 0.01856 once re-scored at 300 steps; the
  apparent search-time win (val 0.0100) vanishes. An early small-scale pilot lives in
  `results/smoke_pilot/` and is explicitly **not** the campaign.

---

## 5. Artifact-to-table/figure map

| Manuscript object | Backing artifact | Verifier |
|---|---|---|
| §4.1 / App. A fidelity, `tab:exp_a_results` | `exp_a/results/*.{csv,json}` + `gradient_delta`/`noise_sweep`/`s3_branch` | `make_exp_a_data.py` |
| Tables 5/6, `fig:exp_b_loss` | `exp_b/results_llm/` (300 files) + `llm_cache/` | `aggregate_table.py` (65/65) |
| §4.3 recursive, `tab:exp_c_results` | `exp_c/results/` (LFS) | `aggregate_table.py` (43/43) |
| §4.3 LIM-ENSO, `tab:lim_*` | `exp_lim_enso/results/agg/T1-T5` + data `metadata.json` | `aggregate.py` |
| App. D, `tab:exp_d_timing` | `exp_d/results/*.json` (LFS) | `aggregate_table.py` (10/10) |
| App. E, `tab:exp_e_results` | `exp_e/results/*.json` | `make_exp_e_data.py` |
| App. F, `tab:exp_f` | `exp_f/results*/` | direct JSON read |
| App. G, `tab:exp_g` | `exp_g/results/*.json` | per-label mean over seeds |
| §4.5 / App. H, `tab:exp_h_*` | `exp_h/results/` + SLURM `.out` | `aggregate_table.py` (275/275) |
| App. I, `tab:exp_i` | `exp_i/results/scaling/*` | `aggregate_scaling.py` |
| App. J, `tab:exp_j` | `exp_j/results/expj*_results.json` | direct JSON read |
| §4.4 / App. L, Table 2 + Fig 3 | `exp_battery/results/` (rescore/ablate/funnel/winners/targets) | re-score scripts; `results/MANIFEST.txt` |
| §4.5 / App. K (FluZoo) | `exp_fluzoo/results/baseline_test.json` + `_pool_124_125/` + island bests | `results/MANIFEST.txt` |

---

## 6. Seeds, integrity, and provenance

- Seeds: fixed in each `config.py`, embedded in each result file. The interpreter runs in
  `float32`; equivalences are reported with explicit tolerances.
- Integrity: `python experiments/check_manifests.py`.
- Data provenance: every real dataset ships a tracked `metadata.json` with source URL/DOI,
  SHA256, and preprocessing: `exp_lim_enso/data/processed/metadata.json`,
  `exp_battery/data/metadata.json`, `exp_fluzoo/data/processed/metadata.json`.
- LFS: `.gitattributes` tracks large results for exp_{a,b,c,d,h}; run `git lfs pull` after
  cloning. Other experiments' summaries are plain git.

---

## 7. Datasets (Data Availability)

| Dataset | Used by | Public source | In-repo provenance |
|---|---|---|---|
| NOAA ERSSTv5 monthly SST | LIM-ENSO (§4.3) | NOAA PSL `sst.mnmean.nc` | tracked `metadata.json` + fetch (SHA256) |
| Severson et al. fast-charging Li-ion | Battery (§4.4) | DOI `10.1038/s41560-019-0356-8`, `data.matr.io/1/` | tracked `metadata.json` (SHA256, 117-cell rule); `gen_target_real.py` |
| CDC ILINet (wILI) | FluZoo (App. K) | Delphi Epidata `fluview` | tracked `metadata.json` (SHA256, epiweek window); `build_data.py` |

All other experiments are fully synthetic and require no external data.
