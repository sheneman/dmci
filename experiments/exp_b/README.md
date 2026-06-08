# Experiment B: End-to-End Optimization of LLM-Generated Differentiable Programs  (manuscript §4.2 `sec:exp_b`; Appendix B `app:exp_b_ext`, Table 5 `tab:exp_b_models`, Table 6 `tab:exp_b_results`, Figure `fig:exp_b_loss`)

## What this shows
A locally-hosted LLM (Qwen 3.6-35B) generates Scheme programs from natural-language
descriptions of 15 scientific models; each generated program is compiled - *without
per-model edits*, only a uniform evaluator-wrapper transform - into a differentiable
module (DMCI) and its physical constants are fit end-to-end by Adam. The experiment
establishes that (1) DMCI and direct compilation of the same generated program produce
identical optimization trajectories (75/75 convergence-epoch match, zero final-loss
difference), (2) all three physics-informed methods (DMCI, direct, hand-coded PyTorch)
converge on 100% of runs while a pure MLP converges on only 65/75, and (3) the DMCI
interpretation overhead is ~73x wall-clock vs direct compilation with identical gradient
quality. This supports the central claim that generated programs can be optimized with no
per-program implementation.

## Files
- `config.py` -- `ExpBConfig`: hyperparameters (Adam lr=0.05, max_epochs=3000, convergence threshold=1e-3, 5 seeds, 20 data points, x_range default (0.1, 5.0)).
- `models.py` -- the 15 `ModelSpec`s (M01..M15): NL description, ground-truth Python fn, true/init parameter values, and *hand-authored* `interp_source` (DMCI) / `direct_source` (direct) Scheme. Loads the self-hosted evaluator from `bootstrap/compiler.scm`. Registry: `ALL_MODELS`, `EQUATION_MODELS`, `PROGRAM_MODELS`, `MODEL_BY_NAME`.
- `llm_generate.py` -- live LLM generation (`generate_live`/`generate_all`) against MindRouter/OpenAI/Anthropic, the shared compiler-constrained `SYSTEM_PROMPT`, validation (`_validate`: compiles + correct within 1e-4), and the JSON cache I/O (`load_from_cache`/`_save_to_cache`).
- `llm_sources.py` -- the UNIFORM, model-independent transform that turns each cached LLM program into its DMCI form (`make_interp_source`: program quoted as data fed to `scheme-eval-program`) and direct form (`make_direct_source`). `apply_llm_sources` swaps these into each `ModelSpec` (LLM-cache mode). This is what backs the "compiled without modification" claim.
- `baselines.py` -- the four training methods (`run_dmci`, `run_direct_compiled`, `run_handcoded_pytorch`, `run_pure_mlp`), the shared compiled-graph training loop `_train_compiled`, data generation, seeded `±30%` parameter perturbation, and `TrainResult`. The pure MLP is 2 hidden layers x 64 units, tanh.
- `runner.py` -- `run_single`: runs one (method, model, seed) and writes `<tag>.json` (summary) + `<tag>.csv` (per-epoch trajectory) into the output dir.
- `run_all.py` -- top-level driver: 4 methods x 15 models x 5 seeds = 300 runs; `--use-llm-cache` selects the LLM-program path; `--generate-only` regenerates the cache. Sets `sys.setrecursionlimit(20000)` for deep interpreted recursion.
- `aggregate_table.py` -- canonical aggregator: reconstructs every Table 6 cell, the aggregate wall-clock figures, the 75-pair convergence-epoch match counts, and the `fig:exp_b_loss` coordinates from `results_llm/*.json`, and self-checks them against the values committed in the manuscript (`PAPER_TABLE6`, `PAPER_AGG`, `PAPER_FIG_LOSS`). `--write-summary` refreshes `paper2/figures/data/exp_b_summary.json`.
- `slurm_submit.sh` -- single-job sbatch; runs `run_all` with hand-authored sources into `results/` (the reference run).
- `slurm_array.sh` -- 4-task array, hand-authored sources -> `results/`.
- `slurm_array_llm.sh` -- 4-task array, `--use-llm-cache` -> `results_llm/` (this is the run the manuscript reports).
- `llm_cache/*.json` -- 15 committed LLM generations (one per model), each marked `compiles=true, correct=true`. The reproducibility artifact for the generation step.
- `runner.py`/`config.py`/`models.py`/`llm_*.py` are the load-bearing modules; the slurm scripts and `aggregate_table.py` are the entry points.
- NOTE - scratch/untracked, NOT load-bearing (safe to delete): `_agg_csv_tmp.py`, `_agg_llm_tmp.py`, `compare_llm_rerun.py` (ad-hoc comparison of `results/` vs `results_llm/`); none are referenced by any script, the manuscript, or `reproduce.sh`. `aggregate_table.py` is the canonical aggregator.

## Data
- **Inputs:** Fully synthetic. Training targets are generated in `baselines._generate_data` by evaluating each model's `ground_truth` Python function on a deterministic `torch.linspace` grid over the model's `x_range` (no RNG in the data; the only RNG is the per-seed `±30%` parameter init in `_make_params`, seeded by `torch.manual_seed(seed)`). The LLM-generated programs themselves are the *other* input; they are committed in `llm_cache/*.json` and regenerable via `python3 -m experiments.exp_b.run_all --generate-only --api mindrouter` (requires `MINDROUTER_API_KEY` in `.env`; live regeneration is non-deterministic, hence the committed cache is the source of truth).
- **Outputs:** Two committed result sets, each 300 runs = 300 `.json` (summary: converged, convergence_epoch, final_loss, total_wall_time, param errors, tier, domain) + 300 `.csv` (per-epoch loss/grad_norm/wall_time/param trajectory) + a `MANIFEST.txt`:
  - `results/` -- reference run using the **hand-authored** `interp_source`/`direct_source` in `models.py`.
  - `results_llm/` -- run using the **actual LLM-generated** programs (`--use-llm-cache`). **This is the set the manuscript Table 6 / Figure / aggregates are computed from** (`aggregate_table.RESULTS_DIR = results_llm`).
- **Provenance:** `MANIFEST.txt` present in both `results/` and `results_llm/` (600-file inventories); both verified by `python3 -m experiments.check_manifests` (exp_b: OK / OK). `paper2/figures/data/exp_b_summary.json` is the committed per-model summary the figure data is drawn from. Both result dirs are committed locally (Git/LFS), ~18 MB each including CSVs.

## How to run
```bash
# (Optional) regenerate the LLM programs into llm_cache/ (non-deterministic; needs MINDROUTER_API_KEY):
python3 -m experiments.exp_b.run_all --generate-only --api mindrouter

# Manuscript run (LLM-generated programs) - on HPC (CPU `eight` partition):
sbatch experiments/exp_b/slurm_array_llm.sh        # -> results_llm/  (the run the paper reports)

# Reference run (hand-authored sources):
sbatch experiments/exp_b/slurm_submit.sh           # -> results/
#   or the array form:
sbatch experiments/exp_b/slurm_array.sh            # -> results/

# Local single-model smoke test (fast):
python3 -m experiments.exp_b.run_all --use-llm-cache --models M02_beer_lambert \
    --seeds 1 --max-epochs 200 --output-dir /tmp/exp_b_smoke

# Reproduce Table 6 / Figure / aggregates and self-check vs manuscript:
python3 -m experiments.exp_b.aggregate_table       # prints 65/65 cells match, 60/60 fig coords
```

## Expected results
From Appendix B Table 6 (`tab:exp_b_results`), DMCI, 5 seeds each (all 75 DMCI runs converge):

| Model | Params | Avg Epoch | Avg Loss | Avg Time | MLP Conv |
|-------|-------:|----------:|---------:|---------:|---------:|
| M01 Coulomb | 1 | 556 | 0.0000 | 222 s | 0/5 |
| M02 Beer–Lambert | 1 | 66 | 0.0016 | 21 s | 5/5 |
| M03 Michaelis–Menten | 2 | 1753 | 0.0007 | 378 s | 5/5 |
| M04 Arrhenius | 2 | 308 | 0.0001 | 130 s | 5/5 |
| M05 Damped oscillator | 3 | 135 | 0.0000 | 103 s | 1/5 |
| M06 Logistic growth | 2 | 388 | 0.0001 | 231 s | 5/5 |
| M07 Power law | 2 | 752 | 0.0003 | 232 s | 4/5 |
| M08 Euler ODE | 1 | 23 | 0.0030 | 241 s | 5/5 |
| M09 Taylor e^ax | 1 | 165 | 0.4803 | 3127 s | 5/5 |
| M10 SiLU | 2 | 248 | 0.0004 | 160 s | 5/5 |
| M11 Recursive filter | 1 | 17 | 0.0002 | 257 s | 5/5 |
| M12 Newton sqrt | 1 | 0 | 0.0000 | 203 s | 5/5 |
| M13 Composed transforms | 2 | 356 | 0.0003 | 215 s | 5/5 |
| M14 Anomaly scorer | 3 | 100 | 0.0002 | 182 s | 5/5 |
| M15 Horner polynomial | 4 | 2128 | 0.0010 | 1992 s | 5/5 |

Aggregate claims (all reproduced exactly by `aggregate_table.py`):
- DMCI = direct convergence-epoch match: **75/75**, max |final-loss diff| = **0.00**.
- DMCI = hand-coded match: **73/75** (the 2 mismatches both on M11, differing 6–8 epochs).
- Convergence: physics-informed methods **75/75 (100%)**; pure MLP **65/75 (87%)** (0/5 on M01, 1/5 on M05, 4/5 on M07).
- MLP-vs-DMCI mean-loss ratio: **50.3x** (M02), **~4,100x** (M14, computed 4063.8x), **4.4x** (M11).
- Wall-clock: DMCI **513 s** mean vs direct **7.0 s** (**73x**) vs hand-coded **2.1 s** (**248x**).
- Figure `fig:exp_b_loss`: 60/60 mean-final-loss coordinates match within 2%.
- Generation: Qwen 3.6-35B produced compilable+correct Scheme for **15/15** models on the final prompt (`llm_cache/*.json`, all `compiles=true, correct=true`).

## Environment, seeds, hardware
- **Python:** 3.11.10 (HPC `module load python/3.11.10` matches the project `.venv`); PyTorch (autograd) + `neural_compiler` (this repo) + `openai`/`dotenv` for generation. No GPU required (CPU autograd).
- **Seeds:** 5 fixed seeds (0–4) per (method, model); `torch.manual_seed(seed)` drives both the ±30% Gaussian parameter perturbation and MLP weight init. Data grid is deterministic (no RNG).
- **Hardware / cost:** Run on HPC `eight` CPU partition (`fortyfive.hpc.uidaho.edu`), 4 cpus-per-task, 32 GB, ≤24 h walltime per array task. Total DMCI cost dominated by the deep-recursion models: M09 Taylor ~3127 s/run, M15 Horner ~1992 s/run; simple equations ~21 s/run (M02). All 300 DMCI+direct+hand-coded+MLP runs fit in the array's 24 h budget. The single-model local smoke test (M02, 1 seed, 200 epochs) runs in a few seconds.
