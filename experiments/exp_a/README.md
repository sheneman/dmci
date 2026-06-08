# Experiment A: Does Interpretation Preserve Gradients?  (manuscript §4.1 `sec:exp_a`, Appendix `app:exp_a_ext`)

## What this shows
Optimizing the learnable constants of six Scheme program families (scalar arithmetic
through three-parameter nested composition) via the Differentiable Meta-Circular
Interpreter (DMCI) produces **trajectories indistinguishable from direct compilation
and from a hand-coded PyTorch interpreter**: the three autograd methods agree on
convergence epoch and final loss to within `7e-7`, and at random parameter settings
the relative gradient error is `0` and cosine similarity is `1.0` to numerical
precision. This is the empirical backbone of the paper's central claim that tagged-value
dispatch, dict-heap indirection, environment lookup, and evaluator recursion do not
measurably degrade gradient fidelity (Theorem `thm:gradient`). Supporting material
covers the ~14x per-evaluation overhead decomposition, noise robustness, and the S3
branch-dependent-constant boundary case (a parameter feeding *only* a branch condition
receives zero gradient - inherited from the source program, not introduced by DMCI).

## Files
- `config.py` -- frozen `ExpAConfig`: 10 seeds (0–9), max 3000 epochs, conv. threshold `1e-3`, Adam lr `0.05`, 8-point grid on `[0.5, 3.0]`, 5 methods.
- `programs.py` -- the six program families P1–P6 (`interp_source` for DMCI, `direct_source` for direct compilation, `hc_expr`/`hc_program` AST for the hand-coded interpreter, `data_fn` synthetic-target generator, targets, and seeded inits).
- `baselines.py` -- the five optimization methods: `direct`, `compiled_interp` (DMCI), `handcoded_interp`, `finite_diff` (central differences), `evolution_strategy` ((μ,λ)-ES). Defines `run_method`, `TrainResult`, and the pure-Python `HandCodedInterpreter`.
- `runner.py` -- `run_single` executes one (method, program, seed); writes per-run `<method>_<program>_<NN>.csv` (per-epoch loss/params/grad-norm/wall-time) and `.json` summary.
- `run_all.py` -- top-level driver. Builds the 300-run matrix (5×6×10), supports `--slurm-task-id`, `--methods`, `--programs`, `--seeds`, `--skip-existing`, `--ablations-only`.
- `analyze.py` -- aggregates per-run JSONs into convergence/param-error LaTeX tables and mean±σ loss curves. (Note: only skips `ablations.json`; will KeyError if the auxiliary `gradient_delta.json`/`noise_sweep.json`/`s3_*.json` files are present in the same dir - run it on a method-results-only view, or it is superseded by `make_exp_a_data.py` for the figures.)
- `plot.py` -- matplotlib paper figures (uses `analyze.py` helpers).
- `ablations.py` -- the three ablations: dict-vs-tensor heap, lazy-vs-eager evaluation, and gradient-path autograd-node count per program. (Run via `run_all.py --ablations-only`.)
- `gradient_delta.py` -- gradient-fidelity study: at 50 random parameter settings per program, compare DMCI vs direct gradients (`relative_error`, `cosine_similarity`). Produces `results/gradient_delta.json`.
- `noise_sweep.py` -- noise-robustness sweep (review item R4): re-runs constant recovery through DMCI on additive-Gaussian-corrupted targets for σ∈{0,0.02,0.05,0.10,0.20}. Produces `results/noise_sweep.json`.
- `exp_s3_branch.py` -- S3 branch-dependent-constant case: `train` mode (direct/DMCI, 10 seeds) and `basin` mode (21-point α sweep). Produces `results/s3_branch_*.json`.
- `profile_decomposition.py` -- cProfile-based decomposition of the DMCI-vs-direct per-evaluation overhead on P2, 100 iters. Produces `results/profile_decomposition.txt`.
- `slurm_submit.sh` -- serial full run (all 300 + ablations) on GPU node n128.
- `slurm_array.sh` -- 8-task array splitting the 300 runs by method across the `eight` CPU partition.
- `slurm_gradient_delta.sh`, `slurm_noise.sh`, `slurm_profile.sh`, `slurm_s3_branch.sh` -- submission scripts for the auxiliary studies.

## Data
- **Inputs:** Fully **synthetic, no external dataset**. For each program the 8 inputs are `torch.linspace(0.5, 3.0, 8)` and the targets are the *noiseless* outputs of the same program at its true constants, computed by the `data_fn` lambda in `programs.py` (e.g. P1 target = `0.5·x²`, P6 = `2·(x+0.5)²+1`). Initial constants are produced by `_make_params` in `baselines.py` via seeded Gaussian perturbation (±30%, `torch.manual_seed(seed)`). The noise sweep adds `σ·std(y)·N(0,1)` (`noise_sweep.py`). No fetch/download step is needed - everything regenerates from the seed.
- **Outputs (in `results/`):**
  - `<method>_<program>_<NN>.csv` / `.json` - 300 per-run files (5 methods × 6 programs × 10 seeds). CSV: per-epoch `epoch, loss, param_*_value, param_*_error, grad_norm, wall_time_s`. JSON: `converged, convergence_epoch, final_loss, final_param_errors, total_wall_time`.
  - `gradient_delta.json` - per-program gradient relative error and cosine similarity over 50 samples (all `0.0` / `1.0`).
  - `noise_sweep.json` - `cells` of `(program, sigma, mean_rel_param_err, …)` for P1–P5.
  - `s3_branch_train_{direct,dmci}{,_summary}.json` - S3 10-seed training (both 0/10 converged, final loss `3.9336`, bit-identical params across methods).
  - `s3_branch_basin{,_summary}.json` - 21-point α₀ sweep (`final_alpha == initial_alpha` everywhere; minimum at α*=1, loss 0).
  - `profile_decomposition.txt` - the overhead table (Total 13.8×).
  - `old_buggy_s3/` - **superseded** earlier S3 outputs; not used by the manuscript. Ignore; current S3 files live at the top of `results/`.
- **Provenance:** `results/MANIFEST.txt` lists every committed file (617; verify with `python3 -m experiments.check_manifests`). Raw outputs are tracked via **Git LFS** (`*.csv`/`*.json` under `results/` carry the `lfs` filter). There is **no per-run `metadata.json`** capturing the exact interpreter version / torch version per file; `profile_decomposition.txt` records its own env header (Python 3.8.13, torch 1.13.1+cu117), which predates the repo's current `torch>=2.0` requirement - re-run on the target env if exact wall-clock reproduction matters (the numerical claims are env-independent). The committed `results/ablations/ablations.json` backs the gradient-path node-count claim (9–19 direct, 18–30 interp; 5–18 extra nodes) and the dict-vs-tensor-heap / lazy-vs-eager ablation outcomes; regenerate it via `--ablations-only`.

## How to run
```bash
# Full 300-run sweep + ablations (GPU node, serial; graph cache shared across seeds)
sbatch experiments/exp_a/slurm_submit.sh

# Or split across the CPU partition (8-task array, by method)
sbatch experiments/exp_a/slurm_array.sh

# Auxiliary studies (Appendix)
sbatch experiments/exp_a/slurm_gradient_delta.sh   # -> results/gradient_delta.json
sbatch experiments/exp_a/slurm_noise.sh            # -> results/noise_sweep.json
sbatch experiments/exp_a/slurm_s3_branch.sh        # -> results/s3_branch_*.json (3-task array)
sbatch experiments/exp_a/slurm_profile.sh          # -> results/profile_decomposition.txt

# Local equivalents (from repo root, inside .venv)
python3 -m experiments.exp_a.run_all --output-dir experiments/exp_a/results --skip-existing
python3 -m experiments.exp_a.run_all --ablations-only          # -> results/ablations/ablations.json (committed)
python3 -m experiments.exp_a.gradient_delta --n-samples 50
python3 -m experiments.exp_a.noise_sweep --programs P1_single_const P2_multi_const P3_recursive P4_higher_order P5_multi_function --sigmas 0.0 0.02 0.05 0.10 0.20 --seeds 5 --max-epochs 1200 --patience 100 --output experiments/exp_a/results/noise_sweep.json
python3 -m experiments.exp_a.exp_s3_branch --mode train --method dmci --output-dir experiments/exp_a/results
python3 -m experiments.exp_a.profile_decomposition --n-iters 100

# Regenerate the paper's figure .dat files from results/*.csv
python3 paper-arxiv/figures/make_exp_a_data.py
```

## Expected results
- **Trajectory equivalence (§4.1, Fig. `fig:exp_a_convergence`):** DMCI, direct, and hand-coded agree on convergence epoch and final loss to within `7e-7` (verified: max |DMCI−direct| final-loss diff `2.0e-7`, |DMCI−handcoded| `6.9e-7`). Gradient relative error `0`, cosine similarity `1.0` within `3e-7` (`gradient_delta.json`).
- **Per-program DMCI summary (Table `tab:exp_a_results`), best-checkpoint values:**
  | Program | Conv | Avg Epoch | Train Loss | Avg Time | Param Error |
  |---|---|---|---|---|---|
  | P1 single const | 10/10 | 20 | `1.5e-4` | 2.4 s | α: 0.0006 |
  | P2 multi const | 10/10 | 124 | `2.7e-6` | 9.0 s | a: 0.0002, b: <1e-4 |
  | P3 recursive | 10/10 | 61 | `4.0e-5` | 58.6 s | α: 0.0002 |
  | P4 higher-order | 10/10 | 48 | `1.1e-5` | 10.7 s | α: 0.0005 |
  | P5 multi-function | 10/10 | 247 | `2.0e-4` | 31.6 s | a: 0.005, b: 0.009 |
  | P6 composed | 10/10 | 2017 | `8.5e-4` | 210.3 s | a: 0.014, b: 0.010, c: 0.067 |
- **Baselines (§4.1, Appendix `app:exp_a_full`):** all three autograd methods 60/60 (100%); finite differences 50/60 (83%) at 12.5–534× wall time; (μ,λ)-ES 27/60 (45%), failing on P2/P6 (0/10) and P5 (1/10).
- **Overhead (Table `tab:overhead_decomp`):** DMCI 13.8× direct per-evaluation; tagged-value wrap/unwrap 41.4%, Python overhead 32.1%, graph walking 16.9%.
- **Noise (Table `tab:exp_a_noise`, Fig. `fig:exp_a_noise`):** geomean rel. param error `1.3e-6` (σ=0) → `1.9e-2` (σ=0.10) → `3.3e-2` (σ=0.20); P5 most sensitive (27.8% at σ=0.20).
- **S3 (Appendix `app:s3`, Fig. `fig:s3_basin`):** direct and DMCI both 0/10, final loss `3.93`, zero loss difference; α stays at its init; basin minimum at α*=1 (loss 0), plateaus ≈4 (below) and ≈139 (above).

## Environment, seeds, hardware
- **Python/Torch:** repo requires `torch>=2.0`, `numpy>=1.24` (`requirements.txt` / `pyproject.toml`); committed `profile_decomposition.txt` was produced under Python 3.8.13 / torch 1.13.1+cu117 (older env - re-run on target env for exact timings).
- **Seeds:** 10 fixed seeds `(0…9)` per (method, program); gradient_delta uses seeds offset at 10000; noise sweep uses 5 seeds. Inits are fully determined by `torch.manual_seed(seed)` in `_make_params`.
- **Hardware:** designed for HPC (`fortyfive.hpc.uidaho.edu`): `slurm_submit.sh` targets GPU node `n128`; the auxiliary/array scripts target the `eight` CPU partition. Runs are CPU-bound (scalar autograd through the interpreter) and reproduce on CPU; results were generated on HPC, not locally.
- **Wall-clock (per Table `tab:exp_a_results`, DMCI training runs):** ~2.4 s (P1) to ~210 s (P6) per run; the full 300-run sweep plus the slow finite-diff/ES baselines is the dominant cost (finite differences alone run up to 534× a direct run). Auxiliary studies: gradient_delta ~31 s total; profile ~minutes; S3/noise within their SLURM time limits (≤8 h walls requested).
