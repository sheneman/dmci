# Experiment I: Differentiable Calibration of a Composite Ecosystem Model (parameter-dimension scaling)  (manuscript Appendix `app:exp_i`, "Experiment I: Differentiable Calibration of a Composite Ecosystem Model"; Table `tab:exp_i`, Figure `fig:exp_i_heldout`)

## What this shows
On a realistic multi-parameter *composite* GPP / ecosystem model (free-parameter count
grows with the number of plant functional types), exact-gradient calibration through the
compiled DMCI interpreter degrades the **most gracefully with parameter dimension**: at an
equal 300 s wall-clock budget, single-start Adam through DMCI holds a narrow held-out-MSE
band (1.7e-2 to 1.5e-1) and is the best method at every d >= 18, while gradient-free
differential evolution wins only at d <= 12 and erodes thereafter, and a curvature-aware
multi-start L-BFGS fitter is erratic / diverges at high d under a fixed budget. This is the
quantitative core of the correctness-vs-optimization-success limitation (manuscript
`sec:tradeoffs`) and the dimension-scaling evidence cited by the Experiment-D/F/H
inner-fitter ablation (manuscript "What did the interpreter contribute?", and the
"degrade most gracefully" claim).

## Files
- `config.py` -- drivers (Q, T, psi), driver ranges, fit hyperparameters (lr 0.02, max 2000 epochs, conv. threshold 1e-3, 3 seeds), and the (now-historical) pilot go/no-go thresholds.
- `models.py` -- FATES-anchored composite-GPP `ModelSpec` builder (`build_static_model(n_pft)`): emits the DMCI `interp_source` (bootstrap evaluator + GPP expression as quoted data), the matching `direct_source`, the Python `ground_truth`, per-PFT synthetic targets, and the pre-registered black-box `PARAM_BOUNDS`. Also the optional recursive carbon-pool model (used only to show `lambdify` rejects recursion).
- `harness.py` -- the fit loop and all optimizers: `run_dmci` (single-start batched Adam), `run_lbfgs_multistart` (log-reparam + strong-Wolfe L-BFGS + multi-start through the batched graph), `run_diffevo` (scipy differential evolution on the transcribed Python forward model), `generate_data` (seeded stratified driver sample), `heldout_mse` (primary metric, fresh driver split at seed+10000), and the `load_ameriflux` real-data loader **stub**.
- `run_comparison.py` -- the d-scaling driver: for each `--param-counts` (n_pft) and seed, runs the three optimizers at an equal `--budget-s` and writes `results/scaling/comparison_<method>[_<out-tag>].json`.
- `aggregate_scaling.py` -- merges the per-task `comparison_*_pft*.json` into `results/scaling/scaling_combined.json` and the pgfplots `results/scaling/exp_i_scaling.dat` (mean held-out MSE vs d per method) that the manuscript figure consumes.
- `smoke_scaling.py` -- high-d feasibility probe (times compile + one batched Adam epoch at d=96/126) to confirm the SLURM wall-clock before launching the array.
- `run_pilot.py` -- the original `<2 h` go/no-go pilot (1 static-GPP family, 6-criteria verdict); superseded by the d-scaling sweep for the manuscript but retained.
- `lambdify_baseline.py` -- `sexp -> sympy -> jax.grad` comparator (the "why not just autodiff?" rebuttal): parity on closed-form GPP, hard rejection on the recursive carbon pool (engineering-cost delta referenced in `app:exp_j`).
- `slurm_scaling.sh` -- **canonical** SLURM job array (`--array=0-8`, one task per n_pft in {1,2,3,4,6,8,11,16,21}); CPU `eight` partition; activates the self-contained `.venv` (no module load).
- `slurm_comparison.sh` -- single-job variant of the sweep (NOTE: still has a stale `module load python/3.11.10` line; prefer `slurm_scaling.sh`).
- `slurm_pilot.sh`, `slurm_smoke.sh` -- SLURM wrappers for the pilot and the smoke probe.

## Data
- **Inputs:** *Synthetic*, fully regenerable. Drivers (Q, T, psi) are seeded stratified samples (`harness.generate_data`, `torch.Generator().manual_seed(seed)`); ground-truth targets are deterministic per PFT (`models._pft_target`, first 5 hand-set, k>=5 drawn from `random.Random(1000+k)` inside the pre-registered `PARAM_BOUNDS`); responses are noiseless (`noise_std=0`). No external dataset is required for the manuscript result. A real-data path (AmeriFlux US-Ha1 BASE light-response) exists only as a **stub** (`harness.load_ameriflux`, `AMERIFLUX_NOTE`): CSV would go at `experiments/exp_i/data/US-Ha1_BASE.csv` (CC-BY-4.0, portal registration + DOI); `data/` is currently empty and the loader raises with acquisition instructions. The AmeriFlux fit is NOT part of any reported manuscript number.
- **Outputs (committed in `results/`):**
  - `results/scaling/comparison_dmci_pft{1,2,3,4,6,8,11,16,21}.json` -- one per d (=6*n_pft), each with 3 seeds x 3 methods; per cell: `best_mse`, `heldout_mse`, `mean_param_rel_error`, `t_fit_s`, `n_iters`.
  - `results/scaling/scaling_combined.json` -- all rows merged.
  - `results/scaling/exp_i_scaling.dat` -- mean held-out MSE vs d per method; **byte-identical** (values) to the manuscript figure data `paper-arxiv/figures/exp_i_heldout.dat` (verified; only the header row differs).
  - `results/comparison_dmci.json` -- a superseded earlier partial run (d=6..24 only); not used by the manuscript.
- **Provenance:** No `metadata.json` / `MANIFEST.txt` is present. Provenance is instead: (a) committed inputs are regenerable from the fixed seeds above; (b) the table/figure values recompute exactly from the committed per-seed JSON via `aggregate_scaling.py`; (c) run host/date are in the (gitignored) SLURM logs under `logs/`. **Recommended addition:** a `results/scaling/MANIFEST.txt` recording git SHA, host, `python3 --version`, torch/scipy/jax versions, budget, and the exp_i_scaling.dat checksum.

## How to run
```bash
# Full d-scaling sweep (canonical), HPC, CPU 'eight' partition, ~300s/method budget:
sbatch experiments/exp_i/slurm_scaling.sh 300 dmci      # array task per n_pft

# Or a single n_pft directly (e.g. d=48):
python3 -u -m experiments.exp_i.run_comparison \
    --method dmci --budget-s 300 --param-counts 8 --seeds 0 1 2 \
    --out-tag pft8 --output-dir experiments/exp_i/results/scaling

# Feasibility probe before launching the array:
python3 -m experiments.exp_i.smoke_scaling

# Aggregate the array outputs into the combined JSON + manuscript .dat:
python3 -m experiments.exp_i.aggregate_scaling
# then copy results/scaling/exp_i_scaling.dat -> paper-arxiv/figures/exp_i_heldout.dat
# (rename header columns "d dmci dmci_lbfgs_ms diffevo" -> "nparams dmci lbfgs diffevo"
#  to match the pgfplots y= keys; values are unchanged).
```

## Expected results
Mean held-out MSE by parameter count d (3 seeds, equal 300 s/method budget) -- Table
`tab:exp_i` / Figure `fig:exp_i_heldout`; bold = best per row in the manuscript:

| d   | dmci (Adam) | dmci_lbfgs_ms | diffevo  |
|-----|-------------|---------------|----------|
| 6   | 2.5e-2      | 1.8e-12       | **4.5e-14** |
| 12  | 4.6e-2      | 5.5e-2        | **4.6e-3**  |
| 18  | **1.7e-2**  | 4.2e-2        | 4.0e-2   |
| 24  | **4.8e-2**  | 2.7e-1        | 5.7e-2   |
| 36  | **1.5e-1**  | 2.7e-1        | 4.9e-1   |
| 48  | **1.5e-1**  | 2.8e0         | 2.9e-1   |
| 66  | **8.4e-2**  | 5.2e-1        | 3.5e-1   |
| 96  | **7.4e-2**  | 2.0e0         | 1.2e-1   |
| 126 | 1.4e-1      | **1.4e-1**    | 2.0e-1   |

Key claims: DMCI-Adam best at every d >= 18; diffevo wins only at d <= 12; multi-start
L-BFGS diverges at high d (2.8e0 at d=48, 2.0e0 at d=96). No method reaches the 1e-3
convergence threshold beyond d=6, so the result is a **relative** robustness comparison.
These nine rows are the only manuscript numbers this experiment populates.

## Environment, seeds, hardware
- Python 3.11.10 in the repo's self-contained `uv` `.venv` (torch, scipy, numpy, jax); no
  conda/module load on compute nodes (the `python/3.11.x` modulefile is absent on n0xx --
  `slurm_scaling.sh` reflects this; `slurm_comparison.sh` still has a stale `module load`).
- Seeds: 0, 1, 2 per (d, method). Data, targets, param inits, and DE/L-BFGS restarts are
  all derived deterministically from these (see `generate_data`, `_pft_target`,
  `_make_params`, `_random_init`); held-out split uses seed+10000.
- Hardware: HPC CPU `eight` partition (scalar interpreter favors CPU), 4 cpus-per-task,
  16 GB, run as a 9-task array. **Never run on the local Mac.**
- Wall-clock: equal 300 s/method budget per (d, seed). Observed `t_fit_s`: DMCI-Adam up to
  ~303 s, diffevo up to ~332 s, `dmci_lbfgs_ms` up to ~651 s (the budget is checked
  *between* L-BFGS restarts, so a single 200-iter strong-Wolfe restart can overshoot --
  a minor fairness caveat that does not affect the qualitative ranking). Whole array
  completes within the 4 h SLURM limit per task.
