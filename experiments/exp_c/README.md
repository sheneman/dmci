# Experiment C: Recursive Scientific Models  (manuscript §4.3 `sec:exp_c`, Appendix `app:exp_c_ext`)

## What this shows
Gradient fidelity through the compiled self-hosted interpreter extends to inherently
recursive scientific programs (coupled ODEs, iterated maps, recursive filters) with up to
30 state-threading steps: DMCI (program-as-data through the compiled interpreter) and direct
compilation match on convergence epoch for all 36/36 (model, seed) pairs and are bit-for-bit
identical in loss at every epoch and seed, while DMCI converges on 36/36 runs. The experiment
also shows that recursive structure is a strong inductive bias versus a structure-free MLP
(e.g. cascaded EMA, C08: 1,102x mean final-loss gap; decay chain, C03: 42x), with two
smooth-target models (C05 continued fraction, C06 damped pendulum) favoring the MLP.
This is the recursive-program half of §4.3; the real-data half of §4.3 (LIM-ENSO) lives in
`experiments/exp_lim_enso/` and Appendix `app:exp_lim_enso_ext`, not here.

## Files
- `models.py` -- the eight recursive ModelSpecs (C01-C08). Each carries `interp_source` (self-hosted
  evaluator `bootstrap/compiler.scm` + the program as quoted Scheme data, for DMCI), `direct_source`
  (the same recursive program compiled directly), a NumPy/`math` `ground_truth` matching the Euler/
  iteration logic exactly, target/init parameter values, and the input sampling range `x_range`.
- `baselines.py` -- the four methods (`dmci`, `direct_compiled`, `handcoded_pytorch`, `pure_mlp`),
  the shared compiled-graph training loop `_train_compiled`, the hand-coded PyTorch loop, the
  `SimpleMLP` (2 hidden layers, width 64, Tanh), data generation, seeding, and the `TrainResult` record.
- `config.py` -- `ExpCConfig` defaults: `max_epochs=3000`, `convergence_threshold=1e-3`, `lr=0.05`,
  `n_seeds=5`, `n_data_points=20`.
- `runner.py` -- `run_single(method, model, seed, cfg, output_dir)`: runs one (method, model, seed)
  and writes the per-run `.csv` (full loss/grad-norm/param trajectory) and `.json` (summary) to `results/`.
- `run_all.py` -- top-level driver / CLI (sets `sys.setrecursionlimit(5000)`); iterates methods x models
  x seeds with `--skip-existing`, `--model`, `--models`, `--categories`, `--methods`, `--seeds` flags.
- `aggregate_table.py` -- rebuilds Table `tab:exp_c_results` from `results/*.json` and self-checks every
  cell plus the cross-method equivalence claims against the values hard-coded from the manuscript
  (prints `SELF-CHECK vs manuscript: 43/43 cells match`).
- `slurm_array.sh` -- SLURM array (one task per model, `--array=0-7`, partition `eight`, 24h walltime).
- `slurm_submit.sh` -- single-node SLURM submission running all models sequentially.
- `__init__.py` -- package marker.
- The C08 loss-trajectory figure (`fig:c08_loss`) is generated outside this dir by
  `paper-arxiv/figures/make_c08_data.py`, which reads `paper-arxiv/figures/data/*_C08_cascaded_ema_*.csv`
  (verified-identical copies of the C08 CSVs in `results/`); see `paper-arxiv/figures/README.md`.

## Data
- **Inputs:** Fully synthetic, no external dataset. Each model is fit to 20 (input, target) points;
  targets are produced by the model's own `ground_truth` Python function (in `models.py`) at the true
  parameter values, with inputs sampled by `torch.linspace` over that model's `x_range`
  (e.g. `[2.0, 15.0]` for C01, `[0.1, 1.5]` for C06; the two 2-input models sample a grid). Generated
  deterministically by `baselines._generate_data`; no random input noise. Parameter initialization is
  seeded per run (`baselines._make_params` -> `torch.manual_seed(seed)`), so all data and inits are
  exactly regenerable from `models.py` + the seed. The DMCI program source embeds `bootstrap/compiler.scm`
  (the self-hosted evaluator) read at import time.
- **Outputs:** `results/` holds 288 files = 144 `.json` + 144 `.csv` (4 methods x 36 (model, seed) pairs;
  C01 and C06 use 3 seeds, the other six use 5). Each `{method}_{model}_{seed:02d}.json` carries
  `converged`, `convergence_epoch`, `final_loss`, `final_param_errors`, `total_wall_time`, `category`,
  `domain`. Each matching `.csv` carries the full per-epoch trajectory: `epoch, loss, grad_norm,
  wall_time_s`, plus one `<param>_value` column per learnable parameter.
- **Provenance:** `results/MANIFEST.txt` lists all 288 files; verify with
  `python3 -m experiments.check_manifests`. Result `.json`/`.csv` are committed via Git LFS
  (`.gitattributes`: `experiments/exp_c/results/*.{json,csv} filter=lfs ...`); on a fresh clone run
  `git lfs pull` before aggregating. There is no separate `metadata.json`; the run configuration lives
  in `config.py`, and `aggregate_table.py` carries the manuscript's expected values for drift detection.

## How to run
```
# Full experiment (HPC; DMCI C01/C06 take ~7-9.6 h/seed, so prefer the array form):
sbatch experiments/exp_c/slurm_array.sh          # one SLURM task per model (C01..C08)
# or single node, all models sequentially:
sbatch experiments/exp_c/slurm_submit.sh

# Direct invocation (equivalent to what the SLURM scripts call):
python3 -m experiments.exp_c.run_all --output-dir experiments/exp_c/results --skip-existing
# Single model / category / method subsets:
python3 -m experiments.exp_c.run_all --model C08_cascaded_ema --skip-existing
python3 -m experiments.exp_c.run_all --categories coupled_ode iterative

# Reproduce Table tab:exp_c_results and self-check against the manuscript:
python3 -m experiments.exp_c.aggregate_table     # prints "SELF-CHECK vs manuscript: 43/43 cells match"

# Regenerate the C08 figure data (Fig fig:c08_loss):
python3 paper-arxiv/figures/make_c08_data.py
```

## Expected results
From Appendix `app:exp_c_ext`, Table `tab:exp_c_results` (per-model DMCI epoch / loss, MLP epoch / loss,
MLP/DMCI mean-final-loss ratio):

| Model | Depth | DMCI epoch | DMCI loss | MLP epoch | MLP loss | Loss ratio |
|---|---|---|---|---|---|---|
| C01 Lotka-Volterra | 20 | 1410 | 0.0008 | n/a | n/a | n/a (MLP fails all 3 seeds) |
| C02 SIR epidemic | 30 | 5 | 0.0000 | 140 | 0.0002 | 25x |
| C03 Decay chain | 25 | 32 | 0.0061 | 611 | 0.2559 | 42x |
| C04 Logistic map | 10 | 33 | 0.0001 | 70 | 0.0004 | 4x |
| C05 Continued fraction | 8 | 37 | 0.0023 | 123 | 0.0004 | 0.2x |
| C06 Damped pendulum | 20 | 1829 | 0.0009 | 227 | 0.0006 | 0.7x |
| C07 IIR filter | 12 | 49 | 0.0004 | 1339 | 0.0328 | 74x (median ~13x) |
| C08 Cascaded EMA | 10 | 24 | 0.0003 | 939 | 0.3699 | 1102x (median ~2x) |

Cross-method claims (Appendix `app:exp_c_ext`, "Trajectory equivalence through recursion"):
DMCI = direct compilation on 36/36 convergence epochs and bit-identical final loss (max |diff| = 0.00
over 36 pairs); DMCI = hand-coded PyTorch on 35/36, the single mismatch being C08 seed 3 (43 vs 37 epochs).
DMCI / direct / hand-coded each converge on 36/36. The C08 loss trajectory populates Fig `fig:c08_loss`
(DMCI and direct overplot exactly; MLP first reaches loss < 1e-3 at epoch 939 vs 24 for DMCI, ~39x slower).
The C07/C08 mean ratios are inflated by a single post-convergence diverging MLP seed (robust medians ~13x
and ~2x). All of the above are checked by `aggregate_table.py` (43/43 self-check cells).

## Environment, seeds, hardware
- **Python/PyTorch:** repo `.venv` (uv-managed, self-contained; no conda/module on HPC). PyTorch CPU
  (no CUDA needed; runs on the `eight` CPU partition). `sys.setrecursionlimit(5000)` is set in `run_all.py`
  (DMCI through the interpreter recurses deeply for 20-30 step models).
- **Seeds:** `torch.manual_seed(seed)` per run, seeds `0..n-1`; six models use 5 seeds, C01 and C06 use 3
  (their ~1,400-1,800-epoch convergence makes DMCI runs 6-9.6 h each). Both data targets (noise-free) and
  parameter initialization are fully determined by `models.py` + seed.
- **Hardware / wall-clock:** run on HPC `fortyfive` partition `eight` (4 CPUs, 32 GB, 24 h walltime per
  the SLURM scripts). DMCI wall times range from ~250 s (C04, C05) to ~7-9.6 h (C01, C06); direct
  compilation runs the same models in ~2-14 s (most) to ~250 s (C01, C06), a 74-158x per-run overhead.
  The committed `results/` were produced on HPC; `aggregate_table.py` and a short `--max-epochs` smoke
  run reproduce locally on a Mac in seconds.
