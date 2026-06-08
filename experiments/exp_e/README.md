# Experiment E: Discrete-Continuous Operator Recovery  (manuscript Appendix `app:exp_e`, Table `tab:exp_e_results`, Figure `fig:exp_e_success`)

## What this shows
DMCI's Gumbel-Softmax soft-dispatch can perform **joint discrete-continuous program search** -- recovering two unknown operators *and* a continuous constant from noisy samples -- but gradient-based discrete search is hard. On a 64-combination operator space, DMCI recovers the correct operator pair in only **10.8%** of restarts (above random's 2.1%, far below exhaustive's 100% and evolutionary's 82.9%), confirming the manuscript's positioning that DMCI's strength is *continuous* parameter optimization, not discrete structure search. Where DMCI *does* recover the structure, it achieves the **best continuous-constant precision** of any method (mean `|a-a*| = 3.7e-4`, 2.2x better than exhaustive least-squares). This grounds the correctness-vs-optimization distinction and the Gumbel-Softmax relaxation referenced in the main text (Sections `sec:related`, `sec:exp_battery`, `sec:tradeoffs`).

## Files
- `exp_e1.py` -- main driver. Builds the soft-choice Scheme template `(* alpha (soft-choice (...outer...) w1))` with a nested inner `soft-choice` for the eight inner operators; implements all four methods: DMCI soft-dispatch (straight-through Gumbel-Softmax with temperature annealing + derivative-matching loss), exhaustive enumeration (64 least-squares fits), an elitist GA, and random search. Also contains `run_task`, `aggregate`, and the CLI.
- `config.py` -- experiment config: the 12 target tasks `TASKS` (op1_idx, op2_idx, a_star, description), the operator vocabulary (`OPERATORS`/`OP_LABELS`, 8 ops), and `ExpE1Config` (hyperparameters: 20 restarts, 3000 epochs, lr 0.05, tau 1.0->0.1, 64 data points, noise_std 0.01, deriv_weight 1.0, GA pop 32 / 100 gens / tournament k=3).
- `slurm_submit_e1.sh` -- SLURM array job (`--array=0-11`, partition `eight`, 4 CPUs, 16G, 12h) running one task per array index. NOTE: `reproduce.sh` line 28 refers to this as `slurm_submit.sh`; the actual filename is `slurm_submit_e1.sh`.
- `__init__.py` -- package marker (empty).
- `results/exp_e1_T01.json` ... `exp_e1_T12.json` -- per-task outputs (committed). See Data below.
- `../../paper-arxiv/figures/make_exp_e_data.py` -- regenerates the pgfplots `.dat` files for Table/Figure from `results/`.

## Data
- **Inputs:** Fully synthetic, no external dataset. Generated in-process by `generate_data()` in `exp_e1.py` from each target `f*(x) = a* . op1(x, op2(x))`. Data is deterministic per restart: `torch.manual_seed(seed)` with `data_seed = task_idx*1000 + restart`; 64 points half on `[0.2, 3.0]` and half mirrored negative, plus 1% Gaussian noise (`noise_std=0.01`); derivative targets `dys` from autograd. DMCI restart seed = `task_idx*10000 + restart`; random/exhaustive/evolutionary seeds offset by 20000/30000/40000. No raw-data files -- inputs are regenerable from seeds alone.
- **Outputs:** `results/exp_e1_T{01..12}.json` (committed, ~65 KB each). Each contains `summary` (per-method success counts/rates, mean const error on successes, mean wall time), `config` (the full `ExpE1Config` snapshot used), and full per-restart records for all four methods (`dmci_restarts`, `random_restarts`, `exhaustive_restarts`, `evolutionary_restarts`) with selected operators, fitted `alpha`, final loss, `op1_correct`/`op2_correct`/`both_correct`, `const_error`, wall time, and (for DMCI) softmax probabilities and logits.
- **Provenance:** No `metadata.json` or `MANIFEST.txt` present. Provenance is instead carried inside each JSON via the embedded `config` block (verified to match `config.py`: 3000 epochs, lr 0.05, tau 1.0->0.1, 20 restarts, GA pop 32/100 gens). The committed `.dat` files under `paper-arxiv/figures/` are byte-consistent with these results. A `metadata.json` recording git commit, hostname, torch version, and run date *should* be added for a stricter artifact, since the current JSONs do not record those.

## How to run
```
# All 12 tasks on HPC (one SLURM array index per task, 20 restarts each):
sbatch experiments/exp_e/slurm_submit_e1.sh

# Single task locally / interactively (task index 0-11):
python3 -m experiments.exp_e.exp_e1 --task 0 --output-dir experiments/exp_e/results

# All 12 tasks in one process:
python3 -m experiments.exp_e.exp_e1 --task all --output-dir experiments/exp_e/results

# Re-print the aggregate table from existing results:
python3 -m experiments.exp_e.exp_e1 --aggregate-only --output-dir experiments/exp_e/results

# Regenerate the manuscript figure/table data files:
python3 paper-arxiv/figures/make_exp_e_data.py
```

## Expected results
Populates **Table `tab:exp_e_results`** and **Figure `fig:exp_e_success`**. Aggregate over 240 restarts (12 targets x 20), reproduced exactly from the committed `results/`:
- Discrete recovery success: **DMCI 10.8%** (26/240), **Exhaustive 100%** (240/240), **Evolutionary 82.9%** (199/240), **Random 2.1%** (5/240).
- Continuous precision on DMCI's 26 correct restarts: mean `|a-a*| = 3.7e-4`, median `1.6e-4`; vs exhaustive `8.3e-4` (2.2x better). Over all 240 restarts DMCI mean `|a-a*| = 1.11` (dominated by the 214 wrong-structure fits).
- Per-target DMCI success rates 5-25%, evolutionary 20-100%, random 0-15% (all twelve verified against the table values and `exp_e1_success_rates.dat`).
- Wall-clock (manuscript): DMCI ~689 s/restart, exhaustive ~0.012 s, evolutionary ~0.57 s; DMCI ~57,000x slower than exhaustive. (Committed DMCI per-restart times are ~675-714 s.)

## Environment, seeds, hardware
- Python 3.11, PyTorch 2.5.1 (local box; HPC `.venv` per project convention). CPU-only (partition `eight`, no GPU needed).
- Seeds: all deterministic and derived from `(task_idx, restart)` as above; no wall-clock or RNG-state nondeterminism in data/restart selection.
- Hardware: results in `results/` were produced on the HPC `eight` CPU partition (SLURM array, 4 CPUs/16G per task).
- Approximate cost: DMCI dominates at ~689 s per restart x 240 = roughly 46 CPU-hours total for DMCI; the three baselines are negligible (<3 minutes combined). A full single-task run (20 DMCI restarts) is ~3.8 hours, hence the 12-way array. Local single-task reproduction is feasible but slow; reduce `--n-epochs` for a smoke test (full numbers require the 3000-epoch setting).
