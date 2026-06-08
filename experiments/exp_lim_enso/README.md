# Experiment LIM-ENSO - a real-data dynamical-systems MLE as a program, fit by exact DMCI gradients

Fit a **Linear Inverse Model** (LIM / linear-Gaussian state-space model) to tropical
Indo-Pacific SST EOF/PC time series, with the **full Kalman-filter negative log-likelihood
folded THROUGH the DMCI meta-circular interpreter**. The LLM emits the Scheme NLL program
(program-as-data); the interpreter produces **exact gradients** w.r.t. the transition
operator `F` and the noise covariances `Q`, `R`, which an optimizer uses to recover the LIM
by maximum likelihood. We then score the fitted operator on **held-out ENSO forecast skill**
and run the standard **LIM operator diagnostics** (ENSO-mode period/decay, eigen-timescales,
optimal growth, the Green-function reference).

**The headline is CAPABILITY**, not optimizer wall-clock: a real-data dynamical-systems MLE,
expressed as a program the LLM can emit, optimized by exact DMCI gradients, with zero
per-structure transcription. DMCI is interpreter-bound and runs on **CPU** (the scalar walk
over the tagged-tensor heap is *slower* on GPU). **We NEVER frame this as beating dynamax** -
dynamax and the numpy twin are *correctness oracles*; the classical Green-function operator
`G = C(τ)C(0)⁻¹` (Penland & Sardeshmukh 1995) is the *scientific reference forecaster*, never
an NLL competitor and never a speed contest.

---

## Phase order (P0 → P3)

- **P0 - data.** `data/build_data.py` fetches ERSSTv5, masks the tropical Indo-Pacific,
  area-weights by `√cos(lat)`, removes climatology, detrends, 3-mo running-means, and takes
  an economy SVD → `data/processed/` (`pcs.npy [898,20]`, `eofs.npy`, `pc_std.npy`,
  `lat/lon/mask`, `metadata.json`). **Built locally**, then rsync'd to HPC.
- **P1 - core + gate.** Locked, verified core (`config/params/models/reference/gate`). The
  numerical **GO/NO-GO gate** (`gate.py`) re-runs PD / forward-parity / finite-difference-
  gradient checks at the real `(D,T)` and emits `gate_{D}.json {GO: bool}`. **The MLE only
  runs on GO.**
- **P2 - baselines + forecast.** `baselines.py` (the optimizer portfolio over the shared
  DMCI NLL) + `forecast.py` (held-out skill + LIM diagnostics). Driven by `runner.py` /
  `run_all.py`, aggregated by `aggregate.py`. **← this layer.**
- **P3 - LLM program-as-data.** The Scheme NLL source (`models.kalman_lim_src`) is what an
  LLM emits given the op surface + binding contract; the *same* program is reused per
  `(structure,D,T)` with only the bound tensors changing. P3 has the LLM author the
  combine-algebra for non-`S0` structures (`S1/S3/S4/S5`) - never `D×D` literals.

The **logdet fix** (committed): a `logdet` primitive (`torch.slogdet`) replaces `(log (det
S))`. `det S` underflows (~1e-20 at D=20) and hits the `log` clamp, corrupting the log-det
term for D≥15; `logdet` sums `log|LU pivots|` and stays exact (rel ~1e-8 at D=20). `r_floor`
**stays physical at 0.1** - it is the primary float32 PD lever, not a fudge factor.

---

## The `[T,D]` `as_matrix` binding contract

`pcs.npy` has shape **`[T, D_max]`**, **row = time step (month)**, **column = PC index**. For
a fit at dimension `D`:

```python
obs = as_matrix(pcs[:T_train, :D])     # nested basis: slice columns 0..D-1
```

Inside the Scheme program `obs` is that matrix and **`(ref obs k)` gathers ROW `k`** - the
`D`-vector of PCs at month `k`. `k` is the loop counter (data-independent), as the interpreter
requires. PCs are unit-variance normalised; multiply column `j` by `pc_std[j]` to recover the
raw expansion coefficient. The physical SST-anomaly field for a normalised PC state `x` is

```
field[s] = ( Σ_j  x[j] · pc_std[j] · eofs[j,s] ) / √cos(lat[s])
```

(un-normalise the PCs, project through the weighted EOFs, then **un-weight** by `√cos(lat)`).

---

## Narrative: capability + program-as-data

One compiled artifact - the DMCI interpreter - makes an *arbitrary* runtime-emitted dynamical
model differentiable. The LIM Kalman NLL is **the same program** across every `(D, structure)`
cell; only the bound `F/Q/R/obs` tensors change. That is the program-as-data thesis: the LLM
proposes the model *as code*, and DMCI calibrates it with exact gradients and **zero
per-structure transcription**. The optimizer portfolio (`dmci_adam` primary, `lbfgs_multistart`,
`diffevo_batched`) all optimize the **identical compiled objective** - we compare *optimizers*,
not objectives. The scientific pay-off (`forecast.py`) - held-out Nino-3.4 ACC/RMSE vs
persistence / damped-persistence / the Green-function LIM, plus the recovered ENSO mode - is
*pure numpy/scipy on the fitted float matrices*; nothing there touches the interpreter.

---

## Gate-first go/no-go protocol

1. Build/stage data (P0). 2. **Run the gate** at each `D` - it writes `gate_{D}.json` and
exits non-zero unless every requested `D` is **GO** (G1 NLL-finite, G2 PD every step, G3
forward parity vs the float32 twin, G4 autograd == central-FD on the float64 twin, G5
det-underflow margin; G6 batched-bind is non-blocking). 3. **Only on GO** run `run_all`. The
**portfolio winner is selected on held-out VALIDATION forecast skill** (Nino-3.4 ACC at the
headline lead), **NEVER train NLL** - the exp_i overfit lesson ("converged ≠ recovered"): the
model that *forecasts* best wins, not the one that drives the training likelihood lowest.

---

## Local deps + rsync staging

Data is built **locally** (network fetch + SVD), runs go **on HPC** (never the local Mac).
Repo on HPC: `/mnt/ceph/sheneman/src/nncompile`, `uv` `.venv` (self-contained; **no conda, no
`module load`**), **`sbatch` only**, partition `sheneman --nodelist=n128` *or* CPU `eight`
(interpreter-bound → CPU is right). `sys.setrecursionlimit(20000)` is raised **before**
importing `neural_compiler` (handled inside `run_all`/`gate`).

```bash
# stage data + code to HPC
rsync -av experiments/exp_lim_enso/data/processed/ \
    fortyfive.hpc.uidaho.edu:/mnt/ceph/sheneman/src/nncompile/experiments/exp_lim_enso/data/processed/
rsync -av --exclude '__pycache__' --exclude 'results' --exclude 'logs' \
    experiments/exp_lim_enso/ \
    fortyfive.hpc.uidaho.edu:/mnt/ceph/sheneman/src/nncompile/experiments/exp_lim_enso/
```

---

## Exact run commands

```bash
# P0 (LOCAL): build the processed data once
python3 -m experiments.exp_lim_enso.data.build_data

# P1 GATE (HPC): go/no-go at each D (writes gate_{D}.json; non-zero exit unless all GO)
sbatch experiments/exp_lim_enso/slurm_gate.sh         # full DEFAULT.D_list = [6,10,15,20]
sbatch experiments/exp_lim_enso/slurm_gate.sh 10      # a single dimension

# P2 RUN (HPC): full sweep method x D x structure x seed (ONLY after GO).
# slurm_parallel.sh runs the cells with a 16-way process pool on n128 (one fit per core,
# each worker single-threaded; exports OMP/MKL/OPENBLAS_NUM_THREADS=1 to avoid oversubscription).
sbatch experiments/exp_lim_enso/slurm_parallel.sh
sbatch experiments/exp_lim_enso/slurm_parallel.sh --skip-existing          # resume
sbatch experiments/exp_lim_enso/slurm_parallel.sh --D 10                   # single D
sbatch experiments/exp_lim_enso/slurm_parallel.sh --methods dmci_adam --seeds 0 1 2
sbatch experiments/exp_lim_enso/slurm_parallel.sh --workers 32             # more workers (n128 has 64 cores)

# aggregate (anywhere with the results/ dir): tables + .dat + scaling_gate_summary.json
python3 -m experiments.exp_lim_enso.aggregate

# direct module invocation (e.g. an interactive CPU node)
python3 -u -m experiments.exp_lim_enso.run_all --output-dir experiments/exp_lim_enso/results
```

---

## Per-run JSON schema (`results/<tag>.json`)

`tag = {method}_{structure}_D{D}_seed{seed}` (e.g. `dmci_adam_S0_D10_seed0`). `runner.run_single`
also writes a per-iteration `results/<tag>.csv` with columns **`iter,nll,grad_norm,wall_time`**.

```jsonc
{
  "tag": "dmci_adam_S0_D10_seed0",
  "method": "dmci_adam",          // dmci_adam | lbfgs_multistart | diffevo_batched
  "structure": "S0",              // F-assembly variant (Phase 1: S0 dense)
  "D": 10, "seed": 0,
  "T_train": 360, "T_test": 120,
  "final_nll": -512.34,           // train-window NLL (fit objective; NOT the selection metric)
  "converged": true,
  "n_iters": 300,
  "wall_time": 142.7,             // seconds
  "per_step_ms": 49.9,            // wall amortized over n_iters * T interpreter steps
  "k_params": 121,                // params.param_count(D,structure).total
  "aic": 1266.7,                  // 2*k + 2*NLL
  "bic": 1979.3,                  // k*ln(T_train) + 2*NLL
  "heldout_acc":  {"3": 0.71, "6": 0.55, "9": 0.41, "12": 0.28},  // fitted-LIM Nino-3.4 ACC per lead
  "heldout_rmse": {"3": 0.42, "6": 0.58, "9": 0.69, "12": 0.79},  // per lead
  "eig_timescales": [44.1, 12.0, ...],  // per-mode decay (months), sorted desc
  "enso_period_mo": 48.3,         // recovered ENSO-mode oscillation period (months)
  "enso_decay_mo": 9.2,           // recovered ENSO-mode e-folding decay (months)
  "rho_F": 0.94,                  // spectral radius of fitted F (<1 == stable LIM)
  "stable": true,
  "min_detS": 6.8e-08,            // worst-step det(S) at fitted op (float64 twin); PD margin
  "cond_S": 1.46,                 // worst-step cond(S) at fitted op
  "batched": true,               // diffevo_batched only (else null): did the [N,D,D] bind work
  "nan_stall": false,            // did a non-finite NLL/grad force an early stop
  // ---- nested blocks kept verbatim for the manuscript (NOT read by aggregate) ----
  "fitted": { "F": [[...]], "Q": [[...]], "R": [[...]] },
  "forecast": { "meta": {...}, "nino34": {...}, "pcs": {...}, "series": {...} },
  "diagnostics": { "spectral_radius_F": ..., "modes": [...], "enso_mode": {...},
                   "optimal_growth": {...}, "tau_test": {...} }
}
```

`run_all` additionally writes `results/run_all_summary.json` (run log + portfolio winner per
`(structure,D)` on held-out ACC). `aggregate` writes `results/agg/`:
`T1_nll_by_solver.csv`, `T2_robustness_by_solver.csv`, `T3_forecast_skill.csv`,
`T4_scaling_by_D.csv`, the pgfplots `.dat` files (`forecast_acc_by_lead.dat`,
`scaling_by_D.dat`), and `scaling_gate_summary.json` (best-NLL/scaling/portfolio/go-no-go).
