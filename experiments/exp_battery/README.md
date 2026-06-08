# Experiment L: Battery Capacity-Fade Co-Search  (manuscript §4.4 `\label{sec:exp_battery}`; Appendix `\label{app:exp_battery_ext}`; Table 2 `\label{tab:battery}`; Figure 3 `\label{fig:battery_real}`)

> Note: §4.4 / §4.5 numbering varies between manuscript drafts. In `paper-arxiv/paper.tex`
> the battery section is `\label{sec:exp_battery}` (the "Does Co-Search Discover Better
> Scientific Structure?" subsection) and the extended appendix is `\label{app:exp_battery_ext}`.

## What this shows
LLM-and-DMCI **co-search**: a language model (Qwen 3.6, via OpenEvolve) proposes battery
degradation *structures* as Scheme programs while one frozen DMCI interpreter calibrates each
program's continuous rate parameters by exact gradients, and held-out forecast skill selects.
On a **mechanism-labeled synthetic** target (two-reservoir knee) the search *recovers the correct
degradation structure* from a smooth seed that provably cannot express a knee; on **real Severson
lithium-ion cells** it forecasts the held-out late-life fade better than hand-crafted models on the
hard early-extrapolation split. An inner-fitter ablation shows DMCI's exact gradients (not the
structure search alone) are load-bearing. This is the central-claim experiment for the co-search
contribution; FluZoo (`experiments/exp_fluzoo/`, Appendix `app:exp_fluzoo_ext`) is its cautionary
counterpart.

## Files
- `config.py` -- battery overrides of the shared FluZoo `DEFAULT` config: `N_SERIES=1` (one capacity
  series per cell), `T_CYCLES=100` grid, two held-out splits `KSPLIT_LATE=70` / `KSPLIT_EARLY=45`,
  and the calibration knobs `BCFG` (`adam_iters=400`, `lr=0.05`, `seeds=(0,1,2)`).
- `curves.py` -- closed-form numpy capacity curves for the 5 reference structures + their ground-truth
  parameters (`TRUE_PARAMS`) and per-cell jitter (`JITTER`); one source of truth shared by the
  synthetic generator and the forecast read-out.
- `synth.py` -- mechanism-labeled synthetic capacity-fade cell generator (seeded numpy `Generator`,
  `NOISE=2e-3` coulometry-grade noise) from `curves.TRUE_PARAMS`.
- `structures.py` -- the 5 reference degradation structures as DMCI Scheme programs (smooth→knee axis:
  `sqrt_t_SEI`, `power_law_kp`, `stretched_exp`, `two_reservoir_min`, `sigmoidal_knee`) plus the
  `CAN_KNEE` capability map. These are both the search seed/reference families and the baselines.
- `score.py` -- fit-early / forecast-late DMCI scoring for *named* structures (`score_pooled`,
  `score_cell`, `fit_structure`); calibration runs through the compiled interpreter, forecast is read
  out in closed form and cross-checked against the DMCI forward path (`dmci_predict_check`).
- `oe_score.py` -- scores an *arbitrary evolved* program end-to-end through DMCI: a validity funnel
  (`screen`) then batched per-cell calibration + held-out forecast via the interpreter's predict path
  (`run_predict_batched`). Returns `combined_score = -holdout_rmse` and MAP-Elites structural features.
- `pilot.py` -- local CPU de-risk pilot (no cluster): (1) gradient health vs finite differences,
  (2) landscape ruggedness, (3) structure-recovery confusion matrix. Writes `results/pilot.json`.
- `openevolve/initial_program.py` -- the OpenEvolve seed: the smooth `sqrt_t_SEI` Scheme model the
  search starts from (cannot express a knee).
- `openevolve/oe_evaluator.py` -- OpenEvolve evaluator bridging a candidate program to `oe_score`;
  reads the target from `$BAT_TARGET` (default `results/target.pt`), `adam_iters` from `$BAT_ADAM_ITERS`
  (default 60, the cheap search-time fitness), ksplit from `$BAT_KSPLIT`.
- `openevolve/run_battery.py` -- the OpenEvolve driver (LLM-ensemble mutation + MAP-Elites + islands;
  feature dims `complexity` × `knee_capable`; Qwen 3.6 `real` preset via MindRouter).
- `openevolve/gen_target.py` -- generate + save the SYNTHETIC search target (`results/target.pt`,
  default `two_reservoir_min`, 12 cells, seed 0).
- `openevolve/gen_target_real.py` -- build the REAL Severson target (`results/target_real.pt`): load
  per-cell capacity-vs-cycle, normalize to SOH (`Q/Q0`), resample onto the common `T=100` life-fraction
  grid (`--grid lifefrac`, vs the `absolute` negative-control window). Restricted numpy-only unpickler.
- `openevolve/convert_pkl.py` -- USER-RUN (`! python3 -m ...`) one-shot converter: deserializes the
  vetted `severson_capacity.pkl` with the restricted unpickler, re-saves as safe `.npz`, prints the
  grid-decision report. (Run under your authority; the agent classifier blocks pickle deserialization.)
- `openevolve/rescore.py` -- rigorous re-score of the 3 SYNTHETIC island winners + references at both
  `iters=60` (search-time) and `iters=300` (rigorous). Writes `results/battery_rescore.json`.
- `openevolve/rescore_real.py` -- rigorous re-score of the 3 REAL island winners vs smooth families +
  naive floors at `iters=300`, both splits. Writes `results/battery_rescore_real[_ks{70,45}].json`.
- `openevolve/rescore_knee.py` -- re-score the hand KNEE references (two-reservoir, sigmoidal) on the
  REAL target at `iters=300` (so "beats the best hand model" includes hand knees). Writes
  `results/battery_rescore_real_knee[_ks*].json`.
- `openevolve/ablate_inner.py` -- inner-fitter ablation: replace DMCI exact-gradient Adam with
  gradient-free differential evolution on the SAME structures at matched wall-clock. Writes
  `results/battery_ablate_inner[_ks*].json`.
- `openevolve/funnel.py` -- canonical-AST distinct-structure funnel over an island archive
  (`--prefix bat_island` synthetic / `bat_real_island` real); produces the "N programs / M distinct
  structures" counts. Writes `results/funnel_<prefix>.json`.
- `openevolve/smoke_real.py` -- quick low-iter smoke of the reference structures on `target_real.pt`
  (sanity check before the multi-hour island runs).
- `openevolve/run_real_islands.sh` -- launches 3 OpenEvolve islands on the real target (`bat_real_island_{0,1,2}`).
- `openevolve/slurm_battery_node.sh` -- one OpenEvolve island on one node (200 iters, 24 workers).
- `openevolve/slurm_rescore.sh` -- generic SLURM wrapper for any rescore/ablate module.

## Data
- **Inputs (synthetic):** generated, fully reproducible. `openevolve/gen_target.py` calls
  `synth.make_cells` with a seeded numpy `Generator` (default `--mechanism two_reservoir_min
  --cells 12 --seed 0`), ground truth in `curves.TRUE_PARAMS`. No file needed in git; regenerate with
  the command below. The manuscript's synthetic leg uses 12 mechanism-labeled cells (knee near cycle 55).
- **Inputs (real):** REAL - Severson et al. 2019 commercial A123 LFP/graphite fast-charging dataset
  (124-cell modeling subset), Nature Energy DOI `10.1038/s41560-019-0356-8`, public data at
  `https://data.matr.io/1/`. This repo ships a per-cell capacity-vs-cycle extract under
  `data/raw/severson_capacity.{pkl,npz}` (120 raw cells, batches b1/b2/b3; provenance: the rg1990
  "knee-finder" capacity pkl, re-encoded to a pickle-free `.npz` by `convert_pkl.py`). `data/raw/` is
  **gitignored** (binary), so on a fresh checkout fetch the dataset from the Severson source above (or
  any of the `.pkl`/`.csv`/`.mat` forms `gen_target_real.load_source` accepts) and rebuild the target.
  `gen_target_real.py` drops non-finite/anomalous channels, leaving **117 cells**, and resamples each
  onto the `T=100` life-fraction grid (`Q0` = first kept cycle's capacity).
- **Outputs (present locally / git-tracked):**
  - `results/target_real.pt` -- the built REAL target: `{"obs":[117,100,1] float32, "mechanism":
    "real_severson", "cell_ids":[...117], "grid":"lifefrac", "eol_soh":0.80}`. Verified present.
  - `results/pilot.json` -- the local de-risk pilot output (gradient health PASS for all 5 structures;
    structure recovery 4/5 at ksplit 70 and 3/5 at ksplit 45; mean held-out spread ~1000% / ~460%).
    This is the GO/no-go gate, **not** the manuscript co-search numbers.
- **Outputs (HPC-only - NOT in this repo; must be regenerated on the cluster):**
  - `results/target.pt` -- the synthetic search target (regenerate with `gen_target.py`).
  - `results/bat_island_{0,1,2}/` and `results/bat_real_island_{0,1,2}/` -- the OpenEvolve run dirs
    (per-island `best/best_program.py` + `checkpoints/`); inputs to `rescore*`, `ablate_inner`, `funnel`.
  - `results/battery_rescore.json` -- synthetic recovery RMSE (Table 2 top).
  - `results/battery_rescore_real[_ks70|_ks45].json` + `..._real_knee[_ks*].json` -- real forecast
    RMSE incl. naive floors (Table 2 bottom).
  - `results/battery_ablate_inner[_ks*].json` -- DE-vs-DMCI ablation (the inner-fitter paragraph).
  - `results/funnel_bat_island.json` / `results/funnel_bat_real_island.json` -- distinct-structure
    counts (600 progs / 119 distinct synthetic; 601 progs / 160 distinct real).
- **Figure data (git-tracked, in the paper tree):** `paper-arxiv/figures/battery_real_lifefrac.dat`
  and `battery_real_absolute.dat` (header `k mean p10 p90`, 100 grid points) feed Figure 3. These are
  derived from `target_real.pt` / the `absolute`-grid build; no checked-in script regenerates them
  (see Gaps).
- **Provenance:** No `metadata.json` / `MANIFEST.txt` exists in `results/`. `target_real.pt` self-
  documents (`grid`, `eol_soh`, `cell_ids`). **Should be added:** a `results/MANIFEST.txt` listing
  which JSON populates which Table 2 / Figure 3 cell, the OpenEvolve seed/iterations, and the dataset
  source URL/DOI, since the co-search result files are HPC-only and otherwise undocumented.

## How to run
```bash
# --- Local de-risk (CPU, no cluster, ~1 h) ---
python3 -m experiments.exp_battery.pilot --smoke            # fast plumbing pass
python3 -m experiments.exp_battery.pilot                    # full pilot -> results/pilot.json

# --- Build targets ---
python3 -m experiments.exp_battery.openevolve.gen_target \
        --mechanism two_reservoir_min --cells 12 --seed 0   # -> results/target.pt (synthetic)
# Real: first place the Severson capacity extract at data/raw/severson_capacity.pkl, then:
! python3 -m experiments.exp_battery.openevolve.convert_pkl # USER-RUN: pkl -> safe .npz + report
python3 -m experiments.exp_battery.openevolve.gen_target_real \
        --src experiments/exp_battery/data/raw/severson_capacity.npz \
        --grid lifefrac                                     # -> results/target_real.pt (117 cells)

# --- Co-search outer loop (HPC; needs MINDROUTER_API_KEY in .env) ---
sbatch experiments/exp_battery/openevolve/slurm_battery_node.sh 0 \
       experiments/exp_battery/results/bat_island_0 24      # synthetic, one island (x3 islands)
bash   experiments/exp_battery/openevolve/run_real_islands.sh  # real, launches 3 islands

# --- Rigorous re-scores + ablation + funnel (HPC, ~6 h each) ---
sbatch experiments/exp_battery/openevolve/slurm_rescore.sh experiments.exp_battery.openevolve.rescore
sbatch experiments/exp_battery/openevolve/slurm_rescore.sh experiments.exp_battery.openevolve.rescore_real
sbatch experiments/exp_battery/openevolve/slurm_rescore.sh experiments.exp_battery.openevolve.rescore_knee
sbatch --export=ALL,BAT_RESCORE_KS=70 experiments/exp_battery/openevolve/slurm_rescore.sh \
       experiments.exp_battery.openevolve.ablate_inner
python3 -m experiments.exp_battery.openevolve.funnel --prefix bat_island
python3 -m experiments.exp_battery.openevolve.funnel --prefix bat_real_island
```

## Expected results
**Table 2 (top) - synthetic, held-out RMSE @ 300 steps** (`results/battery_rescore.json`):
smooth √t seed `0.0301`, hand two-reservoir `0.0143`, hand sigmoidal `0.0149`,
**co-search (recovered) `0.0111`** (2.7× better than the seed). Two of three islands converge to
`min(q0-B1·√k, q0-B2·k)` - algebraically the generating family - from a seed that cannot express a knee.
Funnel: 600 programs / **119 distinct structures** across 3 islands (48/42/65).

**Table 2 (bottom) - real Severson, held-out tail RMSE late/early** (`battery_rescore_real*.json`):
persistence `0.084`/`0.089`, linear extrapolation `0.080`/`0.078`, best smooth family `0.083`/`0.086`,
best hand knee (sigmoidal) `0.051`/`0.057`, **co-search selected (island2, three-reservoir bottleneck)
`0.051`/`0.034`** - 1.7× better than the best hand model on the early-extrapolation split, matching on
the late split. Funnel: 601 programs / **160 distinct structures**.

**Inner-fitter ablation** (`battery_ablate_inner*.json`): DE vs DMCI (late/early) island0 `1.98`/`1.76`
vs `0.072`/`0.065`, island1 `0.25`/`0.21` vs `0.027`/`0.167`, island2 `0.90`/`0.92` vs `0.051`/`0.034`
over 468–702 free params - gradient-free is **9–27× worse** on the identical structures.

**Figure 3** (`battery_real_{lifefrac,absolute}.dat`): life-fraction grid shows flat-then-rollover
(mean fade ≈0.19 SOH); absolute first-100-cycle window is near-flat (mean fade <0.002 SOH) - the
structure-blind negative control motivating the life-fraction grid.

**Pilot (`results/pilot.json`, present):** gradient health PASS (max grad rel-err ≤1.4e-3, DMCI-vs-
closed-form ≤1e-7 across all 5 structures); GO verdict (rugged held-out landscape, structure recovery
works). This validates the substrate, not the co-search outcome.

## Environment, seeds, hardware
- **Python/torch:** Python 3.11, PyTorch (CPU float32 native; DMCI is float32 throughout). Real-data
  build uses numpy (+ optional pandas/h5py for `.csv`/`.mat` sources); the report path is torch-free so
  it runs on the HPC login node. Co-search needs the `openevolve` package + `python-dotenv` and a
  MindRouter API key (`MINDROUTER_API_KEY`, `MINDROUTER_BASE_URL` in repo-root `.env`).
- **Seeds:** synthetic generator seed 0 (`gen_target`/`synth`, `np.random.default_rng`); calibration
  multi-start seeds `(0,1,2)` (`BCFG.seeds`); OpenEvolve `random_seed=0`, island seeds 0/1/2.
- **Calibration budgets:** search-time fitness `adam_iters=60` (the under-converged proxy, the FluZoo
  lesson); rigorous re-score `iters=300`. `lr=0.05`, `T_CYCLES=100`, splits 70/45, `recursionlimit=20000`.
- **Hardware / cost:** local de-risk on Mac CPU (pilot ~3500 s = ~1 h). Co-search and re-scores on the
  University of Idaho `fortyfive` cluster (gpu-8 / sheneman n128 etc.), CPU-bound interpreter
  (`torch.set_num_threads(1)`, OpenEvolve workers=24/node). Each island = 200 iterations, 24h SLURM
  wall limit; each `iters=300` rescore eval ≈1160 s (6 programs × 2 splits ⇒ 6 h limit); the DE
  ablation uses a matched per-fit wall-clock budget (default 1000 s/fit).
- **Verified during this audit:** `gen_target` (synthetic, 12 cells → `[12,100,1]`), `pilot.gradient_health`
  (PASS, all 5 structures), and `oe_score.score_evolved` on `target_real.pt` (`[117,100,1]`, finite
  holdout RMSE) all run locally.
