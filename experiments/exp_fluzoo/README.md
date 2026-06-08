# Experiment K: FluZoo - Influenza Co-Search and the Fitness-Fidelity Stress Test  (manuscript Appendix `\label{app:exp_fluzoo_ext}`; also the "When co-search overfits its own fitness" paragraph in §4.5, `paper-arxiv/paper.tex`)

## What this shows
A language model (Qwen 3.6) proposes influenza-model *structures* as Scheme programs that one frozen
DMCI interpreter calibrates by exact gradients on real CDC ILINet weighted-%ILI data, while held-out
forecast skill selects programs - the same LLM-and-DMCI co-search machinery used for the battery
flagship (§4.5 / Appendix `app:exp_battery_ext`). FluZoo is the **cautionary half** of that story: under
a cheap 60-step search-time fitness the evolved program *appears* to beat a hand-written regional SEIR,
but re-scored under a converged 300-step protocol the advantage **vanishes** (val 0.0148 vs 0.0149,
test 0.0181 vs 0.0181 - a tie). The methodological point: the inner-loop fitness must be converged
enough to be a faithful skill signal, or the outer structure search overfits the loose proxy.

## Files
- `config.py` -- single source of truth (seasons, splits, recursion limit, Adam/forecast budgets, paths).
- `paramspec.py` -- generic `(params ...)` schema -> unconstrained-transform layer (softplus/sigmoid/tanh/free); Adam optimizes raw leaves, mapped back differentiably.
- `programs.py` -- program representation: S-expression parse, the DMCI fold (Gaussian NLL), forecast read-back (`yhat` contract + PREDICT base-case AST swap), and the SIR/SEIR/SEIRS national + regional-vector reference programs.
- `validity.py` -- 8-stage validity funnel (parse, op-prescan, compile, free-vars, finite forward, finite gradient, stable rollout, forecastable) + canonical structural de-dup hash. Produces the funnel table (T1).
- `calibrate.py` -- inner loop: multi-start Adam over a program's raw parameters, summing per-season training NLL; selection is never on training NLL.
- `forecast.py` -- held-out skill via filter-then-forecast: freeze structural params, re-fit only IC params per test origin, roll 1–4 wk, score RMSE/MAE/corr (the selection metric).
- `baselines.py` -- non-DMCI baselines (persistence, damped, seasonal-naive, AR(p)) scored on identical origins/horizons.
- `runner.py` -- atomic sweep unit: calibrate one program, score its held-out skill (JSON).
- `run_all.py` -- spawn-pool sweep driver: calibrate every accepted program -> select on validation -> report test -> emit skill-vs-#programs curve. Writes `results/run_all_summary.json`.
- `aggregate.py` -- builds the manuscript tables/figure data (T1 funnel, T2 forecast, T3 scaling .csv/.dat, diversity) into `results/agg/`.
- `llm_generate.py` -- outer loop (i.i.d. sampling): MindRouter Qwen writes whole Scheme programs; VALIDATE->REPAIR via `validity.py`; `llm_cache/`; `--offline` runs the funnel on the references with no API.
- `evolve.py` -- LLM-guided evolutionary outer loop (hand-rolled island/mutation co-search; the closed-loop version of `llm_generate`).
- `smoke.py` -- fast local self-test of the inner loop and harness (no API needed).
- `diagnostic_landscape.py` -- diagnostic for whether FluZoo's flat fitness is intrinsic or a short-horizon/metric artifact.
- `openevolve/` -- the **outer loop actually used for the manuscript stress test** (AlphaEvolve-style):
  - `run_oe.py` -- runs FluZoo evolution through the OpenEvolve package (LLM ensemble = mutation operator, MAP-Elites + islands).
  - `oe_evaluator.py` -- bridges OpenEvolve to the DMCI inner loop (cascade: funnel then calibrate+forecast); `combined_score = -val_rmse`.
  - `mock_evaluator.py` -- instant structure-based pseudo-fitness for plumbing tests (no DMCI).
  - `initial_program.py` -- the seed Scheme program OpenEvolve diff-edits.
  - `pool_oe.py` -- pools the per-island OpenEvolve `best/` outputs, finds the global best, re-scores it on TEST (the headline number).
  - `slurm_oe.sh` / `slurm_oe_islands.sh` / `slurm_oe_node.sh` -- HPC submission (n128 single run; eight-partition meta-island array).
- `data/{fetch_ilinet,preprocess_flu,build_data}.py` -- the CDC ILINet pipeline (fetch -> preprocess -> processed arrays + metadata).
- `slurm_generate.sh`, `slurm_sweep.sh`, `slurm_sweep_array.sh`, `slurm_evolve.sh`, `slurm_evolve_islands.sh` -- HPC submission for the i.i.d./evolutionary (non-OpenEvolve) pipeline.

## Data
- **Inputs:** REAL. CDC ILINet weighted %ILI via the **Delphi Epidata `fluview` endpoint**
  (`https://api.delphi.cmu.edu/epidata/fluview/`; cited `farrow2015delphi`, `cdc_fluview`). National +
  10 HHS regions (R=11 series), MMWR epiweeks **2010w40–2025w39**, latest (finalized) issue with
  per-region release dates `2026-05-29` recorded for a pinned as-of snapshot. Values are stored as
  proportions (wILI/100) to match the model observable. **Fetch/regenerate:** `python3 -m
  experiments.exp_fluzoo.data.build_data` (downloads raw JSON, writes processed arrays + provenance).
  Raw payloads (`data/raw/*.json`, ~2.5 MB) and processed arrays (`data/processed/*.npy`) are
  **gitignored** (`data/.gitignore` tracks only `metadata.json`); they live locally / on HPC and are
  rebuilt by `build_data.py`. Splits: train seasons 2010–2017 (417 wk), val 2018/2021/2022 (156 wk),
  test 2023/2024 (104 wk), pandemic season 2020 (53 wk) held out as distribution shift. T=782 weeks.
- **Outputs (what lands in `results/`):**
  - `run_all_summary.json` -- selected program, val/test mean RMSE, per-horizon table, skill curve, baselines, config.
  - `seir_regional.json`, `sir_regional.json` -- per-program calibration + forecast records (params, NLL, AIC/BIC, per-horizon RMSE/MAE/corr).
  - `agg/{T1_funnel.csv, T2_forecast.csv, T3_scaling.csv, T3_scaling.dat, diversity.csv, agg_summary.json}` -- the table/figure data.
  - `openevolve/oe_output/best/{best_program.py, best_program_info.json}` and `oe_output/logs/*.log` -- OpenEvolve outputs.
- **Provenance:** `data/processed/metadata.json` is present and complete (source, URL, `source_sha256`,
  epiweek window, regions, release dates, splits, binding contract, wILI range). **There is no
  `MANIFEST.txt` for `results/`, and one is needed** (see Gaps): the committed `results/` files are a
  *smoke/integration* run, not the manuscript campaign, and nothing currently records that distinction.

## How to run
```bash
# 1. Build data locally (public API; rsync data/processed to HPC afterward)
python3 -m experiments.exp_fluzoo.data.build_data

# 2. Sanity-check the harness (no API/data needed beyond step 1)
python3 -m experiments.exp_fluzoo.smoke
python3 -m experiments.exp_fluzoo.llm_generate --offline       # funnel on reference programs

# 3a. i.i.d. zoo + sweep (HPC; DMCI is interpreter-bound, never run locally):
sbatch experiments/exp_fluzoo/slurm_generate.sh               # campus MindRouter (Qwen)
sbatch experiments/exp_fluzoo/slurm_sweep.sh                  # n128 spawn-pool sweep
python3 -m experiments.exp_fluzoo.aggregate                   # build results/agg/ tables

# 3b. The OpenEvolve outer loop that produced the manuscript stress-test numbers:
python3 -m experiments.exp_fluzoo.openevolve.run_oe --mock --models gpt55 --iterations 3   # plumbing test
sbatch experiments/exp_fluzoo/openevolve/slurm_oe.sh         # n128, --iterations 300 --workers 24
sbatch experiments/exp_fluzoo/openevolve/slurm_oe_islands.sh # eight-partition meta-island array (1-10)
python3 -m experiments.exp_fluzoo.openevolve.pool_oe         # pool islands, re-score global best on TEST

# Quick local integration check of the i.i.d. orchestration (1 program, tiny config):
python3 -m experiments.exp_fluzoo.run_all --programs reference --limit 1 --workers 1 \
    --train-seasons 2014 --val-seasons 2018 --test-seasons 2023 --seeds 0 \
    --adam-iters 5 --refit-iters 2 --origin-stride 40
```

## Expected results
The numbers reported in the manuscript (Appendix `app:exp_fluzoo_ext` and the §4.5 paragraph):
- Three long-horizon OpenEvolve islands evolved to completion (**200 iterations each**).
- Global best = a spatial mean-field-coupling structure with dual exposed/infectious reporting
  (**11 parameters**); search-time validation RMSE **0.0100** under the cheap 60-step calibration.
- Re-scored at the rigorous **300-step** protocol with rolling forecast origins and IC refit:
  - evolved program: **val 0.01477 / test 0.01812** (manuscript also rounds to 0.0148 / 0.0181)
  - hand-written regional SEIR seed: **val 0.01488 / test 0.01805** (rounds to 0.0149 / 0.0181)
  - SEIRS variant: **val 0.01473 / test 0.01856**
  - => a tie on both held-out splits; the apparent search-time advantage erases.
These populate the prose of Appendix K; there is no numbered Table/Figure for FluZoo in `paper-arxiv`.

## Environment, seeds, hardware
- Python 3.11, PyTorch (CPU), NumPy. `sys.setrecursionlimit(20000)` before importing `neural_compiler`
  (set by `config.DEFAULT.recursion_limit` / the OpenEvolve evaluator); threads pinned to 1
  (`OMP/MKL/OPENBLAS_NUM_THREADS=1`, `torch.set_num_threads(1)`).
- Hardware: HPC `fortyfive.hpc.uidaho.edu` - n128 (CPU, partition `sheneman`, 56 cpus) for the n128
  OpenEvolve run; eight-partition (16 cpus/node) for the meta-island array. DMCI calibration is
  CPU/interpreter-bound and is **never** run on the local Mac.
- Seeds: OpenEvolve island array uses `--seed = SLURM_ARRAY_TASK_ID` (1–10); the i.i.d. sweep takes
  `--seeds`. Data build is deterministic (latest-issue snapshot pinned by `release_dates` +
  `source_sha256` in `metadata.json`).
- Cost: a single forward+backward over a multi-season rollout is seconds on one core; OpenEvolve at
  300 iterations / 24 workers runs within the 24 h n128 wall-clock; the eight-partition array uses
  12 h/node with trimmed Adam (`OE_ADAM_ITERS=45`).

---

### Reproducibility caveat (read before trusting `results/`)
The headline manuscript numbers (0.01477/0.01812, 0.01488/0.01805, 0.01473/0.01856, 0.0100; "3 islands,
200 iterations each") were produced on HPC and **are not in this directory** - a repo-wide grep finds
them in no file. The committed `results/` and `results/agg/` files are a **tiny local smoke run**
(1 train season 2014, 5 Adam iters, 4 reference programs, `sir_regional` selected with test RMSE 0.0367
and NaN correlations) and do **not** correspond to anything in the manuscript. The committed OpenEvolve
output under `openevolve/oe_output/` is likewise a 2–3 iteration *mock/failed-LLM* run (best program is
generation 0, val_rmse 0.0155). To reproduce the manuscript stress test, run step 3b on campus/HPC and
collect `pool_oe` output; the resulting `best_program_info.json` (300-step re-score) is what should be
archived alongside this README.
