# Experiment D: Structural Search and the Cost of Interpretation  (manuscript Appendix `app:exp_d`, "Experiment D: Structural Search and the Cost of Interpretation")

## What this shows
DMCI remains *correct* but is *not* faster than direct recompilation when the program
structure itself varies (a genetic-programming symbolic-regression loop). The experiment
measures the per-candidate cost of running DMCI versus directly recompiling each candidate,
and establishes (a) a fixed ~25x per-candidate overhead with **no** wall-clock crossover at
any candidate budget, and (b) **method equivalence**: GP + direct-recompile and GP + DMCI
explore the identical candidate-program sequence and select the same best program / best
loss on every seed (both execute the same underlying math). It populates
Table `tab:exp_d_timing` and Figure `fig:exp_d_crossover`.

## Files
- `exp_d.py` -- main driver. Runs one GP search (`--method gp_direct|gp_dmci|both`, `--seed N`),
  evaluating 50 candidates x 20 generations = 1,000 candidates; for each candidate it times
  compile (`t_compile`) and the 20-epoch Adam inner constant-fit (`t_train`), and writes a
  per-run JSON to `results/`.
- `gp.py` -- minimal genetic-programming framework: `GPNode` AST, ramped half-and-half
  initialization, subtree mutation/crossover, tournament selection, and `to_scheme` /
  const-collection helpers. Function set `{+, -, *, /, sin, cos}`, terminals `{x, const_N}`.
- `config.py` -- `ExpDConfig` dataclass with all defaults (pop 50, 20 generations,
  tournament size 3, crossover 0.8, inner 20 epochs, inner lr 0.05, 20 data points on
  x in [0.5, 3.0], target a=2 b=3 c=0.5, 5 seeds).
- `aggregate_table.py` -- reconstructs every cell of Table `tab:exp_d_timing` from the 10
  result JSONs (mean compile/train/total ms, %-compile, DMCI/direct ratio) and self-checks
  against the manuscript values; exits non-zero on drift. Run: `python3 -m experiments.exp_d.aggregate_table`.
- `analyze.py` -- per-seed summaries plus a budget-vs-time crossover table; writes
  `results/crossover_data.json` (a secondary artifact, not directly consumed by the paper).
- `slurm_submit.sh` -- HPC array job (`--array=0-9`): tasks 0-4 = gp_direct seeds 0-4,
  tasks 5-9 = gp_dmci seeds 0-4, on the `eight` CPU partition.
- `__init__.py` -- package marker.

Figure `.dat` files are produced by `paper-arxiv/figures/make_exp_d_data.py`
(and the identical `paper-dmci/figures/make_exp_d_data.py`), which reads `results/` and
averages cumulative wall-time across seeds.

## Data
- **Inputs:** fully synthetic, no external dataset. Target function
  f(x) = a*sin(b*x) + c*x^2 with a=2, b=3, c=0.5 (= 2 sin(3x) + 0.5 x^2), sampled at 20
  evenly spaced points on x in [0.5, 3.0] by `_generate_data` / `_target_fn` in `exp_d.py`.
  Determinism is fixed by `random.seed(seed)` + `torch.manual_seed(seed)` in `run_gp`; the
  per-candidate inner fit reseeds with `torch.manual_seed(seed + candidate_id)`. Seeds 0-4.
  The DMCI path additionally embeds the self-hosted evaluator from `bootstrap/compiler.scm`.
- **Outputs:** `results/gp_{direct,dmci}_seed0{0..4}.json` (10 files, ~440-480 KB each,
  tracked via Git LFS per `.gitattributes`). Each file contains the full `config`, a
  `candidates` list of 1,000 dicts (each with `generation`, `candidate_id`, `scheme_source`,
  `n_consts`, `tree_size`, `tree_depth`, `t_compile`, `t_train`, `t_total`, `final_loss`,
  `const_values`), plus `total_wall_time`, `best_loss`, `best_source`,
  `n_candidates_evaluated`. (`results/crossover_data.json` is regenerated on demand by
  `analyze.py` and is not committed.)
- **Provenance:** `results/MANIFEST.txt` lists the 10 committed JSONs; verify with
  `python3 -m experiments.check_manifests`. There is no per-run `metadata.json` (timing/
  config provenance lives inside each JSON's `config` block and `total_wall_time`); a
  small `metadata.json` recording host/CPU/python/torch versions would be a useful add.

## How to run
```
# Regenerate all 10 runs on the cluster (CPU; timing-sensitive, run on HPC not laptop):
sbatch experiments/exp_d/slurm_submit.sh

# Or a single run locally / interactively:
python3 -m experiments.exp_d.exp_d --method both --seed 0 \
    --output-dir experiments/exp_d/results

# Reproduce Table tab:exp_d_timing and self-check against the manuscript:
python3 -m experiments.exp_d.aggregate_table

# Regenerate the figure .dat files from results:
python3 paper-arxiv/figures/make_exp_d_data.py
```

## Expected results
From Table `tab:exp_d_timing` (per-candidate cost decomposition, 5 seeds x 1,000 candidates
= 5,000 evaluations per method):

| Method | t_compile | t_train | t_total | % Compile | Ratio |
|--------|-----------|---------|---------|-----------|-------|
| Direct | 4.1 ms    | 148 ms  | 152 ms  | 2.7%      | 1x    |
| DMCI   | 22.9 ms   | 3,824 ms| 3,846 ms| 0.6%      | 25.3x |

`aggregate_table.py` reproduces all 10 of these cells exactly (10/10 self-check pass).
Figure `fig:exp_d_crossover`: cumulative wall time at 1,000 candidates is 152.10 s (direct)
vs 3,846.44 s (DMCI) -> ~25x, with no crossover. **Method equivalence:** per seed, both
methods report identical `best_loss` and `best_source` (verified: seed best losses
18.726223 / 17.516260 / 3.802500 / 7.543960 / 17.585594 for seeds 0-4 respectively, and the
full 1,000-candidate `scheme_source` sequence is identical between methods). GP solution
quality is not an object of study here -- only timing and method equivalence.

## Environment, seeds, hardware
- Python 3 with PyTorch (CPU); imports `neural_compiler.compiler.compile_program`,
  `neural_compiler.evaluator.evaluate`, `neural_compiler.runtime.tagged_value`. On HPC use
  the repo's `uv` `.venv` (`source .venv/bin/activate`); no conda/module-load.
- Seeds: 0-4 (outer GP RNG + torch); inner constant fit reseeds per candidate.
- Hardware: HPC `eight` CPU partition, 4 cpus/task, 16 GB, 8 h wall limit per task
  (`slurm_submit.sh`). Timing is CPU-bound and laptop runs will not match the table, so
  generate/verify on the cluster.
- Approximate wall-clock per run (from committed `total_wall_time`): direct ~146-166 s/seed;
  DMCI ~3,450-4,713 s/seed (~57-79 min). All 10 runs together are a few hours of CPU time.
