# Experiment G: Runtime Compositional Modeling  (manuscript Appendix `app:exp_g`, Table `tab:exp_g`)

## What this shows
Because DMCI treats programs as data, a library of symbolic model components are
just strings, *composition* is string manipulation, and a freshly composed
program flows through the same compiled interpreter with no per-composition
engineering. This experiment demonstrates three claims: (1) gradient descent
through the compiled interpreter identifies the **correct** way to compose two
components from data when the candidate structures are distinguishable
(product G2: 0.0004 vs >= 0.108 wrong; chain G3: 0.0004 vs >= 0.0007 wrong);
(2) where the structures are poorly separable the limit is identifiability, not
the mechanism (sum G1: correct 0.942 does not beat the wrong chain 0.772); and
(3) hot-swapping a component recompiles in 20-35 ms with zero engineering.

## Files
- `exp_g.py` -- main driver. Builds each composition by string manipulation,
  compiles it fresh via `neural_compiler` + `bootstrap/compiler.scm`, fits
  parameters with batched DMCI + Adam, and times hot-swap recompilation.
  Has a CLI (`--problem`, `--seed`, `--output-dir`) and writes one JSON per
  (problem, seed) to `results/`.
- `config.py` -- defines the 6-module library (`MODULE_LIBRARY`: exponential
  decay, oscillation, polynomial2, sigmoid, power law, gaussian as Scheme
  templates), the 3 target problems (`PROBLEMS`: G1 sum, G2 product, G3 chain)
  with ground-truth lambdas and true params, and `ExpGConfig` (default
  hyperparameters: 2000 epochs, lr 0.01, threshold 1e-4, patience 200,
  64 data points, noise_std 0.02, 5 seeds).
- `modules.py` -- composition engine: `instantiate_module`, `compose_sum`,
  `compose_product`, `compose_chain` (regex substitution of `x` for chaining),
  `build_composition`, `composition_label`. This is the "composition is string
  manipulation" core.
- `slurm_submit.sh` -- HPC batch script (array 0-2 over the 3 problems,
  looping seeds 0-4 internally) on the `eight` CPU partition.
- `__init__.py` -- empty package marker.
- `results/` -- 15 output JSON files (3 problems x 5 seeds), committed.
- `logs/` -- empty; populated by SLURM stdout/stderr at run time.

## Data
- **Inputs:** synthetic, generated in-process by `generate_data()` in
  `exp_g.py`. For each problem the targets are sampled from the ground-truth
  lambda in `config.PROBLEMS` over a fixed `x_range` (G1/G2: 0-5, G3: 0-3),
  64 evenly spaced points, plus Gaussian noise `noise_std=0.02`. The data RNG
  is seeded with the run seed (`torch.manual_seed(seed)`); parameter init is
  seeded with `seed + 2000`. No external dataset; no download required. Fully
  regenerable from the seed.
- **Outputs:** `results/<problem>_seed<NN>.json`, one per (problem, seed).
  Each file contains: `problem`, `seed`, `config` (the full `ExpGConfig` used,
  so seeds/lr/epochs are self-documenting), a `fits` list of 6 records
  (2 individual modules, 1 correct composition, 2 wrong compositions,
  1 hot-swapped composition), and a `hot_swap` block with the three
  recompilation times in ms, their mean (`mean_compile_ms`), and the swapped
  MSE. Each fit record has `label`, `expression`, `param_names`,
  `fitted_values`, `final_mse`, `n_epochs`, `t_compile`, `t_train`,
  `converged`.
- **Provenance:** No `metadata.json` / `MANIFEST.txt` is present, and there is
  no separate aggregation script -- the manuscript table values are the
  per-label mean of `final_mse` over the 5 seeds and the mean of
  `hot_swap.mean_compile_ms` over the 5 seeds. Hyperparameters are recorded in
  each JSON's `config` block, and the input parameters are reproducible from
  the seeds. The committed JSONs were produced on HPC (`eight` partition,
  2026-05-31); compile-time numbers are hardware-dependent. Recommended
  additions for the artifact: a small `aggregate.py` that reproduces
  Table `tab:exp_g`, and a `MANIFEST.txt` recording host / git SHA / wall
  clock. Aggregation logic (verified against the paper) is:

  ```python
  import json, glob, statistics
  from collections import defaultdict
  agg = defaultdict(lambda: defaultdict(list)); hs = defaultdict(list)
  for f in glob.glob("results/*.json"):
      d = json.load(open(f))
      for fit in d["fits"]:
          agg[d["problem"]][fit["label"]].append(float(fit["final_mse"]))
      hs[d["problem"]].append(d["hot_swap"]["mean_compile_ms"])
  for p in sorted(agg):
      for lab in sorted(agg[p]):
          print(p, lab, round(statistics.mean(agg[p][lab]), 6))
      print(p, "hotswap_ms", round(statistics.mean(hs[p]), 2))
  ```

## How to run
From the repo root (`neural_compiler` and `bootstrap/compiler.scm` must be
importable; `sys.setrecursionlimit(5000)` is set inside the driver).

Single (problem, seed):
```
python3 -m experiments.exp_g.exp_g --problem 0 --seed 0 \
    --output-dir experiments/exp_g/results
```
Problem index: 0 = G1 sum, 1 = G2 product, 2 = G3 chain. Seeds used: 0-4.

Full sweep on HPC (3 problems x 5 seeds = 15 runs):
```
sbatch experiments/exp_g/slurm_submit.sh
```

Full sweep locally:
```
for P in 0 1 2; do for S in 0 1 2 3 4; do \
  python3 -m experiments.exp_g.exp_g --problem $P --seed $S \
    --output-dir experiments/exp_g/results; done; done
```

## Expected results
Populates Table `tab:exp_g` (mean MSE over 5 seeds; best competing wrong
composition; mean hot-swap recompilation time). Reproduced exactly from the
committed `results/` JSONs:

| Problem | Correct op | Correct MSE | Best wrong MSE | Hot-swap recompile |
|---|---|---|---|---|
| G1 decay + oscillation | sum | 0.942 | 0.772 (chain) | 20.0 ms |
| G2 damped oscillation | product | **0.0004** | 0.108 (sum) | 35.2 ms |
| G3 sigmoid of polynomial | chain | **0.0004** | 0.0007 (sum) | 27.2 ms |

Correct compositions win decisively for the product (G2) and chain (G3)
targets; the sum target (G1) is poorly separable from a chain of the same
parts, the manuscript's honest exception. Hot-swap recompilation is 20-35 ms
regardless.

## Environment, seeds, hardware
- Python 3.11.10 (`module load python/3.11.10` on HPC), PyTorch (project
  `.venv`), `neural_compiler` from this repo + `bootstrap/compiler.scm`.
- Seeds: data RNG = run seed; parameter init = seed + 2000; seeds 0-4 per
  problem. All recorded in each JSON's `config`/`seed` fields.
- Hardware for the committed results: HPC `eight` CPU partition
  (`slurm_submit.sh`: 4 CPUs, 16 GB, 24 h wall limit; CPU-only, no GPU),
  run 2026-05-31. The experiment also runs on a laptop CPU.
- Approximate wall clock: seconds per fit at the default 2000 epochs (early
  stopping with patience 200); a full single-problem run (6 fits + hot-swap)
  completes in well under a minute of compute, the whole 15-run sweep in a few
  minutes. Compile/hot-swap times are hardware-dependent (locally ~8-50 ms,
  HPC 20-35 ms).
