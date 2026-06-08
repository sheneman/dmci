# Experiment J: Program-Space Calibration - DMCI vs. compile-each-program  (manuscript Appendix `app:exp_j`, "Experiment J: Program-Space Calibration, DMCI versus Compile-Each-Program"; also cited from `sec:exp_h`, the Related Work, and the ablation/failure-mode tables)

## What this shows
When the object of optimization is a *growing space of distinct, runtime-generated programs* (the regime
LLM-driven discovery produces), one compiled interpreter (DMCI) wins on the axes that scale: **amortized
compile** (one interpreter compile vs. N), **uniform 100% coverage including recursion** that automatic
`sympy.lambdify`→JAX cannot reach, and **zero per-structure engineering** - while parameter-recovery
accuracy is held at *parity* across arms so the comparison isolates cost and coverage, not accuracy. This
is the workflow-economics evidence behind the "programs as data" claim (Table `tab:exp_j`, Figure
`fig:exp_j_compile`), and it reproduces on 260 genuine LLM-authored programs.

## Files
- `corpus.py` -- synthetic generator: structurally-distinct random expression trees (closed-form) and
  damped-Euler relaxations (recursive), each emitted three consistent ways (`to_scheme` for DMCI/B1,
  Python ground truth, `to_jax` hand-port for B2); deterministic from a single seed.
- `arms.py` -- the three arms and their instrumentation: DMCI (`compile_interpreter` once, then
  `evaluate_program` per program), B1 (`lambdify`→`jax.jit(jax.grad)`), B2 (hand-port to jitted JAX);
  plus the shared matched-recovery driver (scipy L-BFGS-B multi-start in log-reparameterized coords).
- `run_expj.py` -- synthetic sweep: four curves vs. N at recursive fractions {0%, 100%}; writes
  `results/expj_results.json`.
- `llm_corpus.py` -- external-validity corpus: cached MindRouter (qwen) generation + `Program`
  construction for real LLM output (closed-form + recursive families).
- `sexp_eval.py` -- generic closed-form Scheme→Python/JAX evaluator + AST/token size proxies, so a real
  LLM program string becomes an Exp J `Program` without hand-writing per-program code.
- `run_expj_llm.py` -- LLM-validation runner; writes `results/expj_llm_results.json`.
- `slurm_expj.sh`, `slurm_expj_llm.sh` -- HPC submit scripts (CPU `eight` partition).
- `LLM_VALIDATION.md` -- detailed external-validity writeup (generation protocol, scoping, results).
- Reuses: `neural_compiler.{compile_interpreter,evaluate_program,save_compiled}`,
  `experiments.exp_i.lambdify_baseline` (B1), and `experiments.exp_f.{exp_f,llm_client}` (parser,
  param detection, LLM client) for the LLM subset.

## Data
- **Inputs (synthetic):** *no external data*. `corpus.py::generate_corpus(n, recursive_fraction, seed=0)`
  deterministically generates N structurally-distinct programs; per-program training data is synthesized
  by `gen_data` from each program's ground-truth callable. Fixed seed = 0 throughout
  (`run_expj.py` calls `generate_corpus(..., seed=0)`; data seed = `seed+1+pid`). Fully regenerable.
- **Inputs (LLM subset):** real programs authored by **MindRouter** (University of Idaho
  OpenAI-compatible endpoint), model `qwen/qwen3.6-27b`, thinking enabled, temperature 0.7. Generation
  is cached under `experiments/exp_j/llm_cache/` keyed by SHA-256 of (model, thinking, system, user).
  **CAVEAT (reproducibility gap): the `llm_cache/` directory is NOT present in this repo and is NOT
  committed** (unlike `experiments/exp_b/llm_cache/`, which is committed). `LLM_VALIDATION.md` currently
  claims "The cache is committed to the repository," which is inaccurate - the cache is absent. To
  replay the LLM subset you must re-query MindRouter (compute-node egress + `MINDROUTER_API_KEY` in
  `.env`), which is non-deterministic against endpoint drift. **However**, the full set of 260 generated
  program Scheme strings (with `pid`, `kind`, `param_names`, `port_loc`, coverage flags) is embedded in
  the committed `results/expj_llm_results.json` under the `programs` key, so the corpus itself is
  documented and the published numbers are auditable from the committed result.
- **Outputs:**
  - `results/expj_results.json` -- synthetic sweep. Top-level `dmci_one_time_compile_s`, `ncg_bytes`
    (289432 = 283 KB); `cells[]` (per recursive-fraction × N: cumulative compile s, cumulative
    engineering LOC, coverage per arm); `recovery[]` (matched param-rel-error per (kind, arm)).
  - `results/expj_llm_results.json` -- LLM subset. Counts (260 = 200 closed-form + 60 recursive);
    `by_family` coverage/compile/eng-LOC per arm; `cumulative_compile` crossover curves; `programs[]`
    (every generated Scheme string); `recovery[]`.
- **Figure data:** `paper-arxiv/figures/exp_j_compile.dat` (and `paper-dmci/figures/exp_j_compile.dat`)
  hold the closed-form crossover curve `N {1,100,10000}` × `{dmci, b1, b2}` plotted in Figure
  `fig:exp_j_compile`; values are copied from `expj_results.json` (B1 591.57, B2 617.72 at N=10⁴).
- **Provenance:** no `metadata.json` / `MANIFEST.txt`. Provenance is partial: the synthetic result is
  fully regenerable from code + seed, and the LLM result embeds its corpus in-JSON, but the run host /
  date / git commit are recorded only loosely (LLM_VALIDATION.md notes "HPC job 5149546"). Recommended
  additions: commit `llm_cache/` (it is small JSON), add a `MANIFEST.txt` recording HPC job IDs,
  hostname, git SHA, and the python/torch/jax versions used for the committed JSONs.

## How to run
```bash
# Synthetic sweep - full (HPC, CPU partition, ~hours at N=10^4; B-arm compile is the cost):
sbatch experiments/exp_j/slurm_expj.sh
# equivalently:
python3 -m experiments.exp_j.run_expj --Ns 1 100 10000 --fractions 0 1 \
    --recover-sample 30 --recover-budget 30 --output-dir experiments/exp_j/results

# Synthetic sweep - small / local smoke (seconds):
python3 -m experiments.exp_j.run_expj --Ns 1 100 --fractions 0 1

# LLM external-validity subset (HPC; FIRST run needs MindRouter egress + .env API key to
# populate llm_cache/; deterministic offline only once the cache exists):
sbatch experiments/exp_j/slurm_expj_llm.sh
# equivalently:
python3 -m experiments.exp_j.run_expj_llm --n-closed 200 --n-recursive 60 \
    --recover-sample 20 --recover-budget 30 --output-dir experiments/exp_j/results
```

## Expected results
Synthetic, at N=10⁴ distinct programs (populates Table `tab:exp_j` and Figure `fig:exp_j_compile`):
- **Cumulative compile:** DMCI **0.02 s** (flat, one 283 KB `.ncg`) vs. B1 **591.6 s** / B2 **617.7 s**
  (closed-form); B2 **1287.1 s** (recursive). Crossover is immediate; gap widens without bound.
- **Coverage:** DMCI 100% both families; B1 100% closed-form / **0% recursive** (`lambdify` cannot
  ingest recursion); B2 100% both.
- **Per-program implementation burden (LOC proxy):** DMCI 0; B1 0 (closed-form) / n/a (recursive);
  B2 **56,361** (closed-form) / **78,005** (recursive).
- **Matched recovery (mean param-rel-error):** parity - closed-form ≈0.44–0.47 (B1 0.436, B2 0.445,
  DMCI 0.469), recursive ≈0.36–0.38 (B2 0.363, DMCI 0.380).

LLM subset, 260 real programs (reported in Appendix `app:exp_j`, "Results (LLM-validation)"):
- **Coverage:** DMCI 100% both; B1 **99%** closed-form (one program `lambdify` could not ingest) /
  **0%** recursive; B2 100% both.
- **Cumulative compile over 260:** DMCI **0.02 s** vs. B1 **23.2 s** / B2 **23.9 s**; B2 engineering
  **5,034** LOC.
- **Matched recovery:** closed-form parity ≈**0.088** for all three arms (n=20). **Honest negative:**
  recursive DMCI recovery **fails (inf)** within the matched budget - these LLM-authored recursive
  programs are hard/ill-identified (same barrier as Experiments F3/I); recursive *coverage* is solved,
  recursive *recovery* remains an open optimization problem.

## Environment, seeds, hardware
- **Python/libs (verified locally):** python 3.11, torch 2.5.1, jax 0.10.1, sympy 1.13.1, scipy 1.14.1.
  HPC venv is a self-contained uv `.venv`; note the slurm scripts also call `module load python/3.11.10`
  (legacy - the uv `.venv` does not require it).
- **Seeds:** synthetic corpus + data seed = 0 (fixed in `run_expj.py`); LLM corpus RNG seed = 0;
  recovery multi-start `seed=0`. `sys.setrecursionlimit(5000)` is set by both runners (needed for the
  recursive interpreter walks).
- **Hardware:** CPU `eight` partition (4 cpus-per-task); synthetic slurm requests 32 GB (the JAX arms
  hold N jitted executables at high N - the memory point the experiment measures), LLM slurm 16 GB.
- **Wall-clock:** synthetic at N=10⁴ is dominated by the B-arm per-program JAX trace/compile
  (~600 s closed-form, ~1287 s recursive B2); slurm `--time=06:00:00`. The DMCI arm and the small/local
  configs run in seconds. The LLM first run is gated on MindRouter latency (concurrent calls); cached
  reruns are fast (committed result is HPC job 5149546).
- **Verification (this audit):** `python3 -m experiments.exp_j.run_expj --Ns 1 3 --fractions 0 1`
  ran end-to-end locally and reproduced the structural story (DMCI flat compile; B1 0% recursive
  coverage; B2 full coverage + growing LOC; matched recovery at parity). A fresh local interpreter
  build yields a 373 KB `.ncg`; the committed result (and the manuscript) use 283 KB - minor
  interpreter-version drift, not a discrepancy in the published artifact.
