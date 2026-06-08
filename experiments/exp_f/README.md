# Experiment F - LLM-in-the-Loop Scientific Model Discovery

A closed discovery loop between an LLM and the differentiable interpreter: the **LLM proposes the
structure** of a scientific model (a Scheme expression), and **DMCI calibrates its parameters** by
gradient descent. Because a program is runtime data that compiles straight to a differentiable
graph, "the LLM discovered a model and we trained it" happens end-to-end with no per-model code.

## The loop (`run_f_thinking.py` / original `exp_f.py`)
For each target (4 phenomena × 3 seeds), up to 5 iterations:
1. **Propose** - LLM emits a Scheme expression for the data (constrained subset: `x`, params
   `a,b,c,d`, binary ops, no `define`/`lambda`).
2. **Validate (+retry)** - parse, arity-check, compile, and evaluate for finiteness; retry on error.
3. **Fit** - compile the expression and recover its parameters (batched DMCI fit).
4. **Analyze residuals** → **Refine** - feed residuals back; LLM proposes an improved structure.
5. Stop when held-out MSE `< 1e-3`, keeping a **best-of-iterations** checkpoint.

## Key finding: the original 6/12 conflated TWO distinct failure modes

The first run (qwen3.6-35b, **no reasoning**, single-start Adam) converged 6/12. Tracing the
failures showed they were *not* one problem but two, **neither a real limit of DMCI's capability**:

| Stage (GPT-5.5) | F1 decay | F2 damped osc. | F3 decay+sine | F4 logistic | Total |
|---|---|---|---|---|---|
| Original (no-think, Adam) | 3/3 | 1/3 | 0/3 | 2/3 | **6/12** |
| **+ thinking** (structure) | 3/3 | 3/3 | 0/3 | 3/3 | **9/12** |
| **+ portfolio fitter, full loop** (parameters) | 3/3 | 3/3 | **2/3** | 3/3 | **11/12** |

*Actuals from HPC job 5149553 (`results_gpt55_think_portfolio/`). The single residual is **F3 seed1**,
which stalled at MSE 1.69e-2: the discovery loop refined *away* from a fittable structure
(`bestIter=2`) rather than the fit failing. In **isolation** - feeding the cached, correct iter-0 F3
structure straight to the multi-start L-BFGS fitter - all three F3 seeds converge to ~3–6e-4 (3/3,
job 5149544, `results_gpt55_think_multistart/`). So the residual is a discovery-loop/held-out-selection
interaction, not an inability to fit the correct structure; the parameter-optimization barrier itself
is removed.*

1. **Structure-discovery failures (F2).** Without reasoning, the model one-shot-guessed the wrong
   functional family. **Enabling thinking** (`enable_thinking`, GPT-5.5 / qwen3.6-27b) fixed F2
   (1/3 → 3/3, one iteration each). Model capability matters: qwen-27b+thinking still missed F2,
   GPT-5.5+thinking did not.
2. **Parameter-optimization failures (F3).** The decisive trace: on F3, GPT-5.5 proposed the
   **correct** structure `a·exp(−b·x) + c·sin(d·x)` on iteration 0 - and it *still* failed, because
   single-start Adam (init ≈ 1) could not recover the sine **frequency** `d≈3`. The landscape in a
   frequency parameter is strongly multimodal; Adam locked into a wrong-frequency basin (~0.4 Hz),
   and the residual-feedback loop then *ping-ponged* between structures, mis-diagnosing a fitting
   failure as a structure failure. A **multi-start L-BFGS** fitter recovered `d` and converged F3
   **3/3** at ~3e-4 - same structure, better optimizer.

## Why a multi-optimizer strategy is needed (and what it is)

**Different program structures induce different loss landscapes, and no single optimizer dominates:**
- smooth/well-conditioned (F1, F4) → local Adam converges instantly;
- multimodal (F3 sine frequency) → needs **broad multi-start L-BFGS** (exact DMCI gradients);
- rugged / high-dimensional (cf. Experiment~I, 18–24 params) → favors **gradient-free** global
  search (differential evolution), where gradient methods overfit or stall.

Because the LLM emits the structure **at runtime**, you cannot hand-pick the solver per model - so
run a **portfolio** and keep the best. Implemented in `portfolio.py`:

- **Compile once per structure**, share the graph across all solvers (the structure is the cost).
- **Cheap-first cascade with early-exit** (the default): local Adam → multi-start L-BFGS → [DE].
  Easy structures stop after cheap Adam; only hard ones pay the expensive multi-start L-BFGS.
- **Held-out selection** - split train/val, judge convergence and pick the winner on **validation**
  MSE. *Essential*: selecting on training MSE picks the most-overfit solver (the Experiment~I
  lesson, where L-BFGS drove train error to ~1e-12 but generalized worse than DE at high dim).
- **Diagnostic**: records which solver won, distinguishing a *hard landscape* (some solver succeeds)
  from a *wrong/unidentifiable structure* (all fail) - exactly the distinction the single-Adam loop
  could not make.

**Why not run the solvers concurrently?** We tried (`concurrent=True`). It is the wrong default
here: the solvers are **CPU/GIL-bound** (DMCI evaluation is Python-level and holds the GIL), so
threads interleave rather than parallelize (wall ≈ sum, not max), and concurrent-all pays the
expensive solver on *every* structure instead of only the hard ones. Thread concurrency pays off
for the **I/O-bound LLM calls** in the loop (which already run 8-wide), not for these solvers. The
cheap-first cascade gives the same best-solver robustness at a fraction of the cost. DE lives in an
opt-in `DEEP_PORTFOLIO` because it is expensive (thousands of evals/fit) and only earns its cost on
rugged/high-dim landscapes.

## Cross-experiment link
F3's bottleneck **is** Experiment~I's domain (continuous parameter optimization on hard landscapes),
and I's mitigation - reparameterization + multi-start L-BFGS - is precisely what cures F3. The
portfolio unifies them: the same fitting machinery serves both.

## Files & running
- `exp_f.py` - original single-Adam discovery loop + shared helpers (data, validation, residuals).
- `llm_providers.py` - multi-provider client (MindRouter qwen / OpenAI gpt-5.5), thinking mode,
  `max_completion_tokens` (16384 for MindRouter to avoid thinking-trace timeouts, 32768 for OpenAI).
- `run_f_thinking.py` - concurrent (8-way over target×seed) thinking-mode runner; `--fitter
  {adam,multistart,portfolio}`, `--targets`, best-of-iterations checkpoint, batched fit.
- `portfolio.py` - the parameterized multi-optimizer portfolio (`Solver`, `DEFAULT_PORTFOLIO`,
  `DEEP_PORTFOLIO`, `fit_portfolio`).

```bash
# thinking-mode discovery, portfolio fitter, on HPC (8-way):
sbatch --export=ALL,SPECS="gpt55_think",FITTER="portfolio" experiments/exp_f/slurm_f_thinking.sh
```
Results → `experiments/exp_f/results_<label>[_<fitter>]/`.
