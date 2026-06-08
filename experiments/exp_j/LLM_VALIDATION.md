# Exp J - LLM-Generated Validation Subset (External Validity)

**Purpose.** The headline Exp J curves use a synthetic, reproducible program space (`corpus.py`):
structurally-distinct random expression trees and damped-Euler relaxations. A fair reviewer asks
whether that synthetic distribution is *representative of programs an LLM actually emits*. This
subset answers that empirically: the programs here are authored by an LLM and run through the
**identical three arms**, so we can check that the cost/coverage/recovery story is not an artifact
of the synthetic generator.

This is an **external-validity check, not a scale claim** - it runs a few hundred real programs, not
10⁴. The scale crossover comes from the synthetic corpus; this subset confirms the *shape* of the
result holds on genuine LLM output.

## Generation
- **Model / endpoint.** MindRouter (University of Idaho OpenAI-compatible endpoint),
  `qwen/qwen3.6-27b`, temperature 0.7. Same provider as Experiment F.
- **Reasoning enabled.** Generation runs with thinking ON (`chat_template_kwargs={"enable_thinking":
  true}`) and `max_completion_tokens=16384` (capped below Experiment F's 32768 to bound the
  thinking-trace length and avoid endpoint timeouts), so the validation corpus is produced under the
  same reasoning regime.
- **Caching.** Every response is cached verbatim under `experiments/exp_j/llm_cache/` keyed by the
  SHA-256 of `(system_prompt, user_prompt)`. The first run (with egress) populates the cache; all
  subsequent runs are **deterministic and offline**, so the measurement is reproducible and not
  gated on live LLM latency or endpoint drift. The raw `llm_cache/` is regenerated, not committed;
  the authoritative record is the committed `results/expj_llm_results.json`, which preserves each
  generated program verbatim, so every reported number is auditable offline from the repository.
- **Two families requested** (`llm_corpus.py`):
  - *Closed-form* `y = f(x; a,b,c,d)` over `{+,-,*,/,exp,log,sin,cos,pow,sqrt,abs}`, no
    `define`/`if`. (System prompt mirrors Experiment F's, with named-parameter and binary-arity
    constraints.)
  - *Recursive/iterative* - a tail-recursive `define` with a fixed step counter (10–20 steps),
    e.g. a damped relaxation. This is exactly the structure `lambdify` cannot ingest.
- **Diversity.** Each request is seeded with a distinct phenomenon hint (16 scientific motifs:
  decay, logistic, damped oscillation, resonance, power law, Michaelis–Menten, …) plus a "use a
  different functional form" nudge on retries. Programs are **de-duplicated by canonical string**
  and **rejection-tested for finiteness** over `x ∈ [0,2]`, identically to the synthetic corpus.
- **Classification is by what parses, not by what we asked for.** A returned program is parsed and
  routed by `is_closed_form` (closed-form vs. recursive/non-closed), so LLM disobedience degrades
  gracefully rather than corrupting a family.

## Ground truth (honest scoping)
- **Closed-form** programs use an **independent Python evaluator** (`sexp_eval.eval_python`) for
  ground-truth data, so no arm is favored - DMCI, B1 (lambdify), and B2 (the JAX evaluator) all fit
  the same evaluator-independent targets.
- **Recursive** programs use **DMCI** to generate ground truth (it is the only arm that executes
  them in this subset). Because the forward function is mathematically evaluator-independent, this
  does not advantage DMCI's *recovery* - recovery starts from random restarts and fits the same
  data any correct evaluator would produce.

## What each arm does (unchanged from the synthetic experiment)
| Arm | Closed-form | Recursive |
|-----|-------------|-----------|
| **DMCI** | one interpreter compile (amortized), `evaluate_program` per program, 100% coverage | same - 100% coverage |
| **B1** (`lambdify→jax.grad`) | parses & jits per program, ~0 engineering | **coverage failure** (`lambdify` cannot ingest the loop) |
| **B2** (hand-port to JAX) | auto-emitted JAX forward, engineering = AST node count | covered *by construction* (a human can always port); engineering = token count + loop scaffolding, **but the port is not auto-emitted** here, so B2 recovery is reported only on the closed-form family |

## What it measures (per arm, split by family)
1. **Coverage** - fraction evaluable with no human intervention. Expectation: DMCI = 100% on both
   families; B1 = 100% on closed-form but **0% on recursive**; B2 = 100% on both.
2. **Cumulative compile time** + the **crossover N** at which the per-program-compile arms exceed
   DMCI's single one-time interpreter compile.
3. **Per-structure engineering** (LOC proxy): DMCI = 0; B2 > 0; B1 = 0 (closed-form) / coverage-fail
   (recursive).
4. **Matched recovery error** - shared scipy L-BFGS-B multi-start in log-reparameterized
   coordinates, so the comparison is about cost, not accuracy. (Closed-form: parity with B1 expected,
   as pre-committed.)

## Results (HPC job 5149546; `results/expj_llm_results.json`)
260 real MindRouter (qwen3.6-27b, thinking) programs - 200 closed-form + 60 recursive - through the
three arms. The synthetic-corpus story reproduces on genuine LLM output:

- **Coverage:** DMCI **100%** on both families. B1 (`lambdify`) **99%** on closed-form (one real LLM
  program it could not ingest) and **0% on recursive** - the coverage collapse, confirmed on real
  output. B2 **100%** on both.
- **Cumulative compile (260 programs):** DMCI **0.02 s** (one interpreter compile, flat) vs. B1
  **23.2 s** / B2 **23.9 s**; the per-program-compile arms cross DMCI's one-time cost at N=1.
- **Per-structure engineering:** DMCI **0 LOC**; B2 **5,034 LOC** (token-count proxy over 260 programs).
- **Matched recovery (mean param-rel-error):** closed-form is at **parity** - DMCI 0.088, B1 0.088,
  B2 0.088 (n=20). **Honest negative:** on the recursive family, DMCI's matched L-BFGS recovery
  **fails (inf)** within budget - these LLM-authored recursive programs have hard,
  often ill-identified landscapes, the same parameter-optimization barrier seen in Experiments F3/I.
  (B1 has no coverage there; B2's recursive port is costed, not auto-emitted, so recovery is reported
  only for DMCI.) So coverage and amortized-compile claims hold on real output; recursive *recovery*
  remains an open optimization problem, not a coverage one.

## Reproduce
```bash
sbatch experiments/exp_j/slurm_expj_llm.sh            # HPC; first run needs MindRouter egress
python3 -m experiments.exp_j.run_expj_llm --n-closed 200 --n-recursive 60
```
Output → `experiments/exp_j/results/expj_llm_results.json` (per-family coverage, cumulative-compile
curves + crossover, engineering LOC, matched-recovery summary, and the full list of generated
programs with their Scheme strings).

## Files
`llm_corpus.py` (cached MindRouter generation + Program construction), `sexp_eval.py` (generic
closed-form Scheme→Python/JAX evaluator + size proxies), `run_expj_llm.py` (the runner),
`slurm_expj_llm.sh`. Reuses Experiment F's `llm_client`/`_parse_sexp`/`detect_used_params` and the
three arms in `arms.py`.
