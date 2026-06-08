# Future Directions — Execution / Performance

Recorded 2026-05-31. Candidate performance directions for the Neural Compiler / DMCI, with an
honest read on viability, difficulty, and payoff. These are **not** scheduled work — a parking
lot for the most promising avenues, with the reasoning preserved.

## Framing: two execution paths with opposite characteristics

Most "make it faster" ideas land differently depending on which path they target:

- **Direct-compiled path** (Scheme → `ComputeGraph`, heap-free, static structure): already fast
  (measured **73–248× faster than DMCI**, Exp B/C), static-shape, fusion/XLA-amenable. *Not* the
  bottleneck.
- **DMCI interpreter path** (the heap-backed, Python-dispatched, dynamically-recursive
  meta-circular evaluator): the actual cost center, and resistant to most graph-compiler tricks
  because its control flow and heap are *dynamic*, not because they're un-tried.

Already realized (so **not** on this list): large-batch / population execution over parameters
(Exp H: 3,848× end-to-end, up to 17,000× forward throughput) and, as of v1.1.7, heap-backed
batching of the DMCI interpreter itself over data-independent input/parameter dimensions.

Empirically **ruled out** for these workloads: `torch.compile` (Inductor) on the batched
evaluator gave *no* benefit (1.02–4.2× *slower*; Exp H Part E) — the programs are 3–30-op graphs
where kernel-launch cost is already negligible and tracing/guard overhead dominates. The paper's
own conclusion: reducing single-evaluation latency needs lowering to a compiled representation
(MLIR), not a tracer on top of the Python evaluator. XLA/JAX functional AD additionally cannot
trace the interpreter at all (dict-backed heap + data-dependent recursion); the existing JAX
backend is scalar-only (`backend/jax_backend.py`).

---

## 1. Compiled VM backend (MLIR / Enzyme lowering of the evaluator) — the real DMCI accelerator

**Idea.** Lower the self-hosted evaluator to LLVM/MLIR so both forward and backward run as native
compiled code, attacking the ~14× per-evaluation *latency* (tagged-value wrapping 41% + Python
dispatch 32%) that batching amortizes but cannot remove. The paper already names this as the fix
(Limitations / §future work).

**Must be differentiable** — non-negotiable; a forward-only VM is useless for DMCI (the thesis is
`∂loss/∂θ` for constants θ in the interpreted program). The viable design is
**compile-then-differentiate with Enzyme**, *not* differentiate-then-compile (torch.compile: no
benefit; XLA: untraceable):

- Enzyme synthesizes the reverse-mode adjoint at the IR level and differentiates **imperative code
  with real control flow + mutable memory** (loops, branches, recursion, heap loads/stores) —
  exactly the interpreter's shape. The differentiability requirement is what *selects* Enzyme over
  XLA, not an obstacle.
- Requirements to stay correct: (a) **activity analysis** — float payloads "active," type tags /
  heap addresses / symbol IDs / control "inactive" (mirrors the current tagged-value design); (b)
  **shadow heap** — adjoint buffer parallel to the bump heap so gradients cross `cons`/`car`/`cdr`;
  (c) **same a.e. gradient semantics** (Theorem 2 branch-boundary measure-zero — inherited, not
  introduced); (d) **validation vs the PyTorch reference** (gradients/loss trajectories match to
  numerical precision, the bar already applied to the JAX/NumPy backends).

**Viability:** real (Enzyme-MLIR is used for differentiable simulation). **Difficulty:** high —
lowering the tagged interpreter + bump heap to MLIR and wiring Enzyme's shadow heap/activity
analysis is a genuine project. **Payoff:** the only thing that attacks DMCI's single-eval latency
(target ~1.3× vs direct compilation) while keeping exact gradients *and* the "any program as data"
property. Escape hatch: for a *fixed* program you can instead lower the *direct-compiled* graph to
XLA/MLIR (easy, fast AD) — but that's the direct path (recompile per program), abandoning the niche.

## 2. Compiler optimization passes for the direct-compiled path (const-fold, CSE, DCE, algebraic simplification)

**Idea.** Add the standard passes the compiler currently lacks (confirmed absent — only ANF + TCO):
constant folding (`(+ 2 3)`→`5`, dead `(if #t a b)`→`a`), common-subexpression elimination (ANF
already names subexpressions, so CSE is natural), dead-code elimination, algebraic simplification
(`(* x 0)`→`0`, `(+ x 0)`→`x`).

**Viability:** high. **Difficulty:** low–moderate (well-trodden; ANF makes CSE/DCE clean).
**Payoff:** **modest, and direct-path only.** These shrink the *direct-compiled* graph — the path
that is already fast. They do **not** help DMCI: there the program is *data* (a `quote_const`), not
graph nodes, so the compiler never sees it to simplify; the interpreter graph itself is compiled
once. Worth doing as low-risk polish (and it helps any emitted `torch.nn.Module`), but it is not a
headline result.

## 3. Batched program-VM — SIMT over *different* programs ("search over programs = one tensor op")

**Idea.** Execute *N different* candidate programs in lockstep as one tensorized VM:
`PC [N]`, `Env [N, E]`, `Stack [N, S]`, evolving thousands of programs per step — turning program
*search* into a single GPU computation (cf. batched RL envs, SIMT).

**Key obstacle (architectural).** v1.1.7 batches over *inputs* for *one* program *because control
flow is data-independent*. Different programs take different branches/recursion per lane — exactly
the data-dependent-control-flow case the lazy recursive interpreter now **rejects** (the v1.1.8
clear-error). The current AST-walking, one-branch-per-call interpreter fundamentally cannot do
this; it requires a **different executor**: a fixed-max-depth, masked/predicated, stack-machine
(bytecode) VM that evaluates all branches and selects per lane.

**Viability:** a ground-up second interpreter. **Difficulty:** highest. **Payoff:** highest ceiling
— genuinely novel, paper-worthy, and the natural substrate for gradient-based program search at
scale. Scope as a **separate research effort**, not an increment to the existing evaluator.

---

## Priority read

The two directions that move the **DMCI** needle are **#1 (compiled VM, latency)** and **#3
(batched program-VM, program-search throughput)** — and notably neither is "XLA on top of the
Python evaluator." **#2** is easy, low-risk polish for the already-fast direct path. Large-batch
population execution (often suggested first) is already done (Exp H + v1.1.7).
