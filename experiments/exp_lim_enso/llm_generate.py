############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# llm_generate.py: LLM program-as-data generator for the LIM-ENSO experiment (Track B2). This is the program-as-data half of the...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""LLM program-as-data generator for the LIM-ENSO experiment (Track B2).

This is the program-as-data half of the P3 thesis: a real LLM (U-Idaho MindRouter,
OpenAI-compatible, model ``qwen/qwen3.6-35b``) authors the Scheme Kalman-NLL program
that DMCI then folds and differentiates. The LLM's ONLY degree of freedom is the
per-structure F-assembly COMBINE-ALGEBRA over named bound factor inputs (e.g. ``Dm``,
``U``, ``V`` for S3; ``dvec`` for S1); the Kalman filter skeleton, the binding contract,
and the op surface are pinned by the SYSTEM_PROMPT so the same program is reusable per
``(structure, D, T)`` with only the bound tensors changing.

The generated program is accepted ONLY after a three-gate VALIDATE -> REPAIR loop that
mirrors the numerical gate (gate.py) at a tiny (D, T):

  GATE1  static unsupported-op prescan (neural_compiler.dmci.unsupported_interpreter_ops):
         catches the silent-evaluate-to-0 footgun BEFORE any eval. An op the interpreter
         lacks (e.g. ``diag``, ``list``, a typo) hits eval-apply's ``(#t 0)`` fallthrough
         and corrupts results silently, so we reject it statically first.
  GATE2  compile_dmci succeeds (the program is well-formed for the meta-circular path).
  GATE3  forward PARITY vs the float32 numpy reference twin (reference.reference_nll_lim)
         AND a NONZERO gradient flowing back to the bound factor inputs, on random
         F/Q/R/obs. A program that compiles but computes the wrong F (or whose F does not
         depend on its factor inputs) is rejected here.

On any gate failure the loop feeds a TARGETED repair hint back to the LLM and retries
(MAX_REPAIRS=3). If the budget is exhausted the structure is recorded honestly as
``status="LLM-failed"`` -- we NEVER silently substitute the hand-written templater
(models.kalman_lim_src); a failed structure simply has no accepted LLM program.

Accepted programs are cached to ``experiments/exp_lim_enso/llm_cache/<structure>.json``
(one MindRouter hit per structure -> reproducible reruns load the cache).

The MindRouter call + _extract_scheme + .env/creds + base_url + model are reused VERBATIM
from experiments/exp_b/llm_sources.py.

CLI:
  python -m experiments.exp_lim_enso.llm_generate --structure S1
  python -m experiments.exp_lim_enso.llm_generate --all
  python -m experiments.exp_lim_enso.llm_generate --structure S1 --offline --source path/to/prog.scm
  python -m experiments.exp_lim_enso.llm_generate --structure S1 --offline   # validate the cache

``--offline`` runs the VALIDATE -> REPAIR gate machinery against a cached or hand-provided
program WITHOUT calling the API, so the gating logic is exercised even where MindRouter is
unreachable (campus-only / VPN). Build the cache/HPC run on-campus; validate anywhere.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import dotenv

# Project root .env (same file Exp B reads): MINDROUTER_BASE_URL / _API_KEY / _MODEL.
_ROOT = Path(__file__).resolve().parent.parent.parent
dotenv.load_dotenv(_ROOT / ".env")

# Raise the recursion limit BEFORE importing neural_compiler (MEMORY: DMCI compile/eval
# machinery recurses even though the LIM filter itself trampolines to O(T)).
from .config import DEFAULT  # config has no heavy import

sys.setrecursionlimit(DEFAULT.recursion_limit)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from . import params, reference  # noqa: E402


CACHE_DIR = Path(__file__).parent / "llm_cache"

# Validation problem size: tiny so the gate is cheap, big enough that F-assembly,
# the matmul/inv/logdet pipeline, and a multi-step accumulation are all exercised.
VALIDATE_D = 4
VALIDATE_T = 5

MAX_REPAIRS = 3

# Parity tolerance for GATE3 (float32 DMCI vs float32 twin, identical arithmetic order).
# Mirrors gate.GATE.parity_rel spirit; a touch looser since this is a generation gate.
PARITY_REL = 3e-3
GRAD_FLOOR = 1e-8   # the summed factor-gradient norm must clear this to count as "nonzero".


# ===========================================================================
# SYSTEM PROMPT -- Kalman/tensor-aware. DIFFERENT from Exp B's (which FORBIDS
# vec/mat and restricts to scalar numeric recursion). Here the tensor ops ARE
# the point, and the LLM authors ONLY the F-assembly combine-algebra.
# ===========================================================================

SYSTEM_PROMPT = """\
You are an expert in Scheme and linear-Gaussian state-space models. You write the \
Scheme source for a Kalman-filter negative-log-likelihood (NLL) over a Linear Inverse \
Model (LIM), to be run THROUGH a differentiable meta-circular interpreter that operates \
on tensor-valued payloads. Emit ONLY Scheme code, no prose, no markdown fences.

THE INTERPRETER'S OPERATOR SURFACE (use ONLY these; anything else silently evaluates to 0):
- arithmetic (broadcasting elementwise over scalars/vectors/matrices): + - * /
- scalar math: sin cos exp sqrt log abs pow min max
- comparison: = < > <= >=
- tensor ops: vec mat ref dot cross norm normalize vsum vlen scale matvec matmul \
transpose trace det logdet inv outer eye zeros ones
- special forms: quote if cond let letrec lambda begin define
- the tail-call form: (recur ...) inside a (loop (...) body) header
There is NO `diag`, NO `list`-of-numbers math, NO `set!`, NO `call/cc`, NO named let.

HARD RULES:
1. ALL arithmetic operators take EXACTLY TWO arguments. Write (+ a (+ b c)), never (+ a b c). \
Same for - * /.
2. Build the identity with (eye D) and a scaled identity with (scale s (eye D)); the zero \
vector is (zeros D). NEVER write a D-by-D or length-D numeric literal; NEVER inline matrix \
constants. D is a fixed integer given to you; substitute it literally where the skeleton uses it.
3. Use (logdet S), NEVER (log (det S)) -- det underflows in float32 and corrupts the log-det term.
4. (ref obs k) gathers row k of the observation matrix as a vector; obs is data-independent \
of the parameters. Do not transform obs.
5. The whole filter is ONE tail-recursive loop using the (loop ((k 0) ...) ...) header and \
(recur (+ k 1) ...) in tail position. Do not use named let or non-tail recursion.

THE F-ASSEMBLY: this is the ONLY part that varies. You receive the NAMES of the bound factor \
inputs and an algebraic spec for the transition operator F in terms of them. Assemble F with \
the tensor ops above (matmul, transpose, +, *, eye, scale, outer ...). For a DIAGONAL F from a \
length-D vector v, use (* (eye D) v) -- elementwise multiply broadcasts v across the identity's \
columns, leaving v on the diagonal. NEVER bind F to a literal; ALWAYS build it from the factors.

THE FILTER SKELETON: reproduce this EXACTLY, substituting the integer D for {D} and T for {T}, \
and substituting your F-assembly bindings for the `;; F-ASSEMBLY HERE` line (they go FIRST in \
the let*, before xpred, so later bindings can use F):

(loop ((k 0)
       (x  (zeros {D}))
       (P  (eye {D}))
       (L  0.0))
  (if (= k {T})
      L
      (let* (;; F-ASSEMBLY HERE: bind F from the factor inputs (see spec)
             (xpred (matvec F x))
             (Ppred (+ (matmul (matmul F P) (transpose F)) Q))
             (y     (ref obs k))
             (e     (- y xpred))
             (S     (+ Ppred R))
             (Sinv  (inv S))
             (Kg    (matmul Ppred Sinv))
             (xnew  (+ xpred (matvec Kg e)))
             (Pnew  (matmul (- (eye {D}) Kg) Ppred))
             (nll   (+ (logdet S) (dot e (matvec Sinv e)))))
        (recur (+ k 1) xnew Pnew (+ L nll)))))

The factor inputs, Q, R, and obs are all bound at evaluation time; they are the program's \
free variables. Output the complete program -- the single (loop ...) form -- and nothing else.
"""


# ===========================================================================
# Per-structure factor-binding CONTRACT.
# ===========================================================================
# Each structure declares:
#   free_factor_names : the factor input symbols the LLM's F-assembly must consume
#                       (besides the fixed Q, R, obs). Their order is the order GATE3 binds.
#   f_spec            : the natural-language algebraic spec handed to the LLM (the ONLY
#                       per-structure variation in the user prompt).
#   make_factors(D)   : build random {name: torch leaf tensor} factor inputs for GATE3.
#   build_F(factors)  : the reference twin's F from those SAME factors (must match the
#                       combine-algebra the LLM is asked to emit). Used by the float32 twin.
# All of S0/S1/S3/S4/S5 are authored: build_F reduces the bound factors with the SAME algebra
# the LLM is asked to emit, so the float32 twin validates the LLM's in-program F. (A structure
# with build_F=None would be reported 'unsupported' rather than run.)

@dataclass
class StructureContract:
    name: str
    free_factor_names: list[str]
    f_spec: str
    make_factors: object       # (D:int, seed:int) -> dict[str, torch.Tensor leaf]
    build_F: object | None     # (factors:dict) -> torch.Tensor [D,D]  (None if unsupported)


def _factors_S0(D: int, seed: int):
    raw = params.init_raw_params(D, "S0", seed=seed)
    F = params.make_F(raw["F_raw"], D, "S0").detach().clone().requires_grad_(True)
    return {"F": F}


def _factors_S1(D: int, seed: int):
    # Factor names/shapes MUST match models.F_FACTOR_INPUTS / params.make_F_factors so the
    # canonical kalman_lim_src prelude validates: Dvec = length-D diagonal vector, F=(* (eye D) Dvec).
    fac = params.make_F_factors(params.init_raw_params(D, "S1", seed=seed)["F_raw"], D, "S1")
    return {"Dvec": fac["Dvec"].detach().clone().requires_grad_(True)}


def _factors_S3(D: int, seed: int):
    fac = params.make_F_factors(params.init_raw_params(D, "S3", seed=seed)["F_raw"], D, "S3")
    return {k: fac[k].detach().clone().requires_grad_(True) for k in ("Dvec", "U", "V")}


def _factors_S4(D: int, seed: int):
    # AR(2) companion of the leading mode, realized as (+ (outer e0 arow) Sub):
    #   arow [D] = [a1, a2, 0..0] (only a1,a2 free leaves), e0 [D] / Sub [D,D] fixed structure.
    fac = params.make_F_factors(params.init_raw_params(D, "S4", seed=seed)["F_raw"], D, "S4")
    arow = fac["arow"].detach().clone().requires_grad_(True)
    e0 = fac["e0"].detach().clone()                       # constant structural factor
    Sub = fac["Sub"].detach().clone()                     # constant structural factor
    return {"arow": arow, "e0": e0, "Sub": Sub}


def _factors_S5(D: int, seed: int):
    # sym/antisym re-parametrization of a full M: F = 0.5(M+M^T) + 0.5(M-M^T).
    fac = params.make_F_factors(params.init_raw_params(D, "S5", seed=seed)["F_raw"], D, "S5")
    M = fac["M"].detach().clone().requires_grad_(True)
    return {"M": M}


CONTRACTS: dict[str, StructureContract] = {
    "S0": StructureContract(
        name="S0",
        free_factor_names=["F"],
        f_spec=("F is bound DIRECTLY as the matrix input `F` (no assembly needed). "
                "Use F as-is; do not rebuild it."),
        make_factors=_factors_S0,
        build_F=lambda f: f["F"],
    ),
    "S1": StructureContract(
        name="S1",
        free_factor_names=["Dvec"],
        f_spec=("F is DIAGONAL. The factor input `Dvec` is a length-D vector holding the "
                "diagonal. Assemble F = diag(Dvec) with (* (eye D) Dvec)."),
        make_factors=_factors_S1,
        build_F=lambda f: torch.eye(f["Dvec"].shape[-1], dtype=f["Dvec"].dtype)
                          * f["Dvec"].unsqueeze(-2),
    ),
    "S3": StructureContract(
        name="S3",
        free_factor_names=["Dvec", "U", "V"],
        f_spec=("F = diag(Dvec) + U V^T (low-rank-plus-diagonal). The factor inputs are the "
                "length-D diagonal vector `Dvec`, and two thin matrices `U` and `V` (each "
                "[D,r]). Assemble F = (+ (* (eye D) Dvec) (matmul U (transpose V)))."),
        make_factors=_factors_S3,
        build_F=lambda f: (torch.eye(f["Dvec"].shape[-1], dtype=f["Dvec"].dtype)
                           * f["Dvec"].unsqueeze(-2)) + f["U"] @ f["V"].transpose(-1, -2),
    ),
    # S4 / S5: factor-assembly is authored (params.make_F + models._f_assembly_prelude); the
    # build_F twin below reduces the SAME factors with the SAME algebra the LLM is asked to emit.
    "S4": StructureContract(
        name="S4",
        free_factor_names=["arow", "e0", "Sub"],
        f_spec=("F is the AR(2) COMPANION of the leading mode. The factor inputs are `arow` "
                "(a length-D vector whose first two entries are the AR coefficients a1,a2 and "
                "the rest are 0), the first basis vector `e0` [D], and the fixed delay-line "
                "matrix `Sub` [D,D] (1's on the first subdiagonal). Assemble "
                "F = (+ (outer e0 arow) Sub) -- outer puts arow in row 0; Sub shifts history."),
        make_factors=_factors_S4,
        build_F=lambda f: (f["e0"].unsqueeze(-1) * f["arow"].unsqueeze(-2)) + f["Sub"],
    ),
    "S5": StructureContract(
        name="S5",
        free_factor_names=["M"],
        f_spec=("F is the SYMMETRIC + ANTISYMMETRIC split of a full matrix `M` [D,D] (the LIM "
                "normal/non-normal decomposition). Bind Ssym = (scale 0.5 (+ M (transpose M))) "
                "and Aanti = (scale 0.5 (- M (transpose M))), then F = (+ Ssym Aanti)."),
        make_factors=_factors_S5,
        build_F=lambda f: 0.5 * (f["M"] + f["M"].transpose(-1, -2))
                          + 0.5 * (f["M"] - f["M"].transpose(-1, -2)),
    ),
}


# ===========================================================================
# MindRouter call + _extract_scheme -- reused VERBATIM from exp_b/llm_sources.py.
# ===========================================================================

def _call_mindrouter(prompt: str, system: str = SYSTEM_PROMPT) -> str:
    import openai
    base_url = os.environ.get("MINDROUTER_BASE_URL", "https://mindrouter.uidaho.edu/v1")
    api_key = os.environ.get("MINDROUTER_API_KEY")
    model = os.environ.get("MINDROUTER_MODEL", "qwen/qwen3.6-35b")
    if not api_key:
        raise ValueError("MINDROUTER_API_KEY not set. Add it to .env or environment.")
    client = openai.OpenAI(base_url=base_url, api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        max_tokens=8192,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    )
    content = response.choices[0].message.content
    if not content:
        content = getattr(response.choices[0].message, "reasoning_content", None)
    if not content:
        raise ValueError("Empty response from MindRouter")
    return content


def _extract_scheme(response: str) -> str:
    code_match = re.search(r"```(?:scheme|lisp)?\s*\n(.*?)```", response, re.DOTALL)
    if code_match:
        return code_match.group(1).strip()
    return response.strip()


# ===========================================================================
# Prompts.
# ===========================================================================

def _user_prompt(structure: str, D: int, T: int) -> str:
    c = CONTRACTS[structure]
    factor_list = ", ".join(c.free_factor_names) if c.free_factor_names else "(none)"
    return (
        f"Write the LIM Kalman-NLL Scheme program for structure {structure} at D={D}, T={T}.\n\n"
        f"F-assembly spec: {c.f_spec}\n\n"
        f"Bound factor input names (the program's free variables, besides Q, R, obs): "
        f"{factor_list}.\n"
        f"Q [{D},{D}], R [{D},{D}], and obs [{T},{D}] are also bound at evaluation time.\n\n"
        f"Substitute D={D} and T={T} literally into the skeleton. Emit ONLY the single "
        f"(loop ...) program."
    )


# ===========================================================================
# VALIDATE -> REPAIR gate (GATE1 static, GATE2 compile, GATE3 parity+grad).
# ===========================================================================

@dataclass
class GateResult:
    ok: bool
    gate: str                # which gate ran last: 'GATE1'|'GATE2'|'GATE3'|'OK'
    detail: str              # human-readable pass/fail detail
    parity_abs: float | None = None
    grad_norm: float | None = None
    repair_hint: str | None = None   # targeted hint to feed the LLM on failure


def _validate_program(structure: str, source: str,
                      D: int = VALIDATE_D, T: int = VALIDATE_T) -> GateResult:
    """Run GATE1 (static op prescan) -> GATE2 (compile) -> GATE3 (parity + nonzero
    factor gradient vs the float32 reference twin) on ``source``. Returns a GateResult
    carrying a targeted repair hint on the first failing gate."""
    from neural_compiler.dmci import (
        compile_dmci, as_matrix, as_vector, unsupported_interpreter_ops)

    c = CONTRACTS[structure]
    if c.build_F is None:
        return GateResult(False, "GATE0",
                          f"{structure} has no authored reference twin (params.make_F "
                          f"raises NotImplementedError) -- cannot validate.")

    # ---- GATE1: static unsupported-op prescan (catch silent-0 BEFORE any eval) ----
    try:
        unknown = unsupported_interpreter_ops(source)
    except Exception as ex:  # malformed datum / parse failure
        return GateResult(False, "GATE1", f"prescan failed to parse program: {ex!r}",
                          repair_hint=("The program could not be parsed as Scheme. Re-emit a "
                                       "single well-formed (loop ...) form with balanced parens."))
    if unknown:
        bad = sorted(unknown)
        return GateResult(
            False, "GATE1",
            f"unsupported operator(s) {bad} would silently evaluate to 0",
            repair_hint=(
                f"You used operator(s) {bad} the interpreter does not implement; they would "
                f"silently evaluate to 0. Remove them and use ONLY the listed ops. In particular "
                f"there is no `diag` (use (* (eye {D}) v)), no `list`/`null?` numeric math, no "
                f"named let, and every +,-,*,/ takes exactly TWO arguments."))

    # ---- GATE2: compile_dmci succeeds ----
    try:
        graph = compile_dmci(source)
    except Exception as ex:
        return GateResult(
            False, "GATE2", f"compile_dmci raised: {ex!r}",
            repair_hint=(f"compile_dmci failed: {ex}. Ensure the program is a single (loop ...) "
                         f"form, every operator is binary where required, and you used the exact "
                         f"skeleton with D={D}, T={T} substituted as integers."))

    # ---- GATE3: forward parity + nonzero factor gradient vs the float32 twin ----
    from neural_compiler.evaluator import evaluate
    from neural_compiler.runtime.tagged_value import unwrap_number

    torch.manual_seed(0)
    factors = c.make_factors(D, seed=0)           # leaf tensors with requires_grad
    # PD-by-construction Q, R and a benign obs window (decoupled from the factor params).
    rawqr = params.init_raw_params(D, structure, seed=0)
    Q = params.make_Q(rawqr["Lq_raw"].detach(), D)
    R = params.make_R(rawqr["r_raw"].detach(), D)
    obs = torch.randn(T, D, generator=torch.Generator().manual_seed(7)).float()

    # Bind factor inputs by their declared feature rank: a length-D vector -> as_vector,
    # a [D,*] matrix -> as_matrix.
    bindings = {}
    for name in c.free_factor_names:
        t = factors[name].to(torch.float32)
        if t.dim() == 1:
            bindings[name] = as_vector(t)
        else:
            bindings[name] = as_matrix(t)
    bindings["Q"] = as_matrix(Q.to(torch.float32))
    bindings["R"] = as_matrix(R.to(torch.float32))
    bindings["obs"] = as_matrix(obs)

    try:
        out = evaluate(graph, bindings, **DEFAULT.EVAL_KW)
        dmci_val = unwrap_number(out)
    except Exception as ex:
        return GateResult(
            False, "GATE3", f"DMCI evaluation raised: {ex!r}",
            repair_hint=(f"The program compiled but failed to evaluate: {ex}. Check that F is "
                         f"assembled from the factor inputs and is [{D},{D}], and that obs is "
                         f"indexed with (ref obs k)."))
    dmci_f = float(dmci_val)

    # reference twin: build F from the SAME factors, run the float32 numpy twin.
    F_ref = c.build_F(factors)
    ref32 = float(np.asarray(
        reference.reference_nll_lim(F_ref, Q, R, obs, D, T, dtype=np.float32)).reshape(()))

    if not np.isfinite(dmci_f):
        return GateResult(False, "GATE3", f"DMCI NLL is not finite ({dmci_f})",
                          parity_abs=None, grad_norm=None,
                          repair_hint=("The DMCI NLL came out non-finite. Make sure you use "
                                       "(logdet S) not (log (det S)), and assemble F (not a "
                                       "literal) so the covariance stays positive-definite."))

    parity_abs = abs(dmci_f - ref32)
    tol = PARITY_REL * max(1.0, abs(ref32))
    if parity_abs > tol:
        return GateResult(
            False, "GATE3",
            f"forward parity failed: |dmci-ref| = {parity_abs:.4e} > tol {tol:.4e} "
            f"(dmci={dmci_f:.6g}, ref={ref32:.6g})",
            parity_abs=parity_abs,
            repair_hint=(
                "The program computes a DIFFERENT value than the reference Kalman filter. The "
                "most common cause is the F-assembly: re-read the F-assembly spec and rebuild F "
                "exactly from the factor inputs. Also keep the filter skeleton VERBATIM (op order "
                "matters: Ppred is (matmul (matmul F P) (transpose F)), the Mahalanobis term is "
                "(dot e (matvec Sinv e)))."))

    # nonzero gradient back to the factor inputs (the program must actually depend on them).
    try:
        dmci_val.backward()
    except Exception as ex:
        return GateResult(False, "GATE3", f"backward raised: {ex!r}", parity_abs=parity_abs,
                          repair_hint=("Autograd could not differentiate the program; ensure F is "
                                       "built from the factor inputs via the tensor ops."))
    gnorm = 0.0
    for name in c.free_factor_names:
        g = factors[name].grad
        if g is not None:
            gnorm += float(g.detach().norm())
    if gnorm <= GRAD_FLOOR:
        return GateResult(
            False, "GATE3", f"factor gradient ~0 (norm={gnorm:.3e}): F does not depend on factors",
            parity_abs=parity_abs, grad_norm=gnorm,
            repair_hint=("F does not actually depend on the factor inputs (gradient is zero). "
                         "Assemble F FROM the factor names given; do not bind F to a constant "
                         "or to (eye D)."))

    return GateResult(True, "OK",
                      f"parity_abs={parity_abs:.3e} (tol {tol:.3e}), factor_grad_norm={gnorm:.3e}",
                      parity_abs=parity_abs, grad_norm=gnorm)


# ===========================================================================
# Cache I/O.
# ===========================================================================

@dataclass
class StructureProgram:
    structure: str
    status: str                 # 'accepted' | 'LLM-failed' | 'unsupported'
    source: str | None
    n_attempts: int             # number of MindRouter hits (0 for offline)
    validation: dict | None     # GateResult of the accepted/last program
    model: str | None           # MindRouter model id (None offline)
    history: list                # list of {attempt, gate, detail} for each failed try


def _cache_path(structure: str) -> Path:
    return CACHE_DIR / f"{structure}.json"


def save_to_cache(prog: StructureProgram) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(prog.structure)
    with open(path, "w") as f:
        json.dump(asdict(prog), f, indent=2)
    return path


def load_from_cache(structure: str) -> StructureProgram | None:
    path = _cache_path(structure)
    if not path.exists():
        return None
    with open(path) as f:
        d = json.load(f)
    return StructureProgram(**d)


# ===========================================================================
# Generation: one MindRouter hit + VALIDATE -> REPAIR loop.
# ===========================================================================

def generate_structure(structure: str, D: int = VALIDATE_D, T: int = VALIDATE_T,
                       max_repairs: int = MAX_REPAIRS,
                       verbose: bool = True) -> StructureProgram:
    """Generate (and cache) the LLM Scheme program for ``structure``.

    Calls MindRouter, extracts Scheme, then runs the GATE1->GATE2->GATE3 validate loop;
    on failure it feeds the targeted repair hint back to the LLM and retries up to
    ``max_repairs`` times. Returns a StructureProgram with status:
      'accepted'    -- a program passed all three gates (cached).
      'LLM-failed'  -- the repair budget was exhausted (NOT cached as accepted; recorded).
      'unsupported' -- the structure has no authored reference twin (build_F is None; none
                       currently -- S0/S1/S3/S4/S5 are all authored).
    """
    if structure not in CONTRACTS:
        raise ValueError(f"unknown structure {structure!r}; known: {sorted(CONTRACTS)}")
    c = CONTRACTS[structure]
    model = os.environ.get("MINDROUTER_MODEL", "qwen/qwen3.6-35b")

    if c.build_F is None:
        prog = StructureProgram(structure, "unsupported", None, 0,
                                {"detail": f"{structure} factor-assembly not authored "
                                           f"(params.make_F raises NotImplementedError)"},
                                model, [])
        save_to_cache(prog)
        return prog

    history: list = []
    user = _user_prompt(structure, D, T)
    last_gate: GateResult | None = None

    for attempt in range(1, max_repairs + 2):   # 1 initial + max_repairs repairs
        prompt = user
        if last_gate is not None and last_gate.repair_hint:
            prompt = (user + "\n\nYOUR PREVIOUS ATTEMPT FAILED validation:\n"
                      f"  {last_gate.detail}\nREPAIR INSTRUCTION: {last_gate.repair_hint}\n"
                      "Re-emit the COMPLETE corrected program (single (loop ...) form), nothing else.")
        if verbose:
            print(f"[{structure}] MindRouter attempt {attempt} ...", flush=True)
        raw = _call_mindrouter(prompt)
        source = _extract_scheme(raw)
        gate = _validate_program(structure, source, D, T)
        if verbose:
            print(f"[{structure}]   {gate.gate}: {'PASS' if gate.ok else 'FAIL'} -- {gate.detail}",
                  flush=True)
        if gate.ok:
            prog = StructureProgram(structure, "accepted", source.strip() + "\n", attempt,
                                    asdict(gate), model, history)
            save_to_cache(prog)
            return prog
        history.append({"attempt": attempt, "gate": gate.gate, "detail": gate.detail,
                        "source": source.strip() + "\n"})
        last_gate = gate

    # repair budget exhausted -- record honestly, do NOT substitute the hand-written templater.
    prog = StructureProgram(structure, "LLM-failed", None, max_repairs + 1,
                            asdict(last_gate) if last_gate else None, model, history)
    save_to_cache(prog)
    return prog


def validate_offline(structure: str, source: str | None = None,
                     D: int = VALIDATE_D, T: int = VALIDATE_T) -> GateResult:
    """Run the VALIDATE gate (GATE1->2->3) on a cached or hand-provided program WITHOUT
    any API call. ``source=None`` validates the currently cached accepted program (proving
    the cache is sound and the gating logic works where MindRouter is unreachable)."""
    if structure not in CONTRACTS:
        raise ValueError(f"unknown structure {structure!r}; known: {sorted(CONTRACTS)}")
    if source is None:
        cached = load_from_cache(structure)
        if cached is None or not cached.source:
            raise FileNotFoundError(
                f"no source given and no cached accepted program for {structure} "
                f"(looked at {_cache_path(structure)})")
        source = cached.source
    return _validate_program(structure, source, D, T)


# ===========================================================================
# CLI.
# ===========================================================================

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="LLM program-as-data generator for the LIM-ENSO Kalman-NLL (Track B2).")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--structure", choices=sorted(CONTRACTS),
                   help="generate/validate a single structure")
    g.add_argument("--all", action="store_true",
                   help="generate/validate every structure")
    ap.add_argument("--offline", action="store_true",
                    help="validate a cached or --source program WITHOUT calling MindRouter")
    ap.add_argument("--source", type=str, default=None,
                    help="(with --offline) path to a hand-written Scheme program to validate")
    ap.add_argument("--D", type=int, default=VALIDATE_D, help="validation state dimension")
    ap.add_argument("--T", type=int, default=VALIDATE_T, help="validation step count")
    ap.add_argument("--max-repairs", type=int, default=MAX_REPAIRS)
    args = ap.parse_args(argv)

    structures = sorted(CONTRACTS) if args.all else [args.structure]
    rc = 0

    for s in structures:
        if args.offline:
            hand_src = None
            if args.source:
                hand_src = Path(args.source).read_text()
            try:
                res = validate_offline(s, hand_src, D=args.D, T=args.T)
            except (FileNotFoundError, ValueError) as ex:
                print(f"[{s}] OFFLINE: SKIP -- {ex}")
                continue
            tag = "PASS" if res.ok else "FAIL"
            print(f"[{s}] OFFLINE VALIDATE {res.gate}: {tag} -- {res.detail}")
            if not res.ok and CONTRACTS[s].build_F is not None:
                rc = 1
        else:
            prog = generate_structure(s, D=args.D, T=args.T, max_repairs=args.max_repairs)
            print(f"[{s}] {prog.status} after {prog.n_attempts} MindRouter hit(s) -> "
                  f"{_cache_path(s)}")
            if prog.status == "LLM-failed":
                rc = 1

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
