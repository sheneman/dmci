############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# models.py: Scheme templater for the LIM Kalman-filter NLL run THROUGH the DMCI interpreter. ``kalman_lim_src(D, T,...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Scheme templater for the LIM Kalman-filter NLL run THROUGH the DMCI interpreter.

``kalman_lim_src(D, T, structure, jitter)`` returns a Scheme SOURCE STRING: the full
linear-Gaussian Kalman-filter negative log-likelihood for a D-dimensional LIM over T
observations, with the transition operator ``F``, process covariance ``Q``, observation
covariance ``R``, and the observations ``obs`` bound as INPUT SYMBOLS at evaluation time
(NOT built inside the program). This is the LIM generalization of the verified det/inv
pilot's local-level (F=I) program (experiments/pilots/kalman_detinv_pilot.py): the SAME
source per (structure, D, T) is reused, only the bound tensors change -- program-as-data.

The per-step filter (mirrored EXACTLY by reference.py's numpy twin):
    xpred = F x                                     (state predict)
    Ppred = F P F^T + Q                             (covariance predict)
    e     = y_k - xpred                             (innovation; y_k = (ref obs k))
    S     = Ppred + R                               (innovation covariance; H = I)
    Sinv  = inv(S)
    Kg    = Ppred Sinv                              (Kalman gain; H = I)
    xnew  = xpred + Kg e
    Pnew  = (I - Kg) Ppred                          (naive update; no Joseph/sqrt form)
    nll  += logdet(S) + e^T Sinv e                 (logdet = slogdet; det S underflows at D>=15)
The accumulator ``L`` sums the Gaussian NLL pieces; the loop is tail-recursive (recur),
so it trampolines to an O(T) walk through the compiled interpreter.

``jitter=True`` wraps S as ``Sj = S + jitter_eps*I`` and feeds ``Sj`` to inv & det (the
PD-escalation lever from the gate; NEVER a redesign). For ``structure != 'S0'`` a leading
``let*`` (``_f_assembly_prelude``) assembles F from bound factor inputs (combine-algebra only
-- the interpreter has no D x D literals); S0 binds F directly. ALL of S0/S1/S3/S4/S5 are wired:
the variants bind their factor inputs (``run_kalman_nll(..., f_factors=...)``) and build F in
the program; the SAME Kalman-NLL body is reused for every structure (program-as-data).

All ops used are in the verified DMCI surface (vec/mat/ref/dot/matvec/matmul/transpose/inv/
det/scale/eye/zeros + binary +,-,*,/ broadcasting over tensor payloads + log). DMCI is
float32-native; bind float32 tensors via ``as_matrix``.
"""

from __future__ import annotations

from .config import DEFAULT

# Structures whose in-program F factor-assembly is authored. S0 binds F directly (no prelude);
# the variants bind their factor inputs and the prelude assembles F via combine-algebra.
_SUPPORTED_STRUCTURES = {"S0", "S1", "S3", "S4", "S5"}

# Per-structure bound FACTOR INPUT symbols the prelude references (mirrors
# params.make_F_factors keys). S0 binds F directly, so it has no factor inputs.
F_FACTOR_INPUTS: dict[str, tuple[str, ...]] = {
    "S0": (),
    "S1": ("Dvec",),
    "S3": ("Dvec", "U", "V"),
    "S4": ("arow", "e0", "Sub"),
    "S5": ("M",),
}

# Trailing feature rank of each factor (1 = vector payload -> as_vector; 2 = matrix -> as_matrix).
# Leading dims on a factor tensor are batch (the DiffEvo [N,...] population). This is intrinsic
# to the factor (not sniffed from D), so a [D,D] M and a [D] Dvec are never confused.
_FACTOR_FEATURE_NDIM: dict[str, int] = {
    "Dvec": 1, "arow": 1, "e0": 1,       # vectors
    "U": 2, "V": 2, "Sub": 2, "M": 2,    # matrices
}


def _f_assembly_prelude(structure: str, D: int) -> str:
    """Leading ``let*`` bindings that assemble ``F`` from bound factor inputs (the combine-
    algebra the LLM emits; the interpreter has NO D x D literals, so F is built from factors).

    The SAME Kalman-NLL body is reused for every structure; only this prelude changes. Each
    binding reduces the factor input symbols of ``F_FACTOR_INPUTS[structure]`` (bound by
    ``run_kalman_nll`` via ``as_matrix``/``as_vector``, decoded from the raw params by
    ``params.make_F_factors``) with the SAME algebra ``params.make_F`` uses, so the in-program
    F matches the torch twin:

      S0 : F bound directly (no prelude).
      S1 : F = (* (eye D) Dvec)                          -- diag(Dvec).
      S3 : F = (+ (* (eye D) Dvec) (matmul U (transpose V)))   -- lowrank+diag.
      S4 : F = (+ (outer e0 arow) Sub)                   -- AR(2) companion of the leading mode.
      S5 : Ssym = (scale 0.5 (+ M (transpose M)));  Aanti = (scale 0.5 (- M (transpose M)));
           F = (+ Ssym Aanti)                            -- LIM normal/non-normal split (==M).

    Returns the inner ``let*`` binding lines (no surrounding parens); ``kalman_lim_src`` splices
    them ahead of the per-step bindings. Empty string for S0.
    """
    if structure == "S0":
        return ""
    if structure == "S1":
        return f"(F (* (eye {D}) Dvec))"
    if structure == "S3":
        return f"(F (+ (* (eye {D}) Dvec) (matmul U (transpose V))))"
    if structure == "S4":
        return "(F (+ (outer e0 arow) Sub))"
    if structure == "S5":
        return ("(Ssym (scale 0.5 (+ M (transpose M))))\n"
                "             (Aanti (scale 0.5 (- M (transpose M))))\n"
                "             (F (+ Ssym Aanti))")
    raise ValueError(f"unknown structure {structure!r}")


def kalman_lim_src(D: int, T: int, structure: str = "S0", jitter: bool = False) -> str:
    """Return the LIM Kalman-NLL Scheme source for dimension ``D`` over ``T`` steps.

    Args:
        D: state dimension (number of PCs). Controls the ``(eye D)`` / ``(zeros D)`` sizes.
        T: number of filter steps (observations consumed). The loop terminates at ``k == T``.
        structure: F-assembly variant (Phase 1: only 'S0' dense is fully wired).
        jitter: if True, regularize ``S`` as ``S + jitter_eps*I`` before inv/det (PD lever).

    Inputs bound at eval (NOT built in-program): ``F`` [D,D], ``Q`` [D,D], ``R`` [D,D] via
    ``as_matrix``; ``obs`` [T,D] via ``as_matrix`` (``(ref obs k)`` gathers row k as a vector).
    """
    eps = DEFAULT.jitter_eps
    # Innovation-covariance binding: plain S, or jittered Sj fed to inv & det.
    # logdet (slogdet-based) NOT (log (det S)): det S = prod of D sub-unit eigenvalues
    # underflows (~1e-20 at D=20) and (a) loses float32 precision in the product and
    # (b) hits the 1e-8 clamp in the `log` primitive, which together corrupt the log-det
    # term for D>=15. logdet sums log|LU pivots| and stays exact (verified rel ~1e-8 at D=20,
    # r_floor=0.1). See neural_compiler/ops/primitives.py:_op_logdet.
    if jitter:
        s_bind = (f"(S     (+ Ppred R))\n"
                  f"             (Sj    (+ S (scale {eps!r} (eye {D}))))\n"
                  f"             (Sinv  (inv Sj))")
        det_term = "(logdet Sj)"
    else:
        s_bind = (f"(S     (+ Ppred R))\n"
                  f"             (Sinv  (inv S))")
        det_term = "(logdet S)"

    prelude = _f_assembly_prelude(structure, D)   # "" for S0 (F bound directly)

    body = f"""(let* ({prelude.strip() + chr(10) + '             ' if prelude.strip() else ''}(xpred (matvec F x))
             (Ppred (+ (matmul (matmul F P) (transpose F)) Q))
             (y     (ref obs k))
             (e     (- y xpred))
             {s_bind}
             (Kg    (matmul Ppred Sinv))
             (xnew  (+ xpred (matvec Kg e)))
             (Pnew  (matmul (- (eye {D}) Kg) Ppred))
             (nll   (+ {det_term} (dot e (matvec Sinv e)))))
        (recur (+ k 1) xnew Pnew (+ L nll)))"""

    src = f"""
(loop ((k 0)
       (x  (zeros {D}))
       (P  (eye {D}))
       (L  0.0))
  (if (= k {T})
      L
      {body}))
"""
    return src


# ===========================================================================
# Compile cache + eval helper (mirrors the pilot's compile_dmci/evaluate_g path).
# ===========================================================================

_GRAPH_CACHE: dict[tuple[str, int, int, bool], object] = {}


def _get_graph(D: int, T: int, structure: str, jitter: bool):
    """Compile (and cache) the DMCI graph for ``(structure, D, T, jitter)``.

    Caching matters: ``compile_dmci`` bakes ONE program into the graph, and a D_max=20,
    T_train-step filter is non-trivial to compile; the same graph is reused across Adam
    iterations and seeds (only the bound F/Q/R/obs tensors change)."""
    key = (structure, D, T, jitter)
    g = _GRAPH_CACHE.get(key)
    if g is None:
        from neural_compiler.dmci import compile_dmci
        src = kalman_lim_src(D, T, structure=structure, jitter=jitter)
        g = compile_dmci(src)
        _GRAPH_CACHE[key] = g
    return g


def run_kalman_nll(F, Q, R, obs_pcs, D, T, structure: str = "S0",
                   jitter: bool = False, grad: bool = False, f_factors=None):
    """Fold the LIM Kalman NLL through the DMCI interpreter; return the scalar NLL.

    Args:
        F, Q, R: assembled ``[D, D]`` float32 torch matrices (from params.make_F/make_Q/
            make_R). If ``grad`` is True they should carry grad (their raw params are leaves)
            so backward populates the raw-parameter gradients.
        obs_pcs: the observation window as a ``[T, D]`` float32 torch tensor (PC series slice).
        D, T: state dimension and number of steps (must match obs_pcs.shape).
        structure, jitter: select the cached graph variant (must match how F was built).
        grad: if False, evaluate under ``torch.no_grad`` (faster forward-only gate/parity);
            if True, keep the autograd graph so the caller can ``.backward()``.
        f_factors: for ``structure != 'S0'`` ONLY, the bound FACTOR INPUTS that the program's
            assembly prelude reduces into F (``params.make_F_factors(F_raw, D, structure)`` --
            grad-carrying so backward reaches the raw params). When given, F is built INSIDE the
            program (the LLM's combine-algebra; ``F`` is NOT bound). When ``None`` for a variant
            we fall back to a no-prelude graph that binds the assembled ``F`` directly (used by
            gradient-free oracle paths, e.g. gate parity), so this stays correct either way.

    Returns the unwrapped scalar NLL as a 0-d torch tensor (use ``.item()`` or ``.backward()``).
    Binds the per-structure inputs via ``as_matrix``/``as_vector`` and calls ``evaluate`` with
    the shared ``EVAL_KW``.
    """
    import torch
    from neural_compiler.dmci import as_matrix
    from neural_compiler.evaluator import evaluate
    from neural_compiler.runtime.tagged_value import unwrap_number

    if obs_pcs.shape[0] < T or obs_pcs.shape[1] < D:
        raise ValueError(f"run_kalman_nll: obs_pcs {tuple(obs_pcs.shape)} too small "
                         f"for (T={T}, D={D})")

    obs = obs_pcs[:T, :D].to(torch.float32).contiguous()
    bindings = {
        "Q": as_matrix(Q.to(torch.float32)),
        "R": as_matrix(R.to(torch.float32)),
        "obs": as_matrix(obs),
    }

    use_factors = (structure != "S0") and (f_factors is not None)
    if use_factors:
        # F is assembled IN-PROGRAM from the bound factor inputs (combine-algebra prelude).
        from neural_compiler.runtime.tagged_value import TensorInput
        g = _get_graph(D, T, structure, jitter)
        for name in F_FACTOR_INPUTS[structure]:
            t = f_factors[name].to(torch.float32)
            bindings[name] = TensorInput(t, _FACTOR_FEATURE_NDIM[name])
    else:
        # No-prelude graph: bind the assembled F directly (S0 always; variants when no factors).
        g = _get_graph(D, T, structure if structure == "S0" else "S0", jitter)
        bindings["F"] = as_matrix(F.to(torch.float32))

    def _eval():
        out = evaluate(g, bindings, **DEFAULT.EVAL_KW)
        return unwrap_number(out)

    if grad:
        return _eval()
    with torch.no_grad():
        return _eval()
