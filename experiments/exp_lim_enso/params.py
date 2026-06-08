############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# params.py: Torch parametrizations of the LIM operator F and noise covariances Q, R. These are built OUTSIDE the DMCI...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Torch parametrizations of the LIM operator F and noise covariances Q, R.

These are built OUTSIDE the DMCI interpreter (the interpreter has no ``set!`` and
cannot cheaply assemble a D x D matrix from raw params), then bound into the program
via ``as_matrix`` so gradients flow from the Kalman NLL back to the raw parameters.
This keeps the optimizer in the positive-definite cone for Q and R BY CONSTRUCTION:

  Q = tril(Lq) @ tril(Lq).T + q_floor*I        (Cholesky factor; PD for any raw Lq)
  R = softplus(r_raw) * I + r_floor*I          (scalar isotropic; PD for any raw r)
  F = structure-specific assembly of raw params (S0 dense, S1 diag, S3 lowrank+diag, ...)

Everything is pure torch, autograd-friendly, and FLOAT32 (DMCI is float32-native; the
bound tensors must match the interpreter's dtype). The numpy reference twin in
reference.py mirrors these in float32 AND float64.

Each structure declares a raw-parameter COUNT ``k`` (param_count) so AIC/BIC in the
analysis can penalize model complexity consistently:  k(F) + k(Q) + k(R).
  k(Q, Cholesky tril of [D,D])         = D*(D+1)/2
  k(R, scalar)                          = 1
  k(F) per structure                    = see PARAM_COUNTS / param_count(D, structure)
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .config import DEFAULT


_DTYPE = torch.float32   # DMCI is float32-native; bound tensors must match.


# ===========================================================================
# Noise covariances Q (Cholesky) and R (scalar) -- both PD by construction.
# ===========================================================================

def make_Q(Lq_raw: Tensor, D: int, q_floor: float = DEFAULT.q_floor) -> Tensor:
    """Process-noise covariance ``Q = L L^T + q_floor*I``  (PD for any raw vector).

    ``Lq_raw`` is the flat lower-triangular Cholesky factor as a length ``D*(D+1)/2``
    vector (row-major over the lower triangle incl. diagonal). We scatter it into a
    lower-triangular ``L``, form ``L L^T``, and add ``q_floor*I`` so Q is strictly PD
    even when L is rank-deficient. Returns a ``[D, D]`` float32 tensor with grad flowing
    to ``Lq_raw``. The ``q_floor`` term is the secondary float32 PD lever (R's floor is
    the primary one)."""
    k = D * (D + 1) // 2
    if Lq_raw.shape[-1] != k:
        raise ValueError(f"make_Q: Lq_raw must have length D*(D+1)/2 = {k} for D={D}, "
                         f"got {Lq_raw.shape[-1]}")
    L = torch.zeros(D, D, dtype=Lq_raw.dtype, device=Lq_raw.device)
    idx = torch.tril_indices(D, D, device=Lq_raw.device)
    L[idx[0], idx[1]] = Lq_raw
    eye = torch.eye(D, dtype=Lq_raw.dtype, device=Lq_raw.device)
    return L @ L.transpose(-1, -2) + q_floor * eye


def make_R(r_raw: Tensor, D: int, r_floor: float = DEFAULT.r_floor) -> Tensor:
    """Observation-noise covariance ``R = softplus(r_raw)*I + r_floor*I``  (scalar, PD).

    Isotropic by design: a single positive scalar ``softplus(r_raw)`` scales the identity,
    plus ``r_floor*I``. ``r_floor`` (default 0.1) is the PRIMARY float32 PD lever -- it keeps
    the innovation covariance ``S = H Ppred H^T + R`` comfortably above the float32 denormal
    floor so ``log(det S)`` stays finite over a long filter. ``r_raw`` is a 0-d tensor (or a
    1-element tensor); returns a ``[D, D]`` float32 tensor with grad flowing to ``r_raw``."""
    r = torch.nn.functional.softplus(r_raw.reshape(())) + r_floor
    return r * torch.eye(D, dtype=r_raw.dtype, device=r_raw.device)


# ===========================================================================
# Transition operator F -- structure dispatch.
# ===========================================================================
# For each structure we document the raw-parameter SHAPE and COUNT k. The LLM emits only
# the COMBINE-ALGEBRA that assembles F from these bound factors inside the Scheme program
# (models._f_assembly_prelude); here we provide the torch twin (make_F) that reduces the SAME
# factors (make_F_factors) with the IDENTICAL algebra, so the bound matrix matches what the
# program builds. All of S0/S1/S3/S4/S5 are wired end-to-end (the variants bind their factor
# inputs and assemble F in-program via the prelude; S0 binds F directly with no prelude).


def param_count_F(D: int, structure: str = "S0", rank: int = DEFAULT.lowrank_rank) -> int:
    """Number of free raw parameters in F for ``structure`` at dimension ``D`` (for AIC/BIC)."""
    if structure == "S0":            # dense [D,D]
        return D * D
    if structure == "S1":            # diagonal
        return D
    if structure == "S3":            # lowrank+diag: diag(d) + U V^T,  U,V in [D,r]
        return D + 2 * D * rank
    if structure == "S4":            # AR(2) companion of the leading mode: 2 AR coeffs
        return 2                     # (a1, a2); see make_F for the [D,D] companion realization
    if structure == "S5":            # sym/antisym re-parametrization of a full F
        return D * D
    raise ValueError(f"unknown structure {structure!r}")


def param_count(D: int, structure: str = "S0", rank: int = DEFAULT.lowrank_rank) -> dict[str, int]:
    """Full free-parameter budget k = k(F)+k(Q)+k(R) for AIC/BIC. Returns the breakdown."""
    kF = param_count_F(D, structure, rank)
    kQ = D * (D + 1) // 2            # Cholesky lower triangle incl. diagonal
    kR = 1                           # scalar isotropic R
    return {"F": kF, "Q": kQ, "R": kR, "total": kF + kQ + kR}


def make_F_factors(raw: Tensor, D: int, structure: str = "S0",
                   rank: int = DEFAULT.lowrank_rank) -> dict[str, Tensor]:
    """Decode flat raw params into the per-structure BOUND FACTOR INPUTS for the program.

    These tensors are exactly the input symbols the Scheme prelude
    (``models._f_assembly_prelude``) references; binding them via ``as_matrix``/``as_vector``
    lets the program ASSEMBLE F from combine-algebra (the LLM emits the algebra, never a
    D x D literal). ``make_F`` below reduces these same factors with the IDENTICAL algebra
    so the torch twin and the in-program F agree bit-for-bit (up to float32 op order).

    Returns a dict keyed by the prelude's input symbol names. Leading ``[...]`` batch dims on
    ``raw`` (e.g. an ``[N, k]`` population) are preserved so the DiffEvo batched path can bind
    ``[N, ...]`` factors in one walk. Layout (trailing dim) matches ``param_count_F``:
      S0 dense        : Fmat [..,D,D]                       (F bound directly; no prelude)
      S1 diagonal     : Dvec [..,D]                         -> F = (* (eye D) Dvec)
      S3 lowrank+diag : Dvec [..,D], U [..,D,r], V [..,D,r] -> F = (+ (* (eye D) Dvec)
                                                                       (matmul U (transpose V)))
      S4 AR2-companion: arow [..,D] (a1,a2,0..), e0 [D], Sub [D,D]
                                                           -> F = (+ (outer e0 arow) Sub),
                                                              the [D,D] companion of the
                                                              leading AR(2) mode (only a1,a2
                                                              free; e0/Sub fixed structure)
      S5 sym/antisym  : M [..,D,D]                          -> F = 0.5(M+M^T)+0.5(M-M^T)
    Float32, grad flowing to ``raw``."""
    lead = raw.shape[:-1]            # batch dims (empty for a single param set)
    if structure == "S0":
        return {"F": raw[..., : D * D].reshape(*lead, D, D)}
    if structure == "S1":
        return {"Dvec": raw[..., :D]}
    if structure == "S3":
        d = raw[..., :D]
        U = raw[..., D: D + D * rank].reshape(*lead, D, rank)
        V = raw[..., D + D * rank: D + 2 * D * rank].reshape(*lead, D, rank)
        return {"Dvec": d, "U": U, "V": V}
    if structure == "S4":
        # Companion-as-outer-product so the prelude is pure combine-algebra (no scatter op):
        #   F = (+ (outer e0 arow) Sub),  arow = [a1, a2, 0..0] (learned top row, first 2 free),
        #   e0 = first basis vector, Sub = fixed first-subdiagonal delay line.
        # Only avec=(a1,a2) carry grad; e0/Sub/the zero pad are CONSTANT structural factors the
        # LLM's algebra multiplies, so param_count_F stays 2.
        avec = raw[..., :2]                                 # (a1, a2)
        zpad = torch.zeros(*lead, D, dtype=raw.dtype, device=raw.device)
        arow = torch.cat([avec, zpad], dim=-1)[..., :D]     # [a1, a2, 0..0] (top row, D>=2)
        e0 = torch.zeros(D, dtype=raw.dtype, device=raw.device); e0[0] = 1.0
        Sub = torch.zeros(D, D, dtype=raw.dtype, device=raw.device)
        if D >= 2:
            sub = torch.arange(1, D)
            Sub[sub, sub - 1] = 1.0
        return {"arow": arow, "e0": e0, "Sub": Sub}
    if structure == "S5":
        return {"M": raw[..., : D * D].reshape(*lead, D, D)}
    raise ValueError(f"unknown structure {structure!r}")


def make_F(raw: Tensor, D: int, structure: str = "S0",
           rank: int = DEFAULT.lowrank_rank) -> Tensor:
    """Assemble the ``[..,D, D]`` transition matrix F from flat raw params per ``structure``.

    This is the TORCH TWIN of the in-program factor assembly: it decodes ``raw`` into the
    same bound factors as ``make_F_factors`` and reduces them with the IDENTICAL combine-
    algebra that ``models._f_assembly_prelude`` emits, so the bound matrix matches what the
    program builds. The raw-param layout matches ``param_count_F``:
      S0 dense        : raw[D*D]      -> raw.reshape(D, D)
      S1 diagonal     : raw[D]        -> diag(Dvec)              == (* (eye D) Dvec)
      S3 lowrank+diag : raw[D+2*D*r]  -> diag(Dvec) + U V^T      (d=first D, then U, V)
      S4 AR2-companion: raw[2]        -> [D,D] companion (outer e0 arow + Sub) of the leading
                                         AR(2) mode (a1,a2 = raw; e0/Sub fixed structure)
      S5 sym/antisym  : raw[D*D]      -> 0.5(M+M^T)+0.5(M-M^T)   (LIM normal/non-normal split)
    Returns float32 with grad flowing to ``raw``; leading batch dims are preserved."""
    fac = make_F_factors(raw, D, structure, rank)
    if structure == "S0":
        return fac["F"]
    eye = torch.eye(D, dtype=raw.dtype, device=raw.device)
    if structure == "S1":
        return eye * fac["Dvec"].unsqueeze(-2)              # (* (eye D) Dvec): diag(Dvec)
    if structure == "S3":
        diag = eye * fac["Dvec"].unsqueeze(-2)
        return diag + fac["U"] @ fac["V"].transpose(-1, -2)
    if structure == "S4":
        # F = (+ (outer e0 arow) Sub): arow lands in row 0 (a1,a2,0..); Sub is the delay line.
        outer = fac["e0"].unsqueeze(-1) * fac["arow"].unsqueeze(-2)
        return outer + fac["Sub"]
    if structure == "S5":
        M = fac["M"]
        Mt = M.transpose(-1, -2)
        return 0.5 * (M + Mt) + 0.5 * (M - Mt)              # == M; the constrained split that
        #                                                     the prelude builds as Ssym+Aanti
    raise ValueError(f"unknown structure {structure!r}")


# ===========================================================================
# Near-stable initialization.
# ===========================================================================

def init_raw_params(D: int, structure: str = "S0", seed: int = 0,
                    rank: int = DEFAULT.lowrank_rank,
                    q_floor: float = DEFAULT.q_floor,
                    r_floor: float = DEFAULT.r_floor) -> dict[str, Tensor]:
    """Near-stable init of the raw (F, Q, R) parameters as leaf tensors (requires_grad).

    Returns ``{"F_raw", "Lq_raw", "r_raw"}`` such that, decoded:
      F      ~ 0.9 * I           (contractive, spectral radius ~0.9 -> stable LIM)
      Q      ~ small isotropic   (Lq diagonal ~ sqrt(0.01) so L L^T ~ 0.01 I, + q_floor)
      R      ~ r_floor * I       (r_raw chosen so softplus(r_raw) ~ 0 -> R ~ r_floor*I)
    Small zero-mean noise breaks symmetry across seeds. All float32, all leaves with grad.
    The decoded matrices are PD/contractive at init so the very first gate eval is benign."""
    g = torch.Generator().manual_seed(int(seed))

    # --- F: contractive near 0.9*I (per-structure raw layout) ---
    nF = param_count_F(D, structure, rank)
    F_noise = 0.02 * torch.randn(nF, generator=g, dtype=_DTYPE)
    if structure == "S0":
        F_raw = (0.9 * torch.eye(D, dtype=_DTYPE)).reshape(-1) + F_noise
    elif structure == "S1":
        F_raw = 0.9 * torch.ones(D, dtype=_DTYPE) + F_noise
    elif structure == "S3":
        F_raw = torch.zeros(nF, dtype=_DTYPE)
        F_raw[:D] = 0.9                                  # diagonal base ~0.9
        F_raw = F_raw + 0.01 * torch.randn(nF, generator=g, dtype=_DTYPE)  # tiny U,V,d noise
    elif structure == "S4":
        # AR(2) of the leading mode near a stable persistent root: roots z^2-a1 z-a2,
        # a1~0.9, a2~0 -> a leading eigenvalue ~0.9 (contractive), rest of the companion is
        # the delay line (roots on/inside the unit circle).
        F_raw = torch.tensor([0.9, 0.0], dtype=_DTYPE) + 0.02 * torch.randn(2, generator=g, dtype=_DTYPE)
    elif structure == "S5":
        # full raw M near 0.9*I; the sym/antisym split re-expresses the SAME M (==M), so a
        # near-0.9*I M gives a contractive, near-normal F at init.
        F_raw = (0.9 * torch.eye(D, dtype=_DTYPE)).reshape(-1) + F_noise
    else:
        F_raw = 0.9 * torch.ones(nF, dtype=_DTYPE) + F_noise

    # --- Q Cholesky: L ~ sqrt(0.01) on the diagonal so L L^T ~ 0.01 I (small process noise) ---
    kQ = D * (D + 1) // 2
    Lq_raw = 0.005 * torch.randn(kQ, generator=g, dtype=_DTYPE)
    # set the diagonal entries (positions where row index == col index in the tril order)
    diag_pos = [i * (i + 1) // 2 + i for i in range(D)]
    Lq_raw[diag_pos] = math.sqrt(0.01)

    # --- R: r_raw s.t. softplus(r_raw) ~ 0 -> R ~ r_floor*I (the floor dominates at init) ---
    r_raw = torch.tensor(-4.0, dtype=_DTYPE)             # softplus(-4) ~ 0.018

    return {
        "F_raw": F_raw.clone().requires_grad_(True),
        "Lq_raw": Lq_raw.clone().requires_grad_(True),
        "r_raw": r_raw.clone().requires_grad_(True),
    }


# ===========================================================================
# Stability diagnostics.
# ===========================================================================

def spectral_radius(F: Tensor) -> Tensor:
    """Spectral radius (max |eigenvalue|) of F. A LIM is stable iff this is < 1.

    Uses ``torch.linalg.eigvals`` (complex); returns a real 0-d float32 tensor. Detached-safe
    for diagnostics, but differentiable if a soft stability penalty is desired."""
    ev = torch.linalg.eigvals(F.to(torch.float64))       # float64 eig for conditioning
    return ev.abs().max().to(F.dtype)


def stability_penalty(F: Tensor, margin: float = 1.0) -> Tensor:
    """Soft penalty ``relu(spectral_radius(F) - margin)^2`` pushing F into the stable disk.

    Returns a 0-d float32 tensor; multiply by ``ExpLimConfig.stability_penalty`` and add to the
    NLL if an unstable F is observed during a run. Default off (weight 0)."""
    sr = spectral_radius(F)
    over = torch.clamp(sr - margin, min=0.0)
    return over * over
