############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# programs.py: Program representation, parsing, the DMCI fold, and forecast read-back for FluZoo. A FluZoo program artifact is...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Program representation, parsing, the DMCI fold, and forecast read-back for FluZoo.

A FluZoo program artifact is two top-level S-expressions:

    (params (beta0 positive 1.5) (amp signed-unit 0.2) ...)   ; machine-readable schema
    (loop ((k 0) ... (yhat (zeros 11)) (L 0.0))               ; the model
      (if (= k NWEEKS) L (let* (...) (recur (+ k 1) ... ypred (+ L nll)))))

The model is a tail-recursive weekly rollout that reads the observation matrix one row
at a time with (ref obs k) (obs is bound [T, R]), accumulates a Gaussian NLL of observed
weighted %ILI into the loop variable `L`, and carries the current predicted observable in
the loop variable `yhat`. Seasonal transmission is built INSIDE the loop from the integer
week counter k (no external per-step forcing) so the heap stays O(T).

Two execution modes share ONE generated program (the dynamics are written once):
  * FIT  -- the program as written returns `L` (the scalar NLL); see run_nll.
  * PREDICT -- a one-token AST swap makes the loop's base case return `yhat` instead of
    `L`; rolling the autonomous model to horizon W and reading `yhat` gives the predicted
    observable for week W. The dynamics never depend on obs, so future weeks are bound to a
    zero-padded matrix; see run_predict. This is what the filter-then-forecast scorer uses.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from neural_compiler.dmci import (
    compile_dmci,
    split_top_level_forms,
    unsupported_interpreter_ops,
    _datum_to_scheme,
    _detect_free_vars,
)
from neural_compiler.parser.scheme_parser import tokenize, _parse_sexpr
from neural_compiler.evaluator import evaluate
from neural_compiler.evaluator.engine import _evaluate_tagged
from neural_compiler.runtime.heap import TensorHeap
from neural_compiler.runtime.tagged_value import (
    make_float, as_matrix, unwrap_number, extract_payload,
)

from .config import DEFAULT, HORIZON_TOKEN
from .paramspec import ParamSpec, parse_param_block, constrain

# 2*pi / 52 weeks-per-year, as a literal (pi is not a named constant in DMCI).
OMEGA = 0.12083048667087144


@dataclass
class FluProgram:
    """A parsed FluZoo program: parameter schema + the compilable model body."""
    specs: list[ParamSpec]
    body: str                       # the model S-expression, still containing NWEEKS
    raw_text: str                   # the full original artifact (schema + body)
    name: str = "anon"

    @property
    def param_names(self) -> list[str]:
        return [s.name for s in self.specs]


def parse_program(text: str, name: str = "anon") -> FluProgram:
    """Split an artifact into its (params ...) schema and its model body."""
    forms = split_top_level_forms(text)
    if len(forms) != 2:
        raise ValueError(f"expected exactly 2 top-level forms (params + model), "
                         f"got {len(forms)}")
    params_form, body = forms
    specs = parse_param_block(params_form)
    return FluProgram(specs=specs, body=body.strip(), raw_text=text, name=name)


def materialize(body: str, T: int) -> str:
    """Substitute the horizon token with the literal week count before compiling."""
    return body.replace(HORIZON_TOKEN, str(int(T)))


def predict_body(body: str) -> str:
    """Derive the PREDICT form: swap the loop's base-case return (`L`) for `yhat`.

    Requires the conventional shape (loop ((..) (yhat ..) (L ..)) (if test L recur)).
    """
    datum, _ = _parse_sexpr(tokenize(body), 0)
    if not (isinstance(datum, list) and datum and datum[0] == "loop" and len(datum) >= 3):
        raise ValueError("forecasting needs a top-level (loop (...) (if test base recur)) body")
    bindings, ifx = datum[1], datum[2]
    var_names = [b[0] for b in bindings if isinstance(b, list) and b]
    if "yhat" not in var_names:
        raise ValueError("program must carry a `yhat` loop variable to forecast")
    if not (isinstance(ifx, list) and len(ifx) == 4 and ifx[0] == "if"):
        raise ValueError("loop body must be (if test base-case recur-branch)")
    swapped = ["loop", bindings, ["if", ifx[1], "yhat", ifx[3]]]
    return _datum_to_scheme(swapped)


_GRAPH_CACHE: dict[tuple, object] = {}


def get_graph(body: str, T: int, mode: str = "fit"):
    """Compile (and cache) the DMCI graph for a model body at horizon T."""
    key = (body, int(T), mode)
    g = _GRAPH_CACHE.get(key)
    if g is None:
        src = materialize(body if mode == "fit" else predict_body(body), T)
        g = compile_dmci(src)
        _GRAPH_CACHE[key] = g
    return g


def free_vars(body: str, T: int) -> set[str]:
    return set(_detect_free_vars(materialize(body, T)))


def unsupported_ops(body: str, T: int) -> set:
    return unsupported_interpreter_ops(materialize(body, T))


def run_nll(prog: FluProgram, raw: dict[str, torch.Tensor], obs: torch.Tensor,
            cfg=DEFAULT, grad: bool = True) -> torch.Tensor:
    """Fold the program through DMCI, returning the 0-d Gaussian NLL it computes."""
    T = int(obs.shape[0])
    graph = get_graph(prog.body, T, mode="fit")
    con = constrain(prog.specs, raw)
    binding = {s.name: make_float(con[s.name]) for s in prog.specs}
    binding["obs"] = as_matrix(obs)
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        out = unwrap_number(evaluate(graph, binding, **cfg.EVAL_KW))
    return out.reshape(())


def run_nll_batched(prog: FluProgram, raw: dict[str, torch.Tensor],
                    obs_batch: torch.Tensor, cfg=DEFAULT, grad: bool = True) -> torch.Tensor:
    """Per-season NLLs for a batched obs `[N, T, R]` in ONE interpreter walk.

    The model is autonomous (the compartment trajectory depends only on the shared
    parameters, not on obs), so a leading batch dimension over the observation matrix
    yields N season likelihoods from a single shared rollout -- an ~N-fold speedup over
    N separate folds. Control flow is data-independent (the loop bound is the integer T),
    so the batched interpreter walk is exact. Returns a length-N tensor.
    """
    T = int(obs_batch.shape[1])
    graph = get_graph(prog.body, T, mode="fit")
    con = constrain(prog.specs, raw)
    binding = {s.name: make_float(con[s.name]) for s in prog.specs}
    binding["obs"] = as_matrix(obs_batch)
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        out = unwrap_number(evaluate(graph, binding, **cfg.EVAL_KW))
    return out.reshape(-1)


def run_predict(prog: FluProgram, raw: dict[str, torch.Tensor], week: int,
                n_regions: int, cfg=DEFAULT) -> torch.Tensor:
    """Roll the fitted autonomous model to `week` and read the predicted observable.

    Returns a length-`n_regions` tensor (the model's %ILI prediction for that week). obs
    is zero-padded to the horizon because the dynamics never depend on it. No grad.
    """
    T = int(week) + 1
    graph = get_graph(prog.body, T, mode="predict")
    with torch.no_grad():
        con = constrain(prog.specs, {n: raw[n].detach() for n in prog.param_names})
        binding = {s.name: make_float(con[s.name]) for s in prog.specs}
        binding["obs"] = as_matrix(torch.zeros(T, n_regions, dtype=torch.float32))
        heap = TensorHeap(max_size=cfg.eval_max_heap)
        tagged = _evaluate_tagged(graph, binding, cfg.eval_max_iter,
                                  cfg.eval_max_depth, cfg.eval_max_heap, heap=heap)
        vec = heap.read(extract_payload(tagged)[..., 0])
    return vec.reshape(-1).detach()


def run_predict_batched(prog: FluProgram, raw: dict[str, torch.Tensor], week: int,
                        n_regions: int, n_seasons: int, cfg=DEFAULT) -> torch.Tensor:
    """Predicted observable at `week` for N held-out seasons at once -> [n_seasons, R].

    The initial-condition parameters in `raw` are batched ([n_seasons] each) so every season
    rolls forward from its own re-estimated state in ONE interpreter walk; structural params
    are shared scalars and broadcast. obs is a zero-padded [n_seasons, T, R] (autonomous).
    """
    T = int(week) + 1
    graph = get_graph(prog.body, T, mode="predict")
    with torch.no_grad():
        con = constrain(prog.specs, {n: raw[n].detach() for n in prog.param_names})
        binding = {s.name: make_float(con[s.name]) for s in prog.specs}
        binding["obs"] = as_matrix(torch.zeros(n_seasons, T, n_regions, dtype=torch.float32))
        heap = TensorHeap(max_size=cfg.eval_max_heap)
        tagged = _evaluate_tagged(graph, binding, cfg.eval_max_iter,
                                  cfg.eval_max_depth, cfg.eval_max_heap, heap=heap)
        vec = heap.read(extract_payload(tagged)[..., 0])
    return vec.reshape(n_seasons, n_regions).detach()


# ---------------------------------------------------------------------------
# Hand-written reference programs (prompt exemplars + classical baselines + self-test).
# Each carries `yhat` (the current predicted observable) so the PREDICT swap works.
# Gaussian NLL uses a floored variance (+ s2 FLOOR) so float32 never divides by zero.
# All arithmetic is BINARY; the only branch tests the integer counter; seasonal beta is
# built from k via cos -- the constraints every generated program must also obey.
# ---------------------------------------------------------------------------

_FLOOR = repr(DEFAULT.obs_var_floor)

#: National seasonally-forced SEIR (single %ILI series; obs is [T,1]).
SEIR_NATIONAL = f"""
(params
  (beta0 positive 1.5)
  (amp   signed-unit 0.2)
  (phase free 0.0)
  (sigma unit 0.5)
  (gamma unit 0.5)
  (rho   unit 0.05 0.2)
  (i0    unit 0.0015 0.01)
  (e0    unit 0.0008 0.01)
  (s2    positive 4e-6))

(loop ((k 0)
       (S (- (- 1.0 i0) e0))
       (E e0)
       (I i0)
       (R 0.0)
       (yhat (zeros 1))
       (L 0.0))
  (if (= k {HORIZON_TOKEN})
      L
      (let* ((beta  (* beta0 (+ 1.0 (* amp (cos (+ (* {OMEGA!r} k) phase))))))
             (force (* beta (* S I)))
             (e2i   (* sigma E))
             (i2r   (* gamma I))
             (Snew  (- S force))
             (Enew  (+ E (- force e2i)))
             (Inew  (+ I (- e2i i2r)))
             (Rnew  (+ R i2r))
             (ypred (vec (* rho I)))
             (y     (ref obs k))
             (resid (- y ypred))
             (var   (+ s2 {_FLOOR}))
             (nll   (+ (/ (dot resid resid) (* 2.0 var)) (* 0.5 (log var)))))
        (recur (+ k 1) Snew Enew Inew Rnew ypred (+ L nll)))))
""".strip()

#: National SEIRS -- SEIR plus waning immunity R->S at rate omega_w (sustains recurrence).
SEIRS_NATIONAL = f"""
(params
  (beta0 positive 1.5)
  (amp   signed-unit 0.2)
  (phase free 0.0)
  (sigma unit 0.5)
  (gamma unit 0.5)
  (omega_w unit 0.02 0.2)
  (rho   unit 0.05 0.2)
  (i0    unit 0.0015 0.01)
  (e0    unit 0.0008 0.01)
  (s2    positive 4e-6))

(loop ((k 0)
       (S (- (- 1.0 i0) e0))
       (E e0)
       (I i0)
       (R 0.0)
       (yhat (zeros 1))
       (L 0.0))
  (if (= k {HORIZON_TOKEN})
      L
      (let* ((beta  (* beta0 (+ 1.0 (* amp (cos (+ (* {OMEGA!r} k) phase))))))
             (force (* beta (* S I)))
             (e2i   (* sigma E))
             (i2r   (* gamma I))
             (wane  (* omega_w R))
             (Snew  (+ (- S force) wane))
             (Enew  (+ E (- force e2i)))
             (Inew  (+ I (- e2i i2r)))
             (Rnew  (- (+ R i2r) wane))
             (ypred (vec (* rho I)))
             (y     (ref obs k))
             (resid (- y ypred))
             (var   (+ s2 {_FLOOR}))
             (nll   (+ (/ (dot resid resid) (* 2.0 var)) (* 0.5 (log var)))))
        (recur (+ k 1) Snew Enew Inew Rnew ypred (+ L nll)))))
""".strip()

#: National SIR (no exposed compartment) -- the simplest member of the zoo.
SIR_NATIONAL = f"""
(params
  (beta0 positive 1.5)
  (amp   signed-unit 0.2)
  (phase free 0.0)
  (gamma unit 0.5)
  (rho   unit 0.05 0.2)
  (i0    unit 0.0015 0.01)
  (s2    positive 4e-6))

(loop ((k 0)
       (S (- 1.0 i0))
       (I i0)
       (R 0.0)
       (yhat (zeros 1))
       (L 0.0))
  (if (= k {HORIZON_TOKEN})
      L
      (let* ((beta  (* beta0 (+ 1.0 (* amp (cos (+ (* {OMEGA!r} k) phase))))))
             (force (* beta (* S I)))
             (i2r   (* gamma I))
             (Snew  (- S force))
             (Inew  (+ I (- force i2r)))
             (Rnew  (+ R i2r))
             (ypred (vec (* rho I)))
             (y     (ref obs k))
             (resid (- y ypred))
             (var   (+ s2 {_FLOOR}))
             (nll   (+ (/ (dot resid resid) (* 2.0 var)) (* 0.5 (log var)))))
        (recur (+ k 1) Snew Inew Rnew ypred (+ L nll)))))
""".strip()

#: Regional seasonally-forced SEIR -- the CANONICAL zoo shape: 11-region vector state,
#: shared seasonal beta, per-region observation via (ref obs k) over [T,11]. Smoke-validated
#: (Hadamard S.*I, vector compartments, exact gradients, vector forecast read-back all fold).
SEIR_REGIONAL = f"""
(params
  (beta0 positive 1.5)
  (amp   signed-unit 0.2)
  (phase free 0.0)
  (sigma unit 0.5)
  (gamma unit 0.5)
  (rho   unit 0.05 0.2)
  (i0    unit 0.0015 0.01)
  (e0    unit 0.0008 0.01)
  (s2    positive 4e-6))

(loop ((k 0)
       (S (- (- (ones 11) (scale i0 (ones 11))) (scale e0 (ones 11))))
       (E (scale e0 (ones 11)))
       (I (scale i0 (ones 11)))
       (R (zeros 11))
       (yhat (zeros 11))
       (L 0.0))
  (if (= k {HORIZON_TOKEN})
      L
      (let* ((beta  (* beta0 (+ 1.0 (* amp (cos (+ (* {OMEGA!r} k) phase))))))
             (force (scale beta (* S I)))
             (e2i   (scale sigma E))
             (i2r   (scale gamma I))
             (Snew  (- S force))
             (Enew  (+ E (- force e2i)))
             (Inew  (+ I (- e2i i2r)))
             (Rnew  (+ R i2r))
             (ypred (scale rho I))
             (y     (ref obs k))
             (resid (- y ypred))
             (var   (+ s2 {_FLOOR}))
             (nll   (+ (/ (dot resid resid) (* 2.0 var)) (* 0.5 (log var)))))
        (recur (+ k 1) Snew Enew Inew Rnew ypred (+ L nll)))))
""".strip()


#: Regional SIR -- simplest regional member (no exposed compartment).
SIR_REGIONAL = f"""
(params
  (beta0 positive 1.5)
  (amp   signed-unit 0.2)
  (phase free 0.0)
  (gamma unit 0.5)
  (rho   unit 0.05 0.2)
  (i0    unit 0.0015 0.01)
  (s2    positive 4e-6))

(loop ((k 0)
       (S (- (ones 11) (scale i0 (ones 11))))
       (I (scale i0 (ones 11)))
       (R (zeros 11))
       (yhat (zeros 11))
       (L 0.0))
  (if (= k {HORIZON_TOKEN})
      L
      (let* ((beta  (* beta0 (+ 1.0 (* amp (cos (+ (* {OMEGA!r} k) phase))))))
             (force (scale beta (* S I)))
             (i2r   (scale gamma I))
             (Snew  (- S force))
             (Inew  (+ I (- force i2r)))
             (Rnew  (+ R i2r))
             (ypred (scale rho I))
             (y     (ref obs k))
             (resid (- y ypred))
             (var   (+ s2 {_FLOOR}))
             (nll   (+ (/ (dot resid resid) (* 2.0 var)) (* 0.5 (log var)))))
        (recur (+ k 1) Snew Inew Rnew ypred (+ L nll)))))
""".strip()

#: Regional SEIRS -- SEIR + waning immunity R->S (sustains multi-season recurrence).
SEIRS_REGIONAL = f"""
(params
  (beta0 positive 1.5)
  (amp   signed-unit 0.2)
  (phase free 0.0)
  (sigma unit 0.5)
  (gamma unit 0.5)
  (omega_w unit 0.02 0.2)
  (rho   unit 0.05 0.2)
  (i0    unit 0.0015 0.01)
  (e0    unit 0.0008 0.01)
  (s2    positive 4e-6))

(loop ((k 0)
       (S (- (- (ones 11) (scale i0 (ones 11))) (scale e0 (ones 11))))
       (E (scale e0 (ones 11)))
       (I (scale i0 (ones 11)))
       (R (zeros 11))
       (yhat (zeros 11))
       (L 0.0))
  (if (= k {HORIZON_TOKEN})
      L
      (let* ((beta  (* beta0 (+ 1.0 (* amp (cos (+ (* {OMEGA!r} k) phase))))))
             (force (scale beta (* S I)))
             (e2i   (scale sigma E))
             (i2r   (scale gamma I))
             (wane  (scale omega_w R))
             (Snew  (+ (- S force) wane))
             (Enew  (+ E (- force e2i)))
             (Inew  (+ I (- e2i i2r)))
             (Rnew  (- (+ R i2r) wane))
             (ypred (scale rho I))
             (y     (ref obs k))
             (resid (- y ypred))
             (var   (+ s2 {_FLOOR}))
             (nll   (+ (/ (dot resid resid) (* 2.0 var)) (* 0.5 (log var)))))
        (recur (+ k 1) Snew Enew Inew Rnew ypred (+ L nll)))))
""".strip()


REFERENCE_PROGRAMS: dict[str, str] = {
    "sir_national": SIR_NATIONAL,
    "seir_national": SEIR_NATIONAL,
    "seirs_national": SEIRS_NATIONAL,
    "sir_regional": SIR_REGIONAL,
    "seir_regional": SEIR_REGIONAL,
    "seirs_regional": SEIRS_REGIONAL,
}


def reference_program(name: str) -> FluProgram:
    return parse_program(REFERENCE_PROGRAMS[name], name=name)
