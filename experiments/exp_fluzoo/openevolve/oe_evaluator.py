############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# oe_evaluator.py: OpenEvolve evaluator for the FluZoo program zoo. Bridges OpenEvolve's outer loop (AlphaEvolve-style LLM...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""OpenEvolve evaluator for the FluZoo program zoo.

Bridges OpenEvolve's outer loop (AlphaEvolve-style LLM evolution + MAP-Elites quality-diversity)
to the DMCI inner loop. OpenEvolve evolves the Scheme model string in initial_program.py; this
evaluator reads it, runs the FluZoo validity funnel (cascade stage 1) and the DMCI calibration +
held-out forecast scoring (cascade stage 2), and returns the held-out skill as combined_score plus
structural descriptors (compartments, parameters) that OpenEvolve uses as quality-diversity feature
dimensions.

  evaluate(program_path)        -> EvaluationResult   (full: funnel then DMCI score)
  evaluate_stage1(program_path) -> EvaluationResult   (cheap: validity funnel only)
  evaluate_stage2(program_path) -> EvaluationResult   (expensive: DMCI calibrate + forecast)

OpenEvolve MAXIMISES combined_score, so we report combined_score = -val_rmse (held-out forecast
RMSE; lower is better). Invalid/timed-out programs get a large negative score plus an `artifacts`
message that OpenEvolve feeds back into the next generation's prompt (error-driven repair).
"""

from __future__ import annotations

import os
import sys

sys.setrecursionlimit(20000)
import importlib.util

import numpy as np
import torch

torch.set_num_threads(1)

# Put the repo root on the path so `experiments.exp_fluzoo...` imports resolve in OpenEvolve workers.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from openevolve.evaluation_result import EvaluationResult

from experiments.exp_fluzoo.config import DEFAULT, GATE
from experiments.exp_fluzoo.programs import parse_program
from experiments.exp_fluzoo.validity import screen
from experiments.exp_fluzoo.evolve import score_source
from experiments.exp_fluzoo.forecast import season_matrix
from experiments.exp_fluzoo.data.build_data import load_processed
from neural_compiler.parser.scheme_parser import tokenize, _parse_sexpr
import dataclasses

# Fast scoring config: single-start, season-batched fit (the 300-iter DEFAULT is far too slow for
# an in-the-loop fitness). adam-iters/refit/stride are env-tunable so slower nodes can trim them.
#
# LONG-HORIZON target (8/12/16 wk): the landscape diagnostic showed 1-4-wk %ILI is structure-
# INSENSITIVE (any seasonal model scores ~0.0146), but at long horizons the autonomous dynamics
# unfold and structure differentiates (~11-14% spread; SEIRS/waning beats SIR/SEIR). So the fitness
# targets long horizons, where the program search actually has a hill to climb.
_HORIZONS = tuple(int(h) for h in os.environ.get("OE_HORIZONS", "8,12,16").split(","))
CFG = dataclasses.replace(DEFAULT, adam_iters=int(os.environ.get("OE_ADAM_ITERS", 60)),
                          seeds=(0,), horizons=_HORIZONS)
BAD = -1.0e6
REFIT_ITERS = int(os.environ.get("OE_REFIT_ITERS", 4))
ORIGIN_STRIDE = int(os.environ.get("OE_ORIGIN_STRIDE", 10))

_DATA = None
_PROBE = None


def _data():
    global _DATA, _PROBE
    if _DATA is None:
        _DATA = load_processed()
        _PROBE = torch.tensor(season_matrix(_DATA, CFG.val_seasons[0])[:12], dtype=torch.float32)
    return _DATA


def _model_source(program_path: str) -> str:
    """Import the evolved program file and return its FLU_MODEL Scheme string."""
    spec = importlib.util.spec_from_file_location("flu_oe_prog", program_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.FLU_MODEL


def _features(src: str) -> dict:
    """Structural descriptors for MAP-Elites: compartment count, parameter count, harmonics."""
    prog = parse_program(src)
    datum, _ = _parse_sexpr(tokenize(prog.body), 0)
    bindings = datum[1] if isinstance(datum, list) and len(datum) > 1 else []
    state = [b[0] for b in bindings if isinstance(b, list) and b
             and b[0] not in ("k", "yhat", "L")]
    n_harm = prog.body.count("(cos") + prog.body.count("(sin")
    return {"n_compartments": float(len(state)),
            "n_params": float(len(prog.specs)),
            "n_harmonics": float(n_harm)}


def evaluate_stage1(program_path: str) -> EvaluationResult:
    """Cheap cascade gate: does the program pass the DMCI validity funnel?"""
    _data()
    try:
        src = _model_source(program_path)
    except Exception as exc:  # noqa: BLE001  (the Python wrapper itself failed to import)
        return EvaluationResult(metrics={"runs_successfully": 0.0, "combined_score": BAD},
                                artifacts={"stderr": f"program file did not import: {type(exc).__name__}: {exc}"})
    res = screen(src, cfg=CFG, gate=GATE, probe_obs=_PROBE, name="oe")
    if not res.ok:
        return EvaluationResult(
            metrics={"runs_successfully": 0.0, "combined_score": BAD},
            artifacts={"stderr": f"validity funnel failed at [{res.stage}]: {res.detail}\n"
                                 f"FIX: {res.repair_hint}"})
    feats = _features(src)
    return EvaluationResult(metrics={"runs_successfully": 1.0, "combined_score": 0.0, **feats})


def evaluate_stage2(program_path: str) -> EvaluationResult:
    """Expensive cascade stage: DMCI calibration + held-out forecast skill."""
    _data()
    try:
        src = _model_source(program_path)
        feats = _features(src)
        sc = score_source(src, "oe", _DATA, CFG, seeds=(0,),
                          refit_iters=REFIT_ITERS, origin_stride=ORIGIN_STRIDE)
        vr = float(sc["val_rmse"])
        if not np.isfinite(vr):
            return EvaluationResult(metrics={"combined_score": BAD, "val_rmse": 1.0e9, **feats},
                                    artifacts={"stderr": "non-finite held-out RMSE (unstable rollout "
                                                         "or scoring timeout)"})
        return EvaluationResult(metrics={
            "combined_score": -vr,          # OpenEvolve maximises
            "val_rmse": vr,
            "train_nll": float(sc["train_nll"]),
            **feats})
    except Exception as exc:  # noqa: BLE001
        return EvaluationResult(metrics={"combined_score": BAD},
                                artifacts={"stderr": f"{type(exc).__name__}: {exc}"})


def evaluate(program_path: str) -> EvaluationResult:
    r1 = evaluate_stage1(program_path)
    if r1.metrics.get("runs_successfully", 0.0) < 1.0:
        return r1
    return evaluate_stage2(program_path)
