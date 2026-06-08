############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# mock_evaluator.py: Fast mock evaluator for testing the OpenEvolve <-> FluZoo plumbing (no DMCI scoring). Parses the evolved Scheme...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Fast mock evaluator for testing the OpenEvolve <-> FluZoo plumbing (no DMCI scoring).

Parses the evolved Scheme (stage 1) and returns an instant structure-based pseudo-fitness
(stage 2), so a few OpenEvolve iterations exercise the full loop -- loading the initial program,
LLM diff-editing the Scheme block, the cascade, feature dimensions, and output -- in seconds
instead of the ~12 min/program the real DMCI evaluator costs.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from openevolve.evaluation_result import EvaluationResult
from experiments.exp_fluzoo.openevolve.oe_evaluator import _model_source, _features


def evaluate_stage1(program_path: str) -> EvaluationResult:
    try:
        src = _model_source(program_path)
        _features(src)  # parses both forms; raises if malformed
        return EvaluationResult(metrics={"runs_successfully": 1.0, "combined_score": 0.0})
    except Exception as exc:  # noqa: BLE001
        return EvaluationResult(metrics={"runs_successfully": 0.0, "combined_score": -1.0e6},
                                artifacts={"stderr": f"parse failed: {type(exc).__name__}: {exc}"})


def evaluate_stage2(program_path: str) -> EvaluationResult:
    try:
        src = _model_source(program_path)
        f = _features(src)
        # pseudo-fitness: reward richer structure (mock only) -- instant, no DMCI
        score = -(0.02 - 0.001 * f["n_compartments"] - 0.0005 * f["n_harmonics"])
        return EvaluationResult(metrics={"combined_score": score, "val_rmse": -score, **f})
    except Exception as exc:  # noqa: BLE001
        return EvaluationResult(metrics={"combined_score": -1.0e6},
                                artifacts={"stderr": f"{type(exc).__name__}: {exc}"})


def evaluate(program_path: str) -> EvaluationResult:
    r1 = evaluate_stage1(program_path)
    if r1.metrics.get("runs_successfully", 0.0) < 1.0:
        return r1
    return evaluate_stage2(program_path)
