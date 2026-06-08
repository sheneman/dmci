############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# oe_evaluator.py: OpenEvolve evaluator for the battery capacity-fade program search. Reads the evolved BATTERY_MODEL Scheme, runs...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""OpenEvolve evaluator for the battery capacity-fade program search.

Reads the evolved BATTERY_MODEL Scheme, runs the validity funnel + DMCI fit-early/forecast-late
scoring against the saved target cells (gen_target.py), and returns the held-out skill as
combined_score = -holdout_rmse (OpenEvolve maximises) plus structural descriptors for MAP-Elites.
Invalid / unstable programs get a large negative score + an `artifacts` message OpenEvolve feeds
back into the next prompt (error-driven repair).
"""

from __future__ import annotations

import os
import sys

sys.setrecursionlimit(20000)
import importlib.util

import torch

torch.set_num_threads(1)

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from openevolve.evaluation_result import EvaluationResult

from experiments.exp_battery.config import BCFG, KSPLIT_LATE
from experiments.exp_battery.oe_score import score_evolved

BAD = -1.0e6
_ITERS = int(os.environ.get("BAT_ADAM_ITERS", 60))
_KSPLIT = int(os.environ.get("BAT_KSPLIT", KSPLIT_LATE))
_TARGET = os.environ.get("BAT_TARGET", os.path.join(_HERE, "..", "results", "target.pt"))

_OBS = None


def _obs():
    global _OBS
    if _OBS is None:
        _OBS = torch.load(_TARGET)["obs"]   # [N, T, 1]
    return _OBS


def _model_source(program_path: str) -> str:
    spec = importlib.util.spec_from_file_location("bat_prog", program_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.BATTERY_MODEL


def evaluate(program_path: str) -> EvaluationResult:
    obs = _obs()
    try:
        src = _model_source(program_path)
    except Exception as exc:  # noqa: BLE001
        return EvaluationResult(metrics={"runs_successfully": 0.0, "combined_score": BAD},
                                artifacts={"stderr": f"program did not import: {type(exc).__name__}: {exc}"})
    res = score_evolved(src, obs, _KSPLIT, cfg=BCFG, iters=_ITERS)
    if not res.get("ok"):
        return EvaluationResult(
            metrics={"runs_successfully": 0.0, "combined_score": BAD},
            artifacts={"stderr": f"scoring failed at [{res.get('stage')}]: {res.get('detail')}"})
    hr = float(res["holdout_rmse"])
    return EvaluationResult(metrics={
        "runs_successfully": 1.0,
        "combined_score": -hr,           # OpenEvolve maximises
        "holdout_rmse": hr,
        "train_nll": float(res["train_nll"]),
        "n_state": res["n_state"], "n_params": res["n_params"],
        "n_nonlin": res["n_nonlin"], "knee_capable": res["knee_capable"]})
