############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# runner.py: Single-run executor for Experiment B.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Single-run executor for Experiment B."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .config import ExpBConfig
from .models import ModelSpec, MODEL_BY_NAME
from .baselines import TrainResult, run_method


def run_single(
    method: str,
    model_name: str,
    seed: int,
    cfg: ExpBConfig,
    output_dir: Path | None = None,
) -> TrainResult:
    model = MODEL_BY_NAME[model_name]
    result = run_method(method, model, cfg, seed)

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        tag = f"{method}_{model_name}_{seed:02d}"

        csv_path = output_dir / f"{tag}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["epoch", "loss", "grad_norm", "wall_time_s"]
            header += [f"{n}_value" for n in model.param_names]
            writer.writerow(header)
            for i in range(len(result.loss_history)):
                row = [i, result.loss_history[i], result.grad_norm_history[i],
                       result.wall_time_history[i]]
                for n in model.param_names:
                    if n in result.param_history:
                        row.append(result.param_history[n][i])
                writer.writerow(row)

        json_path = output_dir / f"{tag}.json"
        with open(json_path, "w") as f:
            json.dump({
                "method": result.method,
                "model_name": result.model_name,
                "seed": result.seed,
                "converged": result.converged,
                "convergence_epoch": result.convergence_epoch,
                "final_loss": result.final_loss,
                "final_param_errors": result.final_param_errors,
                "total_wall_time": result.total_wall_time,
                "tier": model.tier,
                "domain": model.domain,
            }, f, indent=2)

    return result
