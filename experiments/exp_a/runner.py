############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# runner.py: Execute one (method, program, seed) configuration and save results to CSV.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Execute one (method, program, seed) configuration and save results to CSV."""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

from .config import ExpAConfig, DEFAULT
from .programs import ProgramSpec, PROGRAMS_BY_NAME, ALL_PROGRAMS
from .baselines import run_method, TrainResult


def result_path(output_dir: str, method: str, program: str, seed: int,
                ext: str = "csv") -> Path:
    return Path(output_dir) / f"{method}_{program}_{seed:02d}.{ext}"


def save_result(result: TrainResult, output_dir: str) -> Path:
    os.makedirs(output_dir, exist_ok=True)
    csv_path = result_path(output_dir, result.method, result.program,
                           result.seed)
    param_names = sorted(result.param_history.keys())

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["epoch", "loss"]
        for pn in param_names:
            header += [f"param_{pn}_value", f"param_{pn}_error"]
        header += ["grad_norm", "wall_time_s"]
        writer.writerow(header)

        spec = PROGRAMS_BY_NAME[result.program]
        for i in range(len(result.loss_history)):
            row = [i, result.loss_history[i]]
            for pn in param_names:
                val = result.param_history[pn][i]
                err = abs(val - spec.target_values[pn])
                row += [val, err]
            row += [result.grad_norm_history[i], result.wall_time_history[i]]
            writer.writerow(row)

    # Summary JSON
    json_path = result_path(output_dir, result.method, result.program,
                            result.seed, ext="json")
    summary = {
        "method": result.method,
        "program": result.program,
        "seed": result.seed,
        "converged": result.converged,
        "convergence_epoch": result.convergence_epoch,
        "final_loss": result.final_loss,
        "final_param_errors": result.final_param_errors,
        "total_wall_time": result.total_wall_time,
    }
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    return csv_path


def run_single(method: str, program_name: str, seed: int,
               cfg: ExpAConfig = DEFAULT) -> TrainResult:
    spec = PROGRAMS_BY_NAME[program_name]
    result = run_method(method, spec, cfg, seed)
    save_result(result, cfg.output_dir)
    return result


def is_complete(output_dir: str, method: str, program: str,
                seed: int) -> bool:
    return result_path(output_dir, method, program, seed).exists()
