############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# run_battery.py: Run battery capacity-fade program evolution through OpenEvolve (AlphaEvolve-style outer loop). Mirrors...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Run battery capacity-fade program evolution through OpenEvolve (AlphaEvolve-style outer loop).

Mirrors experiments/exp_fluzoo/openevolve/run_oe.py: OpenEvolve supplies the LLM-ensemble mutation
operator + MAP-Elites quality-diversity + island population; the battery evaluator supplies the
differentiable substrate (each candidate Scheme degradation model is calibrated through DMCI and
scored on held-out forecast skill). Edit ONLY the Scheme inside BATTERY_MODEL.

    python3 -m experiments.exp_battery.openevolve.run_battery --models real --iterations 200 --workers 24
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[3] / ".env")
except Exception:  # noqa: BLE001
    pass

from openevolve import run_evolution
from openevolve.config import Config, LLMModelConfig

HERE = Path(__file__).resolve().parent
INITIAL = HERE / "initial_program.py"
EVALUATOR = HERE / "oe_evaluator.py"

_BATTERY_RULES = """
HARD CONSTRAINTS for the Scheme degradation model (a compiled interpreter runs it):
- Keep the TWO top-level forms: a (params (name kind init [scale]) ...) schema then a (loop ...) model.
  kinds: positive (>0), unit (0..1), signed-unit (-1..1), free. Declare EVERY symbol the loop uses.
- The loop is a per-CYCLE rollout: (loop ((k 0) ... (yhat (zeros 1)) (L 0.0)) (if (= k NWEEKS) L (let* (...) (recur (+ k 1) ... ypred (+ L nll))))).
  KEEP the `yhat` loop variable (the predicted capacity, (vec Q)) -- it is required to forecast.
- ALL arithmetic is BINARY. Allowed ops ONLY: + - * / = < > <= >= sqrt pow exp log abs min max
  dot vec ref zeros ones scale. Any other head symbol silently evaluates to 0 (corrupts the model).
  There is NO tan/sign/floor/clamp/where/tanh/sigmoid: build sigmoids from exp, saturations from min/max.
- Capacity Q is a function of the cycle counter k (autonomous). The Gaussian NLL of the observed
  capacity (ref obs k) vs (vec Q) accumulates into L with a floored variance (+ s2 1e-06).
- To avoid pow(0,*) NaN gradients at k=0, write powers as (pow (/ (+ k 1e-6) tau) beta), etc.

MODELING IDEAS to explore (the discrete structure search): more degradation reservoirs (lithium
inventory vs active material) combined by a min() bottleneck to create a KNEE; power-law vs sqrt vs
stretched-exponential vs sigmoidal fade; a crack/plating gate that switches on extra loss; reporting
the bottleneck of two declining states. Make ONE meaningful structural change per step.
"""

SYSTEM_MESSAGE = (
    "You are improving a LITHIUM-ION BATTERY capacity-fade model to minimise its held-out forecast\n"
    "error. The model you edit is the SCHEME program inside the `BATTERY_MODEL` string. Keep the\n"
    "triple-quoted string and the two-form (params ...) + (loop ...) structure intact, and make ONE\n"
    "meaningful change per step (add/remove a degradation reservoir, change the fade law, add a knee\n"
    "mechanism, change the observation). A lower held-out RMSE is better.\n" + _BATTERY_RULES)

MODEL_PRESETS = {
    "real": [("qwen/qwen3.6-35b", "mindrouter", 0.5), ("qwen/qwen3.6-27b", "mindrouter", 0.5)],
    "coder": [("qwen2.5-coder:32b", "mindrouter", 1.0)],
}


def _model(name, provider, weight):
    base = os.environ.get("MINDROUTER_BASE_URL", "https://mindrouter.uidaho.edu/v1")
    key = os.environ.get("MINDROUTER_API_KEY", "")
    return LLMModelConfig(name=name, api_base=base, api_key=key, weight=weight,
                          temperature=0.85, max_tokens=65535, timeout=600)


def build_config(iterations, workers, models, seed=0, eval_timeout=1800) -> Config:
    cfg = Config()
    cfg.max_iterations = iterations
    cfg.diff_based_evolution = True
    cfg.random_seed = seed
    cfg.checkpoint_interval = 20
    cfg.llm.models = [_model(*m) for m in MODEL_PRESETS[models]]
    cfg.prompt.system_message = SYSTEM_MESSAGE
    cfg.prompt.num_top_programs = 3
    cfg.prompt.num_diverse_programs = 2
    cfg.database.num_islands = 4
    cfg.database.population_size = 200
    cfg.database.migration_interval = 25
    cfg.database.feature_dimensions = ["complexity", "knee_capable"]
    cfg.database.feature_bins = 8
    cfg.database.random_seed = seed
    cfg.evaluator.cascade_evaluation = False   # the evaluator gates internally (funnel then DMCI)
    cfg.evaluator.parallel_evaluations = workers
    cfg.evaluator.timeout = eval_timeout
    cfg.evaluator.enable_artifacts = True
    cfg.max_tasks_per_child = 8
    return cfg


def main():
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    ap = argparse.ArgumentParser(description="Battery capacity-fade evolution via OpenEvolve")
    ap.add_argument("--iterations", type=int, default=200)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--models", choices=list(MODEL_PRESETS), default="real")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-timeout", type=int, default=1800)
    ap.add_argument("--output-dir", default=str(HERE / "bat_output"))
    args = ap.parse_args()

    cfg = build_config(args.iterations, args.workers, args.models, seed=args.seed,
                       eval_timeout=args.eval_timeout)
    print(f"[run_battery] models={args.models} iters={args.iterations} workers={args.workers} "
          f"islands={cfg.database.num_islands} seed={args.seed}")
    result = run_evolution(initial_program=str(INITIAL), evaluator=str(EVALUATOR), config=cfg,
                           iterations=args.iterations, output_dir=args.output_dir)
    print("\n[run_battery] DONE  best_score:", getattr(result, "best_score", None))


if __name__ == "__main__":
    main()
