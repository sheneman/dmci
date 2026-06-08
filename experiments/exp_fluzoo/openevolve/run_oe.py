############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# run_oe.py: Run FluZoo program evolution through OpenEvolve (AlphaEvolve-style outer loop). OpenEvolve provides the...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Run FluZoo program evolution through OpenEvolve (AlphaEvolve-style outer loop).

OpenEvolve provides the evolutionary machinery -- LLM ensemble as mutation/crossover operator,
MAP-Elites quality-diversity + island population, inspiration sampling, cascade evaluation,
diff-based edits -- and FluZoo provides the differentiable substrate: each candidate Scheme model
is calibrated through the DMCI interpreter and scored on held-out forecast skill (oe_evaluator.py).

    # plumbing test (fast mock evaluator, GPT-5.5 only -- reachable off-campus):
    python3 -m experiments.exp_fluzoo.openevolve.run_oe --mock --models gpt55 --iterations 3
    # real run (on campus/HPC; DMCI scoring + qwen2.5-coder + GPT-5.5):
    python3 -m experiments.exp_fluzoo.openevolve.run_oe --iterations 300 --workers 24

Non-thinking coder models are used (qwen2.5-coder, GPT-5.5) because OpenEvolve does not forward
the `extra_body` needed to disable qwen3.6's thinking, which would truncate the program.
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

from experiments.exp_fluzoo.llm_generate import _PROMPT_RULES

HERE = Path(__file__).resolve().parent
INITIAL = HERE / "initial_program.py"
EVALUATOR = HERE / "oe_evaluator.py"
MOCK_EVALUATOR = HERE / "mock_evaluator.py"

SYSTEM_MESSAGE = (
    "You are improving an INFLUENZA FORECASTING MODEL to minimise its held-out forecast error.\n"
    "The file is Python, but the model you edit is the SCHEME program inside the `FLU_MODEL` string\n"
    "(a compiled differentiable interpreter, DMCI, runs it). Edit ONLY the Scheme inside FLU_MODEL,\n"
    "keep the triple-quoted string and the two-form `(params ...)` + `(loop ...)` structure intact,\n"
    "and make ONE meaningful epidemiological change per step (add/remove a compartment, change the\n"
    "seasonal forcing, the reporting model, or regional coupling). A lower validation RMSE is better.\n\n"
    + _PROMPT_RULES)

#: model presets -> list of (name, provider, weight). Default is the qwen3.6 thinking ensemble
#: (no max_tokens cap, so thinking no longer truncates the program).
MODEL_PRESETS = {
    "real": [("qwen/qwen3.6-35b", "mindrouter", 0.5), ("qwen/qwen3.6-27b", "mindrouter", 0.5)],
    "qwen36": [("qwen/qwen3.6-35b", "mindrouter", 0.5), ("qwen/qwen3.6-27b", "mindrouter", 0.5)],
    "coder": [("qwen2.5-coder:32b", "mindrouter", 1.0)],
    "gpt55": [("gpt-5.5", "openai", 1.0)],
}


def _model(name: str, provider: str, weight: float) -> LLMModelConfig:
    if provider == "openai":
        base = os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        key = os.environ.get("OPENAI_API_KEY", "")
    else:  # mindrouter (campus, OpenAI-compatible)
        base = os.environ.get("MINDROUTER_BASE_URL", "https://mindrouter.uidaho.edu/v1")
        key = os.environ.get("MINDROUTER_API_KEY", "")
    # max_tokens=65535: MindRouter accepts it and qwen3.6 only uses ~8k incl. reasoning, so this is
    # ample headroom for BOTH the thinking and the full program. A concrete high value also avoids
    # passing max_tokens=None through OpenEvolve's SDK path (which intermittently yielded None).
    # Generous timeout for the longer thinking generations.
    return LLMModelConfig(name=name, api_base=base, api_key=key, weight=weight,
                          temperature=0.85, max_tokens=65535, timeout=600)


def build_config(iterations: int, workers: int, models: str, mock: bool,
                 seed: int = 0, eval_timeout: int = 1800) -> Config:
    cfg = Config()
    cfg.max_iterations = iterations
    cfg.diff_based_evolution = True
    cfg.random_seed = seed
    cfg.checkpoint_interval = 20

    cfg.llm.models = [_model(*m) for m in MODEL_PRESETS[models]]
    if hasattr(cfg.llm, "reasoning_effort"):
        cfg.llm.reasoning_effort = "minimal"   # keep GPT-5.5 fast on code edits

    cfg.prompt.system_message = SYSTEM_MESSAGE
    cfg.prompt.num_top_programs = 3            # elite inspirations
    cfg.prompt.num_diverse_programs = 2        # diverse inspirations (exploration)

    # MAP-Elites quality-diversity over structural descriptors (the diversity our search needs)
    cfg.database.num_islands = 1 if mock else 4
    cfg.database.population_size = 40 if mock else 200
    cfg.database.migration_interval = 25
    cfg.database.feature_dimensions = ["complexity", "n_compartments"]
    cfg.database.feature_bins = 8
    cfg.database.random_seed = seed                # distinct seed per meta-island

    # Single evaluate(): our oe_evaluator.evaluate() already runs the validity funnel first and only
    # runs the expensive DMCI scoring if it passes -- so we get the cascade benefit (skip DMCI for
    # invalid programs) WITHOUT OpenEvolve's threshold gating, which otherwise blocked stage 2
    # (our stage-1 pass score of 0.0 never cleared the 0.5 threshold -> no DMCI scoring ran at all).
    cfg.evaluator.cascade_evaluation = False
    cfg.evaluator.parallel_evaluations = workers
    cfg.evaluator.timeout = eval_timeout           # whole-evaluate() cap (funnel + DMCI)
    cfg.evaluator.enable_artifacts = True      # feed funnel errors back into the next prompt
    cfg.max_tasks_per_child = 8                # recycle spawn workers to bound DMCI heap memory
    return cfg


def main():
    # Force spawn: the DMCI evaluator imports torch in the worker; forking a threaded controller
    # is fragile, and spawn is the proven start method for the interpreter (cf. exp_lim_enso).
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    ap = argparse.ArgumentParser(description="FluZoo evolution via OpenEvolve")
    ap.add_argument("--iterations", type=int, default=300)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--models", choices=list(MODEL_PRESETS), default="real")
    ap.add_argument("--mock", action="store_true",
                    help="fast structure-only evaluator (plumbing test, no DMCI)")
    ap.add_argument("--seed", type=int, default=0, help="meta-island seed (distinct per node)")
    ap.add_argument("--eval-timeout", type=int, default=1800,
                    help="per-program DMCI scoring cap (s); raise on slower nodes")
    ap.add_argument("--output-dir", default=str(HERE / "oe_output"))
    args = ap.parse_args()

    cfg = build_config(args.iterations, args.workers, args.models, args.mock,
                       seed=args.seed, eval_timeout=args.eval_timeout)
    evaluator = str(MOCK_EVALUATOR if args.mock else EVALUATOR)
    print(f"[run_oe] models={args.models} mock={args.mock} iters={args.iterations} "
          f"workers={args.workers} islands={cfg.database.num_islands}")
    result = run_evolution(initial_program=str(INITIAL), evaluator=evaluator, config=cfg,
                           iterations=args.iterations, output_dir=args.output_dir)
    print("\n[run_oe] DONE")
    print("best_score:", getattr(result, "best_score", None))
    code = getattr(result, "best_code", "") or ""
    print("best_code (first 600 chars):\n", code[:600])


if __name__ == "__main__":
    main()
