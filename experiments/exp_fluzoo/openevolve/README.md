# FluZoo × OpenEvolve

Runs the FluZoo program search through **[OpenEvolve](https://github.com/algorithmicsuperintelligence/openevolve)**
(the open-source **AlphaEvolve** implementation) as the evolutionary outer loop, with **DMCI** as the
differentiable inner loop. This buys the AlphaEvolve/OpenEvolve citations *and* its features -
**MAP-Elites quality-diversity**, **island** populations, **diff-based** edits, **inspiration sampling**,
and **cascade evaluation** - while DMCI calibrates each candidate program's parameters by exact gradients
to compute its held-out forecast fitness.

## Design

- **The genome** is the Scheme model in the `FLU_MODEL` string of `initial_program.py`. OpenEvolve's
  block markers are Python-comment-specific, so the model is wrapped in a Python file and the LLM
  diff-edits the Scheme inside the string (the `(params ...)` + `(loop ...)` two-form contract).
- **The evaluator** (`oe_evaluator.py`) is the bridge: `evaluate_stage1` runs the DMCI **validity funnel**
  (cheap cascade gate), `evaluate_stage2` runs **DMCI calibration + held-out forecast** scoring. It returns
  `EvaluationResult(metrics={combined_score=-val_rmse, val_rmse, n_compartments, n_params, n_harmonics}, ...)`.
  OpenEvolve maximises `combined_score`; structural metrics become MAP-Elites **feature dimensions**.
  Funnel errors are returned as `artifacts` so OpenEvolve feeds them into the next prompt (error-driven repair).
- **The operators** are an LLM ensemble of **non-thinking coder models** - `qwen2.5-coder:32b` (MindRouter)
  and `gpt-5.5` (OpenAI). qwen3.6 is avoided because OpenEvolve does not forward the `extra_body` needed to
  disable its thinking, which truncates the program.
- **Quality-diversity:** `feature_dimensions = [complexity, n_compartments]` over `num_islands` islands -
  the diversity pressure our hand-rolled loop lacked, and which matters most here because the fitness
  landscape is flat (most reasonable structures forecast the held-out seasons about equally).

## Status

**Plumbing validated end-to-end (mock evaluator, qwen2.5-coder via MindRouter):** OpenEvolve loads the
program, the LLM diff-edits the Scheme, the cascade + evaluator interface work, and the MAP-Elites grid
populates. The real run just swaps the mock for `oe_evaluator.py` (DMCI scoring, validated separately).

## Run

```bash
# fast plumbing test (no DMCI; needs only a reachable LLM)
python3 -m experiments.exp_fluzoo.openevolve.run_oe --mock --models qwen --iterations 3

# real run (on campus/HPC; install openevolve into the .venv first)
pip install openevolve                       # into the project .venv on the node
python3 -m experiments.exp_fluzoo.openevolve.run_oe \
    --models real --iterations 300 --workers 24 --output-dir results/oe
```

`run_oe.py` builds the `Config` programmatically (LLM ensemble with mixed endpoints, the FluZoo contract as
`prompt.system_message`, MAP-Elites + cascade). Results land in `oe_output/` (best program + checkpoints).

## vs. the hand-rolled `evolve.py`

`evolve.py` is a minimal version of this same paradigm (LLM mutation/crossover, islands, elitism). OpenEvolve
adds MAP-Elites quality-diversity, diff-based edits (more stable than full-program regeneration), inspiration
sampling, and mature infra - plus the citeable AlphaEvolve framing. It does **not** change the bottleneck
(DMCI scoring cost), so the season-batched fit/forecast and multi-node efficiency work still apply.
