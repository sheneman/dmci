############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# llm_generate.py: LLM-based Scheme program generation for Experiment B. Two modes: 1. Live generation: call an LLM API to...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""LLM-based Scheme program generation for Experiment B.

Two modes:
1. Live generation: call an LLM API to generate Scheme from NL descriptions.
2. Cached generation: load pre-generated programs from disk for reproducibility.

The experiment uses cached mode by default. Live mode is used to populate the
cache initially and to measure generation success rates.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from dataclasses import dataclass

import dotenv
dotenv.load_dotenv(Path(__file__).parent.parent.parent / ".env")

from .models import ModelSpec, ALL_MODELS


CACHE_DIR = Path(__file__).parent / "llm_cache"

SYSTEM_PROMPT = """\
You are an expert programmer. Given a natural-language description of a scientific \
model or computation, generate a Scheme program that implements it.

Requirements:
- Use standard Scheme syntax (define, lambda, let, if, letrec, cons, car, cdr, etc.)
- The program MUST end with a top-level expression that calls the main function \
using the exact input/parameter variable names provided. Do NOT end with just a define.
- Learnable parameters should appear as top-level input variables (not hard-coded).
- Input variables should also appear as top-level inputs.
- CRITICAL: All arithmetic operators (+, -, *, /) take exactly TWO arguments. \
Write (* a (* b c)) instead of (* a b c). Write (+ a (+ b c)) instead of (+ a b c).
- Use only these primitives: +, -, *, /, pow, exp, log, sqrt, sin, cos, abs, min, max, <, >, <=, >=, =, not, and, or, if
- Do NOT use named let (e.g., (let loop ((i 0)) ...)). Instead use letrec with \
explicit lambdas: (letrec ((loop (lambda (i) ...))) (loop 0))
- SCOPING: Inner lambdas inside letrec CANNOT access variables from the enclosing \
define. Pass all needed variables as explicit arguments to the inner lambda. \
For example, write (letrec ((f (lambda (guess n x) ...))) (f init 5 x)) \
NOT (letrec ((f (lambda (guess n) ... x ...))) (f init 5)).
- Prefer flat top-level defines over letrec when possible.
- Do NOT use list, null?, or pair? — use only numeric recursion with counters.
- Do NOT define top-level lambdas that reference free variables from other defines. \
Instead, pass values as function arguments.
- For recursive programs, use define with explicit recursion or letrec with lambda.
- For higher-order programs, use lambda and function application.
- Do not use set!, call/cc, or I/O operations.
- Use the EXACT variable names given in the input/parameter list as top-level variables.
- Return ONLY the Scheme code, no explanation.
"""


@dataclass
class GenerationResult:
    model_name: str
    nl_description: str
    generated_source: str
    compiles: bool
    correct: bool
    error: str | None = None


def _extract_scheme(response: str) -> str:
    code_match = re.search(r"```(?:scheme|lisp)?\s*\n(.*?)```", response, re.DOTALL)
    if code_match:
        return code_match.group(1).strip()
    return response.strip()


def generate_live(model: ModelSpec, api: str = "anthropic") -> GenerationResult:
    """Generate Scheme source from NL description using an LLM API.

    Args:
        model: The ModelSpec with the NL description.
        api: Which API to use ('anthropic' or 'openai').

    Returns:
        GenerationResult with the generated source and validation results.
    """
    user_prompt = (
        f"Scientific model description:\n{model.nl_description}\n\n"
        f"Input variables: {', '.join(model.input_names)}\n"
        f"Learnable parameters: {', '.join(model.param_names)}\n\n"
        f"Generate a Scheme program that computes this model. "
        f"The program should take {', '.join(model.input_names + model.param_names)} "
        f"as inputs and return the computed value."
    )

    if api == "anthropic":
        generated = _call_anthropic(user_prompt)
    elif api == "openai":
        generated = _call_openai(user_prompt)
    elif api == "mindrouter":
        generated = _call_mindrouter(user_prompt)
    else:
        raise ValueError(f"Unknown API: {api}")

    source = _extract_scheme(generated)
    compiles, correct, error = _validate(model, source)

    result = GenerationResult(
        model_name=model.name,
        nl_description=model.nl_description,
        generated_source=source,
        compiles=compiles,
        correct=correct,
        error=error,
    )

    _save_to_cache(result)
    return result


def _call_anthropic(prompt: str) -> str:
    import anthropic
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def _call_openai(prompt: str) -> str:
    import openai
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=1024,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content


def _call_mindrouter(prompt: str) -> str:
    import os
    import openai
    base_url = os.environ.get("MINDROUTER_BASE_URL", "https://mindrouter.uidaho.edu/v1")
    api_key = os.environ.get("MINDROUTER_API_KEY")
    model = os.environ.get("MINDROUTER_MODEL", "qwen/qwen3.6-35b")
    if not api_key:
        raise ValueError("MINDROUTER_API_KEY not set. Add it to .env or environment.")
    client = openai.OpenAI(base_url=base_url, api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        max_tokens=8192,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    content = response.choices[0].message.content
    if not content:
        content = getattr(response.choices[0].message, "reasoning_content", None)
    if not content:
        raise ValueError("Empty response from MindRouter")
    return content


def _validate(model: ModelSpec, source: str) -> tuple[bool, bool, str | None]:
    """Check if the generated source compiles and produces correct output."""
    try:
        from neural_compiler.compiler import compile_program
        from neural_compiler.evaluator import evaluate
        from neural_compiler.runtime.tagged_value import make_float, unwrap_number

        all_inputs = {n: None for n in model.input_names + model.param_names}
        graph = compile_program(source, inputs=all_inputs, prelude=True)

        test_vals = {n: 1.0 for n in model.input_names}
        test_vals.update(model.target_values)
        tagged = {n: make_float(v) for n, v in test_vals.items()}

        result = evaluate(graph, tagged)
        pred = float(unwrap_number(result))

        expected = model.ground_truth(
            **{n: test_vals[n] for n in model.input_names},
            **model.target_values,
        )

        correct = abs(pred - expected) < 1e-4 * (1 + abs(expected))
        return True, correct, None

    except Exception as e:
        return False, False, str(e)


def _save_to_cache(result: GenerationResult):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{result.model_name}.json"
    with open(path, "w") as f:
        json.dump({
            "model_name": result.model_name,
            "nl_description": result.nl_description,
            "generated_source": result.generated_source,
            "compiles": result.compiles,
            "correct": result.correct,
            "error": result.error,
        }, f, indent=2)


def load_from_cache(model_name: str) -> GenerationResult | None:
    path = CACHE_DIR / f"{model_name}.json"
    if not path.exists():
        return None
    with open(path) as f:
        d = json.load(f)
    return GenerationResult(**d)


def generate_all(api: str = "anthropic", force: bool = False):
    """Generate Scheme for all models and cache results."""
    results = []
    for model in ALL_MODELS:
        if not force:
            cached = load_from_cache(model.name)
            if cached:
                print(f"  {model.name}: cached (compiles={cached.compiles}, correct={cached.correct})")
                results.append(cached)
                continue
        print(f"  {model.name}: generating via {api}...")
        result = generate_live(model, api=api)
        print(f"    compiles={result.compiles}, correct={result.correct}")
        if result.error:
            print(f"    error: {result.error}")
        results.append(result)
    return results
