############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# llm_sources.py: Build DMCI (interp) and direct-compilation sources from the cached LLM programs. Experiment B's LLM step...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Build DMCI (interp) and direct-compilation sources from the cached LLM programs.

Experiment B's LLM step generates a single standalone Scheme program per model
(top-level `define`s followed by a final call expression). This module applies a
single UNIFORM, model-independent transform to each cached program to produce the
two forms the training methods consume:

  - ``direct_source``: the generated program, compiled directly (no interpreter).
    The cached program is already in this form (it is exactly what ``_validate`` in
    ``llm_generate.py`` compiles to check ``compiles``/``correct``).
  - ``interp_source``: the same program wrapped as quoted DATA and handed to the
    compiled self-hosted evaluator via ``(scheme-eval-program (list 'form ...) env)``
    (this is DMCI).

No per-model editing is performed: ``make_interp_source`` / ``make_direct_source``
apply the identical wrapping to all 15 programs. This is what lets Experiment B
claim the LLM-generated programs are compiled *without modification* — the only
transform is the model-independent evaluator wrapper, not hand-authored rewrites.
"""

from __future__ import annotations

from .models import ModelSpec, _all_input_names, EVALUATOR_SOURCE, _make_env
from .llm_generate import load_from_cache


def split_top_level_forms(source: str) -> list[str]:
    """Split a Scheme source string into its top-level S-expressions.

    Tracks parenthesis depth and skips ``;`` line comments. The cached programs
    contain no string or character literals, so a paren counter is sufficient.
    """
    forms: list[str] = []
    depth = 0
    cur: list[str] = []
    i, n = 0, len(source)
    while i < n:
        c = source[i]
        if c == ";":  # line comment to end of line
            while i < n and source[i] != "\n":
                i += 1
            continue
        cur.append(c)
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                form = "".join(cur).strip()
                if form:
                    forms.append(form)
                cur = []
        i += 1
    tail = "".join(cur).strip()
    if tail:  # a bare top-level atom (not expected, but keep it)
        forms.append(tail)
    return forms


def make_direct_source(generated_source: str) -> str:
    """Direct-compilation form: the generated program as-is."""
    return generated_source.strip() + "\n"


def make_interp_source(generated_source: str, names: list[str]) -> str:
    """DMCI form: the program quoted as data, evaluated by the compiled interpreter."""
    forms = split_top_level_forms(generated_source)
    quoted = "\n    ".join("'" + f for f in forms)
    env = _make_env(names)
    return (
        EVALUATOR_SOURCE
        + "\n(scheme-eval-program\n  (list\n    "
        + quoted
        + ")\n  "
        + env
        + ")\n"
    )


def llm_sources_for(model: ModelSpec) -> tuple[str, str]:
    """Return ``(interp_source, direct_source)`` derived from the cached LLM program.

    Raises ``FileNotFoundError`` if there is no cache entry, or ``ValueError`` if the
    cached entry is not marked ``compiles`` and ``correct``.
    """
    cached = load_from_cache(model.name)
    if cached is None:
        raise FileNotFoundError(f"No LLM cache for {model.name}")
    if not (cached.compiles and cached.correct):
        raise ValueError(
            f"Cache for {model.name} not marked compiles+correct "
            f"(compiles={cached.compiles}, correct={cached.correct})"
        )
    names = _all_input_names(model)
    gen = cached.generated_source
    return make_interp_source(gen, names), make_direct_source(gen)


def apply_llm_sources(models: list[ModelSpec]) -> list[str]:
    """In-place: replace each model's interp_source/direct_source with the cache-derived
    forms. Returns the list of model names successfully overridden."""
    overridden = []
    for m in models:
        interp, direct = llm_sources_for(m)
        m.interp_source = interp
        m.direct_source = direct
        overridden.append(m.name)
    return overridden
