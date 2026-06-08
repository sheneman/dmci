############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# sexp_eval.py: Generic evaluators over an LLM-emitted Scheme s-expression, so a *real* LLM program string can be turned into...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Generic evaluators over an LLM-emitted Scheme s-expression, so a *real*
LLM program string can be turned into an Exp J `Program` (ground truth + B2 JAX
forward + engineering proxy) without hand-writing per-program code.

This handles the CLOSED-FORM subset (the family the LLM is asked for): the ops
{+, -, *, /, exp, log, sin, cos, pow, sqrt, abs} over the driver `x` and named
parameters. Programs containing `define`/`if`/`lambda`/`let`/list ops are *not*
closed form (`is_closed_form` returns False) — for those, B1 (lambdify) fails
coverage exactly as designed, ground truth comes from DMCI, and B2's port is
costed (LOC) but not auto-emitted. We reuse the s-expression parser already
written for Exp F's arity validator so parsing matches the rest of the pipeline.
"""

from __future__ import annotations

import math

from experiments.exp_f.exp_f import _parse_sexp

_BINARY = {"+", "-", "*", "/", "pow"}
_UNARY = {"exp", "log", "sin", "cos", "sqrt", "abs"}
_NON_CLOSED = {"define", "lambda", "let", "letrec", "if", "cond", "begin",
               "cons", "car", "cdr", "list", "quote", "scheme-eval",
               "scheme-eval-program", "set!"}


def parse(expr: str):
    return _parse_sexp(expr)


def _atom_num(tok):
    try:
        return float(tok)
    except (ValueError, TypeError):
        return None


def is_closed_form(tree) -> bool:
    """True iff the tree uses only the closed-form op set (no define/if/list/...)."""
    if not isinstance(tree, list):
        return True
    if not tree:
        return False
    head = tree[0]
    if isinstance(head, str) and head in _NON_CLOSED:
        return False
    if isinstance(head, str) and head not in _BINARY and head not in _UNARY:
        return False
    return all(is_closed_form(a) for a in tree[1:])


def free_symbols(tree) -> set[str]:
    """Symbol atoms that are neither numbers nor operators (drivers + params)."""
    out: set[str] = set()

    def walk(n):
        if isinstance(n, list):
            for a in n[1:]:
                walk(a)
            return
        if _atom_num(n) is None and n not in _BINARY and n not in _UNARY:
            out.add(n)
    walk(tree)
    return out


def node_count(tree) -> int:
    """AST node count — the per-structure engineering (LOC) proxy for B2."""
    if not isinstance(tree, list):
        return 1
    return 1 + sum(node_count(a) for a in tree[1:])


def token_count(expr: str) -> int:
    """Non-paren token count of a raw program string — size proxy for multi-form
    (recursive) programs that `parse` cannot ingest as a single expression."""
    toks = expr.replace("(", " ( ").replace(")", " ) ").split()
    return sum(1 for t in toks if t not in ("(", ")"))


def eval_python(tree, env: dict) -> float:
    """Closed-form Python evaluator. `env` maps symbol -> float."""
    if not isinstance(tree, list):
        v = _atom_num(tree)
        return v if v is not None else env[tree]
    head, args = tree[0], tree[1:]
    if head in _UNARY:
        a = eval_python(args[0], env)
        return {"exp": math.exp, "log": math.log, "sin": math.sin,
                "cos": math.cos, "sqrt": math.sqrt, "abs": abs}[head](a)
    a = eval_python(args[0], env)
    b = eval_python(args[1], env)
    if head == "+":
        return a + b
    if head == "-":
        return a - b
    if head == "*":
        return a * b
    if head == "/":
        return a / b
    return a ** b  # pow


def eval_jax(tree, jnp, env: dict):
    """Closed-form JAX evaluator (the B2 hand-port stand-in). `env` -> arrays/scalars."""
    if not isinstance(tree, list):
        v = _atom_num(tree)
        return v if v is not None else env[tree]
    head, args = tree[0], tree[1:]
    if head in _UNARY:
        a = eval_jax(args[0], jnp, env)
        return {"exp": jnp.exp, "log": jnp.log, "sin": jnp.sin,
                "cos": jnp.cos, "sqrt": jnp.sqrt, "abs": jnp.abs}[head](a)
    a = eval_jax(args[0], jnp, env)
    b = eval_jax(args[1], jnp, env)
    if head == "+":
        return a + b
    if head == "-":
        return a - b
    if head == "*":
        return a * b
    if head == "/":
        return a / b
    return a ** b  # pow


def make_jax_forward(tree):
    """Build a B2-signature forward: jf(jnp, inputs_dict, params_dict) -> array."""
    def jf(jnp, I, P, _t=tree):
        return eval_jax(_t, jnp, {**I, **P})
    return jf
