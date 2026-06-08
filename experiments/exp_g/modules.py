############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# modules.py: Module composition engine for Experiment G. Symbolic modules are Scheme expression templates. Composition...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Module composition engine for Experiment G.

Symbolic modules are Scheme expression templates. Composition operators
combine them by string manipulation — the composed expression flows through
the same compiled DMCI interpreter as any individual module.
"""

from __future__ import annotations

import re
from .config import ModuleDef, MODULE_BY_NAME


def instantiate_module(module: ModuleDef,
                       prefix: str = "") -> tuple[str, list[str]]:
    pfx = f"{prefix}_" if prefix else ""
    expr = module.template.format(pfx=pfx)
    prefixed_params = [f"{pfx}{p}" for p in module.param_names]
    return expr, prefixed_params


def compose_sum(expr1: str, expr2: str) -> str:
    return f"(+ {expr1} {expr2})"


def compose_product(expr1: str, expr2: str) -> str:
    return f"(* {expr1} {expr2})"


def compose_chain(outer_module: ModuleDef, inner_expr: str,
                  outer_prefix: str) -> tuple[str, list[str]]:
    pfx = f"{outer_prefix}_" if outer_prefix else ""
    outer_template = outer_module.template.format(pfx=pfx)
    chained = re.sub(r'\bx\b', inner_expr, outer_template)
    prefixed_params = [f"{pfx}{p}" for p in outer_module.param_names]
    return chained, prefixed_params


def build_composition(op: str, mod1_name: str, mod2_name: str,
                      mod1_prefix: str = "m1",
                      mod2_prefix: str = "m2") -> tuple[str, list[str]]:
    mod1 = MODULE_BY_NAME[mod1_name]
    mod2 = MODULE_BY_NAME[mod2_name]

    if op == "sum":
        expr1, params1 = instantiate_module(mod1, mod1_prefix)
        expr2, params2 = instantiate_module(mod2, mod2_prefix)
        composed = compose_sum(expr1, expr2)
        return composed, params1 + params2
    elif op == "product":
        expr1, params1 = instantiate_module(mod1, mod1_prefix)
        expr2, params2 = instantiate_module(mod2, mod2_prefix)
        composed = compose_product(expr1, expr2)
        return composed, params1 + params2
    elif op == "chain":
        inner_expr, inner_params = instantiate_module(mod2, mod2_prefix)
        outer_expr, outer_params = compose_chain(
            mod1, inner_expr, mod1_prefix)
        return outer_expr, outer_params + inner_params
    else:
        raise ValueError(f"Unknown composition operator: {op}")


def composition_label(op: str, m1: str, m2: str) -> str:
    short = {"exponential_decay": "decay", "oscillation": "osc",
             "polynomial2": "poly", "sigmoid": "sigm",
             "power_law": "pow", "gaussian": "gauss"}
    s1 = short.get(m1, m1)
    s2 = short.get(m2, m2)
    return f"{op}({s1},{s2})"
