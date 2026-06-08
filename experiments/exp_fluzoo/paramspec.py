############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# paramspec.py: Generic parameter-schema -> unconstrained-transform layer. Because the LLM writes *whole* programs (the "freer...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Generic parameter-schema -> unconstrained-transform layer.

Because the LLM writes *whole* programs (the "freer generation" design), every
program declares its OWN set of free parameters, with its own natural ranges. We
cannot hand-author a parametrization per program the way exp_lim_enso does for its
fixed F/Q/R. Instead every generated program is prefixed with a small machine-readable
`(params ...)` schema block declaring, per parameter, a constraint KIND and an initial
(constrained) value:

    (params
      (beta0 positive   1.5)        ; > 0
      (amp   signed-unit 0.2)       ; (-1, 1)
      (phase free        0.0)       ; unconstrained
      (sigma unit        0.5)       ; (0, 1)
      (rho   unit        0.05 0.2)  ; (0, 0.2)  -- optional scale as the 4th field
      (s2    positive    4e-6))     ; > 0

This module parses that block and builds, for any seed, a dict of UNCONSTRAINED raw
leaf tensors (O(1) scale) plus a differentiable map raw -> constrained values. The
de-risk pilot proved this reparametrization is mandatory: optimizing raw constrained
parameters under one global learning rate diverges (initial-condition params run away).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from neural_compiler.parser.scheme_parser import tokenize, _parse_sexpr

KINDS = ("positive", "unit", "signed-unit", "free")


@dataclass(frozen=True)
class ParamSpec:
    name: str
    kind: str
    init: float          # the *constrained* initial value (what the modeller means)
    scale: float = 1.0   # constrained = scale * base(raw); lets `unit` cover (0, scale)

    def base_target(self) -> float:
        """The init value mapped back through `scale`, i.e. base(raw_init)."""
        return self.init / self.scale if self.scale != 0 else self.init


def _to_float(tok) -> float:
    if isinstance(tok, list):
        raise ValueError(f"expected a number, got a list: {tok}")
    return float(tok)


def parse_param_block(block_text: str) -> list[ParamSpec]:
    """Parse a `(params (name kind init [scale]) ...)` form into ParamSpecs."""
    datum, _ = _parse_sexpr(tokenize(block_text), 0)
    if not isinstance(datum, list) or not datum or datum[0] != "params":
        raise ValueError("parameter block must be a (params ...) form")
    specs: list[ParamSpec] = []
    seen: set[str] = set()
    for entry in datum[1:]:
        if not isinstance(entry, list) or len(entry) < 3:
            raise ValueError(f"bad param entry (need name kind init [scale]): {entry}")
        name, kind = entry[0], entry[1]
        if not isinstance(name, str) or not isinstance(kind, str):
            raise ValueError(f"param name/kind must be symbols: {entry}")
        if kind not in KINDS:
            raise ValueError(f"unknown param kind {kind!r} (allowed: {KINDS})")
        if name in seen:
            raise ValueError(f"duplicate parameter {name!r}")
        seen.add(name)
        # entry is (name kind init) or (name kind init scale).
        init = _to_float(entry[2])
        scale = _to_float(entry[3]) if len(entry) >= 4 else 1.0
        specs.append(ParamSpec(name=name, kind=kind, init=init, scale=scale))
    return specs


def _raw_init(spec: ParamSpec) -> float:
    """Invert the transform: constrained init -> raw (unconstrained) init."""
    t = spec.base_target()
    if spec.kind == "positive":
        t = max(t, 1e-6)
        return math.log(math.expm1(t))           # inverse softplus
    if spec.kind == "unit":
        u = min(max(t, 1e-4), 1.0 - 1e-4)
        return math.log(u / (1.0 - u))            # logit
    if spec.kind == "signed-unit":
        u = min(max(t, -0.999), 0.999)
        return math.atanh(u)
    return t                                       # free


def make_raw(specs: list[ParamSpec], seed: int = 0, jitter: float = 0.0,
             requires_grad: bool = True) -> dict[str, torch.Tensor]:
    """Build the unconstrained raw leaf tensors, optionally jittered for multi-start."""
    g = torch.Generator().manual_seed(int(seed))
    raw: dict[str, torch.Tensor] = {}
    for s in specs:
        base = _raw_init(s)
        if jitter > 0.0:
            base = base + jitter * float(torch.randn((), generator=g))
        raw[s.name] = torch.tensor(base, dtype=torch.float32, requires_grad=requires_grad)
    return raw


def constrain(specs: list[ParamSpec],
              raw: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """raw (unconstrained) -> epidemiologically valid params, differentiably."""
    out: dict[str, torch.Tensor] = {}
    for s in specs:
        r = raw[s.name]
        if s.kind == "positive":
            b = F.softplus(r)
        elif s.kind == "unit":
            b = torch.sigmoid(r)
        elif s.kind == "signed-unit":
            b = torch.tanh(r)
        else:  # free
            b = r
        out[s.name] = s.scale * b
    return out


def param_count(specs: list[ParamSpec]) -> int:
    """Number of free scalar parameters (for AIC/BIC across the zoo)."""
    return len(specs)
