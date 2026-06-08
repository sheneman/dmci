############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# corpus.py: Program corpus for Exp J: a reproducible space of *structurally distinct*, "runtime-generated" Scheme programs...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Program corpus for Exp J: a reproducible space of *structurally distinct*,
"runtime-generated" Scheme programs with a controlled closed-form vs. recursive mix.

Structural distinctness is essential: if programs differed only in target values they would
share a Scheme string (and JAX would cache the jit, and B2 would reuse a port), collapsing the
very costs the experiment measures. So each program is a distinct random expression tree.

A tree is emitted three consistent ways from one spec, so all arms compute the same function:
  - to_scheme  -> the Scheme string an LLM would emit (pre-cached; fed to DMCI / parsed by B1)
  - to_python  -> ground truth (data generation, recovery scoring)
  - to_jax     -> the hand-ported JAX forward (the B2 baseline)

Closed-form trees are expressible by `lambdify` (B1). Recursive trees wrap a closed-form RHS in
an Euler loop -- which `lambdify` cannot translate (coverage failure), so only B2 / DMCI run them.

Ops are restricted to the common, domain-safe subset of {+, -, *, exp, sin, cos} supported
identically by scheme-eval, SymPy/lambdify, and JAX; programs are rejection-sampled to be finite
over their input range.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch

_BINARY = ["+", "-", "*"]
_UNARY = ["exp", "sin", "cos"]
_STEPS = 15
_DT = 0.1


# --- tree generation --------------------------------------------------------

def _gen_tree(g, depth, params, max_params=4):
    """Random expression tree over `x` and params p0..; appends param names to `params`."""
    leaf = depth <= 0 or (float(torch.rand((), generator=g)) < 0.35)
    if leaf:
        r = float(torch.rand((), generator=g))
        if r < 0.55:
            return ("x",)
        if r < 0.85 and len(params) < max_params:
            name = f"p{len(params)}"
            params.append(name)
            return ("param", name)
        return ("const", round(0.5 + 1.5 * float(torch.rand((), generator=g)), 2))
    if float(torch.rand((), generator=g)) < 0.4:
        op = _UNARY[int(torch.randint(len(_UNARY), (1,), generator=g))]
        return (op, _gen_tree(g, depth - 1, params, max_params))
    op = _BINARY[int(torch.randint(len(_BINARY), (1,), generator=g))]
    return (op, _gen_tree(g, depth - 1, params, max_params),
            _gen_tree(g, depth - 1, params, max_params))


def _to_scheme(node) -> str:
    h = node[0]
    if h == "x":
        return "x"
    if h == "param":
        return node[1]
    if h == "const":
        return repr(node[1])
    if h in _UNARY:
        return f"({h} {_to_scheme(node[1])})"
    return f"({h} {_to_scheme(node[1])} {_to_scheme(node[2])})"


def _eval_python(node, x, P):
    h = node[0]
    if h == "x":
        return x
    if h == "param":
        return P[node[1]]
    if h == "const":
        return node[1]
    if h == "exp":
        return math.exp(_eval_python(node[1], x, P))
    if h == "sin":
        return math.sin(_eval_python(node[1], x, P))
    if h == "cos":
        return math.cos(_eval_python(node[1], x, P))
    a = _eval_python(node[1], x, P)
    b = _eval_python(node[2], x, P)
    return {"+": a + b, "-": a - b, "*": a * b}[h]


def _eval_jax(node, jnp, x, P):
    h = node[0]
    if h == "x":
        return x
    if h == "param":
        return P[node[1]]
    if h == "const":
        return node[1]
    if h == "exp":
        return jnp.exp(_eval_jax(node[1], jnp, x, P))
    if h == "sin":
        return jnp.sin(_eval_jax(node[1], jnp, x, P))
    if h == "cos":
        return jnp.cos(_eval_jax(node[1], jnp, x, P))
    a = _eval_jax(node[1], jnp, x, P)
    b = _eval_jax(node[2], jnp, x, P)
    return {"+": a + b, "-": a - b, "*": a * b}[h]


def _node_count(node) -> int:
    if node[0] in ("x", "param", "const"):
        return 1
    return 1 + sum(_node_count(c) for c in node[1:])


# --- Program record ---------------------------------------------------------

@dataclass
class Program:
    pid: int
    kind: str                       # "closed_form" | "recursive"
    scheme: str
    input_names: list[str]
    param_names: list[str]
    targets: dict[str, float]
    bounds: dict[str, tuple[float, float]]
    input_ranges: dict[str, tuple[float, float]]
    ground_truth: Callable          # (**inputs, **params) -> float
    jax_forward: Callable           # (jnp, inputs_dict, params_dict) -> array
    port_loc: int                   # human LOC to hand-port this structure to JAX (B2)


_XR = (0.0, 2.0)
_PR = (0.2, 1.5)


def _finite_over_range(gt, params, targets, n=8) -> bool:
    for i in range(n):
        x = _XR[0] + (_XR[1] - _XR[0]) * (i / (n - 1))
        try:
            v = gt(x=x, **targets)
        except (OverflowError, ValueError, ZeroDivisionError):
            return False
        if not math.isfinite(v) or abs(v) > 1e4:
            return False
    return True


def _make_closed_form(g, pid, seed) -> Program:
    for _attempt in range(50):
        params: list[str] = []
        tree = _gen_tree(g, depth=3, params=params)
        if not params:                          # ensure ≥1 learnable param
            continue
        targets = {p: float(_PR[0] + (_PR[1] - _PR[0]) * torch.rand((), generator=g))
                   for p in params}
        def gt(x, _tree=tree, **P):
            return _eval_python(_tree, x, P)
        if not _finite_over_range(gt, params, targets):
            continue
        return Program(
            pid=pid, kind="closed_form", scheme=_to_scheme(tree),
            input_names=["x"], param_names=params, targets=targets,
            bounds={p: _PR for p in params}, input_ranges={"x": _XR},
            ground_truth=gt,
            jax_forward=(lambda jnp, I, P, _tree=tree: _eval_jax(_tree, jnp, I["x"], P)),
            port_loc=_node_count(tree))
    raise RuntimeError("closed-form rejection sampling failed")


def _make_recursive(g, pid, seed) -> Program:
    """Euler relaxation y_{n+1} = y_n + dt*(g(x) - tau*y_n): g a closed-form tree in the
    driver x, tau a positive damping param. Stable by construction (damped), genuinely
    recursive/stateful, and NOT expressible by lambdify (the loop)."""
    for _attempt in range(80):
        params: list[str] = []
        gtree = _gen_tree(g, depth=2, params=params)   # g(x), closed-form in the driver x
        if not params:
            continue
        tau = f"p{len(params)}"
        params.append(tau)                              # positive damping parameter
        targets = {p: float(_PR[0] + (_PR[1] - _PR[0]) * torch.rand((), generator=g))
                   for p in params}

        def gt(x, _g=gtree, _tau=tau, **P):
            y = 1.0
            for _ in range(_STEPS):
                y = y + _DT * (_eval_python(_g, x, P) - P[_tau] * y)
            return y
        ok = True
        for i in range(6):
            xx = _XR[0] + (_XR[1] - _XR[0]) * (i / 5)
            try:
                v = gt(xx, **targets)
            except (OverflowError, ValueError, ZeroDivisionError):
                ok = False
                break
            if not math.isfinite(v) or abs(v) > 1e4:
                ok = False
                break
        if not ok:
            continue
        g_scm = _to_scheme(gtree)
        scheme = (f"(define (f y n) (if (= n 0) y "
                  f"(f (+ y (* {_DT!r} (- {g_scm} (* {tau} y)))) (- n 1)))) "
                  f"(f 1.0 {_STEPS})")

        def jf(jnp, I, P, _g=gtree, _tau=tau):
            import jax
            x = I["x"]
            return jax.lax.fori_loop(
                0, _STEPS,
                lambda i, y: y + _DT * (_eval_jax(_g, jnp, x, P) - P[_tau] * y), 1.0)

        return Program(
            pid=pid, kind="recursive", scheme=scheme,
            input_names=["x"], param_names=params, targets=targets,
            bounds={p: _PR for p in params}, input_ranges={"x": _XR},
            ground_truth=gt, jax_forward=jf,
            port_loc=_node_count(gtree) + 4)            # +4 for loop scaffolding
    raise RuntimeError("recursive rejection sampling failed")


def generate_corpus(n: int, recursive_fraction: float, seed: int = 0) -> list[Program]:
    """`n` structurally distinct programs with the requested recursive fraction (deterministic)."""
    g = torch.Generator().manual_seed(seed)
    n_rec = round(recursive_fraction * n)
    progs = []
    for pid in range(n):
        if pid < n_rec:
            progs.append(_make_recursive(g, pid, seed))
        else:
            progs.append(_make_closed_form(g, pid, seed))
    return progs


def gen_data(prog: Program, n_data: int = 16, seed: int = 0):
    g = torch.Generator().manual_seed(seed + 1 + prog.pid)
    cols = {}
    for d in prog.input_names:
        lo, hi = prog.input_ranges[d]
        cols[d] = lo + (hi - lo) * torch.rand(n_data, generator=g)
    ys = torch.tensor([prog.ground_truth(**{d: cols[d][i].item() for d in prog.input_names},
                                         **prog.targets) for i in range(n_data)])
    return cols, ys
