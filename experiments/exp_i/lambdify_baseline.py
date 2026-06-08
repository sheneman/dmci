############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# lambdify_baseline.py: The `lambdify -> jax.grad` baseline (pillar 2 of the pilot). This is the strongest "why not just autodiff?"...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""The `lambdify -> jax.grad` baseline (pillar 2 of the pilot).

This is the strongest "why not just autodiff?" rebuttal target: both sympy and
jax are installed, so an LLM-proposed CLOSED-FORM Scheme expression can be parsed,
lambdified, and differentiated with near-zero per-structure human work — matching
DMCI's capability claim on the closed-form subset.

The honest, measurable point is therefore NOT "JAX can't", it is:
  - one-time cost: DMCI ships a compiled differentiable interpreter (parser + env +
    heap + per-op VJP); the lambdify pipeline needs a one-time sexp->sympy mapper
    (this file, ~40 LOC). Both are amortised; per closed-form structure both are ~0.
  - DMCI additionally follows into RECURSIVE/STATEFUL programs (the carbon pool)
    that sympy.lambdify CANNOT ingest without a human manually unrolling the
    recursion — i.e. nonzero per-structure work. sexp_to_sympy raises on those,
    which is exactly the engineering-cost delta we report.

So the pilot uses this file to show (a) parity on the static GPP curve and
(b) a hard failure on the recursive carbon pool.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

# Reuse the s-expression parser already written for exp_f's arity validator.
from experiments.exp_f.exp_f import _parse_sexp


class NotClosedForm(Exception):
    """Raised when an expression cannot be lambdified (recursion/state/defines)."""


_BINARY = {"+", "-", "*", "/", "pow"}
_UNARY = {"exp", "log", "sin", "cos", "sqrt", "abs"}
_UNSUPPORTED = {"define", "lambda", "let", "letrec", "if", "cond", "begin",
                "cons", "car", "cdr", "list", "quote", "scheme-eval",
                "scheme-eval-program"}


def sexp_to_sympy(tree, symbols: dict):
    import sympy
    if not isinstance(tree, list):
        # atom: number literal or a symbol
        try:
            return sympy.Float(float(tree))
        except (ValueError, TypeError):
            symbols.setdefault(tree, sympy.Symbol(tree))
            return symbols[tree]
    if not tree:
        raise NotClosedForm("empty application")
    head = tree[0]
    if isinstance(head, str) and head in _UNSUPPORTED:
        raise NotClosedForm(
            f"'{head}' is recursive/stateful — not expressible via "
            f"sympy.lambdify without manual unrolling")
    args = [sexp_to_sympy(a, symbols) for a in tree[1:]]
    if head in _BINARY:
        if len(args) != 2:
            raise NotClosedForm(f"'{head}' needs 2 args, got {len(args)}")
        a, b = args
        return {"+": a + b, "-": a - b, "*": a * b, "/": a / b,
                "pow": a ** b}[head]
    if head in _UNARY:
        if len(args) != 1:
            raise NotClosedForm(f"'{head}' needs 1 arg, got {len(args)}")
        return {"exp": sympy.exp, "log": sympy.log, "sin": sympy.sin,
                "cos": sympy.cos, "sqrt": sympy.sqrt, "abs": sympy.Abs}[head](args[0])
    raise NotClosedForm(f"unknown operator '{head}'")


@dataclass
class LambdifyResult:
    expressible: bool
    reason: str
    converged: bool
    best_mse: float
    fitted_values: dict
    n_epochs: int
    t_fit_s: float


def _build_jax_fn(expression: str, driver_names, param_names):
    """sexp string -> jax-callable f(drivers..., params...). Raises NotClosedForm."""
    import sympy
    try:
        tree = _parse_sexp(expression)
    except ValueError as e:
        # multiple top-level forms (e.g. a `define` + a call) -> not a single
        # closed-form expression; lambdify cannot ingest it without a human
        # manually reimplementing the recursion.
        raise NotClosedForm(
            f"not a single closed-form expression ({e}); contains defines / "
            f"multiple forms requiring manual reimplementation")
    symbols: dict = {}
    expr = sexp_to_sympy(tree, symbols)
    ordered = [symbols[n] for n in driver_names + param_names if n in symbols]
    # any free symbol not in the declared schema is an error
    free = {s.name for s in expr.free_symbols}
    declared = set(driver_names) | set(param_names)
    missing = free - declared
    if missing:
        raise NotClosedForm(f"undeclared symbols {missing}")
    f = sympy.lambdify(ordered, expr, modules="jax")
    used = [s.name for s in ordered]
    return f, used


def run_lambdify_jax(model, cfg, seed: int) -> LambdifyResult:
    """Fit the same expression via sympy.lambdify -> jax.grad + Adam.
    For recursive models this returns expressible=False (the engineering-cost delta)."""
    try:
        import jax
        import jax.numpy as jnp
    except Exception as e:  # pragma: no cover
        return LambdifyResult(False, f"jax unavailable: {e}", False,
                              float("inf"), {}, 0, 0.0)

    from .harness import generate_data

    try:
        f, used = _build_jax_fn(model.expression, model.input_names,
                                model.param_names)
    except NotClosedForm as e:
        return LambdifyResult(
            expressible=False,
            reason=(f"lambdify cannot ingest this structure ({e}); a human would "
                    f"have to manually unroll/reimplement it -> nonzero "
                    f"per-structure engineering cost (DMCI handles it as-is)."),
            converged=False, best_mse=float("inf"), fitted_values={},
            n_epochs=0, t_fit_s=0.0)

    xs_list, ys = generate_data(model, cfg, seed)
    driver_arrs = {d: jnp.array([xd[d] for xd in xs_list])
                   for d in model.input_names}
    y_arr = jnp.array([y.item() for y in ys])

    # init params identically to the torch path (same seed / prior)
    import torch
    torch.manual_seed(seed)
    init = []
    for name in model.param_names:
        tgt = model.target_values[name]
        init.append(float(torch.tensor(float(tgt))
                          + 0.3 * max(abs(tgt), 0.1) * torch.randn(1).squeeze()))
    theta0 = jnp.array(init)

    name_to_idx = {n: i for i, n in enumerate(model.param_names)}

    def predict(theta):
        call_args = []
        for n in used:
            if n in driver_arrs:
                call_args.append(driver_arrs[n])
            else:
                call_args.append(theta[name_to_idx[n]])
        return f(*call_args)

    def loss_fn(theta):
        pred = predict(theta)
        return jnp.mean((pred - y_arr) ** 2)

    grad_fn = jax.jit(jax.grad(loss_fn))
    loss_jit = jax.jit(loss_fn)

    # hand-rolled Adam (avoid an optax dependency)
    m = jnp.zeros_like(theta0)
    v = jnp.zeros_like(theta0)
    b1, b2, eps = 0.9, 0.999, 1e-8
    theta = theta0
    best_mse = float("inf")
    best_theta = theta0
    patience = 0
    t0 = time.perf_counter()
    epoch = 0
    for epoch in range(cfg.max_epochs):
        g = grad_fn(theta)
        if not bool(jnp.all(jnp.isfinite(g))):
            patience += 1
        else:
            m = b1 * m + (1 - b1) * g
            v = b2 * v + (1 - b2) * (g * g)
            mhat = m / (1 - b1 ** (epoch + 1))
            vhat = v / (1 - b2 ** (epoch + 1))
            theta = theta - cfg.lr * mhat / (jnp.sqrt(vhat) + eps)
        lv = float(loss_jit(theta))
        if lv < best_mse:
            best_mse, best_theta, patience = lv, theta, 0
        else:
            patience += 1
        if best_mse < cfg.convergence_threshold:
            break
        if patience > cfg.early_stop_patience:
            break
    t_fit = time.perf_counter() - t0

    fitted = {n: float(best_theta[name_to_idx[n]]) for n in model.param_names}
    return LambdifyResult(
        expressible=True, reason="closed-form: lambdify parity with DMCI",
        converged=best_mse < cfg.convergence_threshold, best_mse=best_mse,
        fitted_values=fitted, n_epochs=epoch + 1, t_fit_s=t_fit)
