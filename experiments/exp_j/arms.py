############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# arms.py: The three arms of Exp J and their instrumentation. DMCI — compile_interpreter() ONCE, then evaluate_program per...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""The three arms of Exp J and their instrumentation.

  DMCI  — compile_interpreter() ONCE, then evaluate_program per program (zero per-structure
          compile, zero per-structure code, 100% coverage).
  B1    — lambdify -> jax.jit(jax.grad): automatic, near-zero engineering, but only covers
          closed-form programs (recursive => NotClosedForm coverage failure). The strong baseline.
  B2    — hand-port to JAX: a human transcribes each structure into a jitted JAX function
          (engineering = LOC), recompiled per program. Covers everything, at a per-structure cost.

Per program we record: per-program compile/trace time, coverage (could it be evaluated with no
human intervention), and per-structure engineering LOC. Recovery (matched optimizer, equal
wall-clock) uses a shared scipy L-BFGS-B multi-start driver in log-reparameterized coordinates.
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch

from neural_compiler import compile_interpreter, evaluate_program, save_compiled
from neural_compiler.runtime.tagged_value import make_float, unwrap_number
from experiments.exp_i.lambdify_baseline import _build_jax_fn, NotClosedForm
from .corpus import Program, gen_data


# --- DMCI arm: compile the interpreter ONCE ---------------------------------

def dmci_setup() -> dict:
    t0 = time.perf_counter()
    interp = compile_interpreter()
    t_compile = time.perf_counter() - t0
    import tempfile, os
    p = tempfile.mktemp(suffix=".ncg")
    save_compiled(interp, p)
    size = os.path.getsize(p)
    os.remove(p)
    return {"interp": interp, "one_time_compile_s": t_compile, "ncg_bytes": size}


def dmci_prepare(prog: Program, interp) -> dict:
    """No per-program compile; verify the program evaluates (coverage)."""
    t0 = time.perf_counter()
    covered = True
    try:
        cols, _ = gen_data(prog, n_data=2)
        binds = {d: float(cols[d][0]) for d in prog.input_names}
        binds.update(prog.targets)
        v = unwrap_number(evaluate_program(interp, prog.scheme, binds))
        covered = bool(torch.isfinite(v).all())
    except Exception:
        covered = False
    return {"compile_s": 0.0, "covered": covered, "eng_loc": 0,
            "prepare_s": time.perf_counter() - t0}


# --- B1 arm: lambdify -> jax.jit(jax.grad) ----------------------------------

def b1_prepare(prog: Program) -> dict:
    """Automatic translation; covers closed-form only. Times the jit/trace of grad."""
    try:
        f, used = _build_jax_fn(prog.scheme, prog.input_names, prog.param_names)
    except NotClosedForm:
        return {"compile_s": 0.0, "covered": False, "eng_loc": 0}   # coverage failure
    try:
        import jax
        import jax.numpy as jnp
    except Exception:
        return {"compile_s": 0.0, "covered": False, "eng_loc": 0}
    cols, ys = gen_data(prog)
    darr = {d: jnp.asarray(cols[d].numpy()) for d in prog.input_names}
    yj = jnp.asarray(ys.numpy())
    idx = {n: i for i, n in enumerate(prog.param_names)}

    def loss(theta):
        args = [darr[n] if n in darr else theta[idx[n]] for n in used]
        return jnp.mean((f(*args) - yj) ** 2)
    vg = jax.jit(jax.value_and_grad(loss))
    theta0 = jnp.array([prog.targets[n] for n in prog.param_names])
    t0 = time.perf_counter()
    v, _gd = vg(theta0)            # first call: trace + compile
    float(v)                       # block on result
    t_compile = time.perf_counter() - t0
    return {"compile_s": t_compile, "covered": True, "eng_loc": 0}


# --- B2 arm: hand-port to JAX -----------------------------------------------

def b2_prepare(prog: Program) -> dict:
    """Hand-ported JAX forward; covers everything at a per-structure engineering cost
    (LOC), recompiled per program. Times the jit/trace of grad."""
    if prog.jax_forward is None:
        # LLM recursive subset: a human CAN port it (coverage holds by construction) and we
        # cost the port (LOC), but we do not auto-emit a JAX forward — so its compile/recovery
        # are not measured here (DMCI provides recovery; B1 fails coverage).
        return {"compile_s": 0.0, "covered": True, "eng_loc": prog.port_loc}
    try:
        import jax
        import jax.numpy as jnp
    except Exception:
        return {"compile_s": 0.0, "covered": False, "eng_loc": prog.port_loc}
    cols, ys = gen_data(prog)
    darr = {d: jnp.asarray(cols[d].numpy()) for d in prog.input_names}
    yj = jnp.asarray(ys.numpy())

    def loss(theta):
        P = {n: theta[i] for i, n in enumerate(prog.param_names)}
        # vmap the hand-ported forward over the data points
        pred = jax.vmap(lambda *vals: prog.jax_forward(
            jnp, dict(zip(prog.input_names, vals)), P))(*[darr[d] for d in prog.input_names])
        return jnp.mean((pred - yj) ** 2)
    vg = jax.jit(jax.value_and_grad(loss))
    theta0 = jnp.array([prog.targets[n] for n in prog.param_names])
    t0 = time.perf_counter()
    v, _gd = vg(theta0)
    float(v)
    t_compile = time.perf_counter() - t0
    # engineering: a human writes the port once per *structure*; every program here is a
    # distinct structure, so the full LOC is paid each time.
    return {"compile_s": t_compile, "covered": True, "eng_loc": prog.port_loc}


# --- Matched recovery: shared scipy L-BFGS-B multi-start (log-reparam) -------

def _scipy_multistart(loss_grad_log, n_params, n_starts, time_budget, seed):
    """loss_grad_log(log_theta_np) -> (loss, grad_np). Returns (best_loss, best_log_theta)."""
    from scipy.optimize import minimize
    rng = np.random.default_rng(seed)
    lo, hi = math.log(0.2), math.log(1.5)
    best_loss, best = float("inf"), None
    t0 = time.perf_counter()
    s = 0
    while s < n_starts and (time.perf_counter() - t0) < time_budget:
        x0 = rng.uniform(lo, hi, size=n_params)
        try:
            r = minimize(loss_grad_log, x0, jac=True, method="L-BFGS-B",
                         options={"maxiter": 200})
            if r.fun < best_loss:
                best_loss, best = float(r.fun), r.x
        except Exception:
            pass
        s += 1
    return best_loss, best


def recover(arm: str, prog: Program, interp=None, n_starts=16, time_budget=20.0, seed=0):
    """Matched recovery on one program. Returns (best_mse, mean_param_rel_error)."""
    cols, ys = gen_data(prog)

    if arm == "dmci":
        y_t = ys
        names = prog.param_names

        def lg(log_theta):
            raw = torch.tensor(log_theta, dtype=torch.float32, requires_grad=True)
            binds = {d: make_float(cols[d]) for d in prog.input_names}
            for i, n in enumerate(names):
                binds[n] = make_float(torch.exp(raw[i]))
            pred = unwrap_number(evaluate_program(interp, prog.scheme, binds))
            loss = ((pred - y_t) ** 2).mean()
            g = torch.autograd.grad(loss, raw)[0]
            return float(loss), g.detach().numpy().astype(np.float64)
    else:  # b1 / b2 — build a jax loss
        import jax
        import jax.numpy as jnp
        darr = {d: jnp.asarray(cols[d].numpy()) for d in prog.input_names}
        yj = jnp.asarray(ys.numpy())
        names = prog.param_names
        if arm == "b1":
            f, used = _build_jax_fn(prog.scheme, prog.input_names, prog.param_names)
            idx = {n: i for i, n in enumerate(names)}

            def lossfn(log_theta):
                theta = jnp.exp(log_theta)
                args = [darr[n] if n in darr else theta[idx[n]] for n in used]
                return jnp.mean((f(*args) - yj) ** 2)
        else:
            def lossfn(log_theta):
                theta = jnp.exp(log_theta)
                P = {n: theta[i] for i, n in enumerate(names)}
                pred = jax.vmap(lambda *vals: prog.jax_forward(
                    jnp, dict(zip(prog.input_names, vals)), P))(
                        *[darr[d] for d in prog.input_names])
                return jnp.mean((pred - yj) ** 2)
        vg = jax.jit(jax.value_and_grad(lossfn))

        def lg(log_theta):
            v, g = vg(jnp.asarray(log_theta))
            return float(v), np.asarray(g, dtype=np.float64)

    best_loss, best = _scipy_multistart(lg, len(prog.param_names), n_starts, time_budget, seed)
    if best is None:
        return float("inf"), float("inf")
    fitted = {n: float(math.exp(best[i])) for i, n in enumerate(prog.param_names)}
    rel = sum(abs(fitted[n] - prog.targets[n]) / (abs(prog.targets[n]) or 1.0)
              for n in prog.param_names) / len(prog.param_names)
    return best_loss, rel
