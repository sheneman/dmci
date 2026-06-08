############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# baselines.py: Five methods for learning program constants via optimization. 1. direct — compile target program directly,...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Five methods for learning program constants via optimization.

1. direct          — compile target program directly, autograd
2. compiled_interp — compile self-hosted evaluator, pass program as data, autograd
3. handcoded_interp— pure Python/PyTorch tree-walking evaluator, autograd
4. finite_diff     — compiled interpreter, central finite differences
5. evolution_strategy — compiled interpreter, gradient-free (mu,lambda)-ES
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from neural_compiler.compiler import compile_program
from neural_compiler.evaluator import evaluate
from neural_compiler.runtime.tagged_value import make_float, unwrap_number

from .config import ExpAConfig, DEFAULT
from .programs import ProgramSpec, _all_input_names


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class TrainResult:
    method: str
    program: str
    seed: int
    converged: bool
    convergence_epoch: int | None
    final_loss: float
    final_param_errors: dict[str, float]
    loss_history: list[float]
    param_history: dict[str, list[float]]
    grad_norm_history: list[float]
    wall_time_history: list[float]
    total_wall_time: float


# ---------------------------------------------------------------------------
# Graph cache (compile once, reuse across seeds)
# ---------------------------------------------------------------------------

_GRAPH_CACHE: dict[str, object] = {}


def _get_graph(key: str, source: str, input_names: list[str]):
    if key not in _GRAPH_CACHE:
        inputs = {n: None for n in input_names}
        _GRAPH_CACHE[key] = compile_program(source, inputs=inputs, prelude=True)
    return _GRAPH_CACHE[key]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_params(spec: ProgramSpec, seed: int) -> dict[str, nn.Parameter]:
    torch.manual_seed(seed)
    return {
        name: nn.Parameter(
            torch.tensor(spec.init_values[name])
            + 0.3 * max(abs(spec.init_values[name]), 0.1) * torch.randn(1).squeeze()
        )
        for name in spec.param_names
    }


def _build_tagged_inputs(
    spec: ProgramSpec,
    params: dict[str, nn.Parameter],
    x_val: torch.Tensor,
) -> dict[str, torch.Tensor]:
    inputs = {"x": make_float(x_val)}
    for name, param in params.items():
        inputs[name] = make_float(param)
    return inputs


def _build_direct_inputs(
    spec: ProgramSpec,
    params: dict[str, nn.Parameter],
    x_val: torch.Tensor,
) -> dict[str, torch.Tensor]:
    inputs = {"x": make_float(x_val)}
    for name, param in params.items():
        inputs[name] = make_float(param)
    # direct compilation may also use tagged values for consistency
    return inputs


def _generate_data(spec: ProgramSpec, cfg: ExpAConfig):
    xs = torch.linspace(*cfg.x_range, cfg.n_data_points)
    ys = torch.tensor([spec.data_fn(x.item()) for x in xs])
    return xs, ys


def _compute_loss(graph, spec, params, xs, ys, input_builder):
    total = torch.tensor(0.0)
    for x_val, y_val in zip(xs, ys):
        inputs = input_builder(spec, params, x_val)
        result = evaluate(graph, inputs)
        pred = unwrap_number(result)
        total = total + (pred - y_val) ** 2
    return total


def _grad_norm(params: dict[str, nn.Parameter]) -> float:
    grads = [p.grad.flatten() for p in params.values()
             if p.grad is not None]
    if not grads:
        return 0.0
    return torch.cat(grads).norm().item()


def _param_errors(params, spec):
    return {
        name: abs(params[name].item() - spec.target_values[name])
        for name in spec.param_names
    }


def _record(epoch, loss_val, params, spec, grad_norm_val, wall_time,
            result_lists):
    result_lists["loss"].append(loss_val)
    result_lists["grad_norm"].append(grad_norm_val)
    result_lists["wall_time"].append(wall_time)
    for name in spec.param_names:
        result_lists["params"][name].append(params[name].item())


# ---------------------------------------------------------------------------
# Method 1: Direct compilation
# ---------------------------------------------------------------------------

def run_direct(spec: ProgramSpec, cfg: ExpAConfig, seed: int) -> TrainResult:
    graph = _get_graph(
        f"direct_{spec.name}",
        spec.direct_source,
        _all_input_names(spec),
    )
    params = _make_params(spec, seed)
    optimizer = torch.optim.Adam(list(params.values()), lr=cfg.lr)
    xs, ys = _generate_data(spec, cfg)

    lists = {"loss": [], "grad_norm": [], "wall_time": [],
             "params": {n: [] for n in spec.param_names}}
    conv_epoch = None
    t_start = time.perf_counter()

    for epoch in range(cfg.max_epochs):
        t0 = time.perf_counter()
        loss = _compute_loss(graph, spec, params, xs, ys, _build_direct_inputs)
        optimizer.zero_grad()
        loss.backward()
        gn = _grad_norm(params)
        optimizer.step()
        wt = time.perf_counter() - t0

        loss_val = loss.item()
        _record(epoch, loss_val, params, spec, gn, wt, lists)
        if conv_epoch is None and loss_val < cfg.convergence_threshold:
            conv_epoch = epoch
        elif conv_epoch is not None and epoch >= conv_epoch + 50:
            break

    total_wt = time.perf_counter() - t_start
    return TrainResult(
        method="direct", program=spec.name, seed=seed,
        converged=conv_epoch is not None, convergence_epoch=conv_epoch,
        final_loss=lists["loss"][-1],
        final_param_errors=_param_errors(params, spec),
        loss_history=lists["loss"], param_history=lists["params"],
        grad_norm_history=lists["grad_norm"],
        wall_time_history=lists["wall_time"], total_wall_time=total_wt,
    )


# ---------------------------------------------------------------------------
# Method 2: Compiled interpreter
# ---------------------------------------------------------------------------

def run_compiled_interp(spec: ProgramSpec, cfg: ExpAConfig, seed: int) -> TrainResult:
    graph = _get_graph(
        f"compiled_interp_{spec.name}",
        spec.interp_source,
        _all_input_names(spec),
    )
    params = _make_params(spec, seed)
    optimizer = torch.optim.Adam(list(params.values()), lr=cfg.lr)
    xs, ys = _generate_data(spec, cfg)

    lists = {"loss": [], "grad_norm": [], "wall_time": [],
             "params": {n: [] for n in spec.param_names}}
    conv_epoch = None
    t_start = time.perf_counter()

    for epoch in range(cfg.max_epochs):
        t0 = time.perf_counter()
        loss = _compute_loss(graph, spec, params, xs, ys, _build_tagged_inputs)
        optimizer.zero_grad()
        loss.backward()
        gn = _grad_norm(params)
        optimizer.step()
        wt = time.perf_counter() - t0

        loss_val = loss.item()
        _record(epoch, loss_val, params, spec, gn, wt, lists)
        if conv_epoch is None and loss_val < cfg.convergence_threshold:
            conv_epoch = epoch
        elif conv_epoch is not None and epoch >= conv_epoch + 50:
            break

    total_wt = time.perf_counter() - t_start
    return TrainResult(
        method="compiled_interp", program=spec.name, seed=seed,
        converged=conv_epoch is not None, convergence_epoch=conv_epoch,
        final_loss=lists["loss"][-1],
        final_param_errors=_param_errors(params, spec),
        loss_history=lists["loss"], param_history=lists["params"],
        grad_norm_history=lists["grad_norm"],
        wall_time_history=lists["wall_time"], total_wall_time=total_wt,
    )


# ---------------------------------------------------------------------------
# Method 3: Hand-coded PyTorch interpreter
# ---------------------------------------------------------------------------

class HandCodedInterpreter:
    """Tree-walking Scheme evaluator in pure Python/PyTorch.

    Values are torch.Tensor (numbers) or Python tuples/lists (closures, pairs).
    No tagged values, no heap — autograd flows through PyTorch arithmetic only.
    Semantics match bootstrap/compiler.scm.
    """

    def eval_expr(self, expr, env: dict[str, object]) -> torch.Tensor:
        if isinstance(expr, (int, float)):
            return torch.tensor(float(expr))
        if isinstance(expr, str):
            return env[expr]
        if not isinstance(expr, list) or len(expr) == 0:
            return torch.tensor(0.0)

        head = expr[0]

        if head == "quote":
            return expr[1]

        if head == "if":
            test = self.eval_expr(expr[1], env)
            test_val = test.item() if isinstance(test, torch.Tensor) else test
            if test_val != 0.0 and test_val is not False:
                return self.eval_expr(expr[2], env)
            return self.eval_expr(expr[3], env)

        if head == "lambda":
            params, body = expr[1], expr[2]
            captured = dict(env)
            return ("closure", params, body, captured)

        if head == "let":
            bindings, body = expr[1], expr[2]
            new_env = dict(env)
            for binding in bindings:
                name, val_expr = binding[0], binding[1]
                new_env[name] = self.eval_expr(val_expr, new_env)
            return self.eval_expr(body, new_env)

        # Primitives
        if head in ("+", "-", "*", "/"):
            a = self.eval_expr(expr[1], env)
            b = self.eval_expr(expr[2], env)
            if head == "+":
                return a + b
            if head == "-":
                return a - b
            if head == "*":
                return a * b
            return a / b

        if head in ("=", "<", ">", "<=", ">="):
            a = self.eval_expr(expr[1], env)
            b = self.eval_expr(expr[2], env)
            a_v = a.item() if isinstance(a, torch.Tensor) else float(a)
            b_v = b.item() if isinstance(b, torch.Tensor) else float(b)
            if head == "=":
                return torch.tensor(1.0) if a_v == b_v else torch.tensor(0.0)
            if head == "<":
                return torch.tensor(1.0) if a_v < b_v else torch.tensor(0.0)
            if head == ">":
                return torch.tensor(1.0) if a_v > b_v else torch.tensor(0.0)
            if head == "<=":
                return torch.tensor(1.0) if a_v <= b_v else torch.tensor(0.0)
            return torch.tensor(1.0) if a_v >= b_v else torch.tensor(0.0)

        # Function application
        func = self.eval_expr(head, env) if isinstance(head, (list, str)) else head
        args = [self.eval_expr(a, env) for a in expr[1:]]
        return self._apply(func, args)

    def _apply(self, func, args):
        if isinstance(func, tuple) and func[0] == "closure":
            _, param_names, body, captured_env = func
            new_env = dict(captured_env)
            for p, a in zip(param_names, args):
                new_env[p] = a
            return self.eval_expr(body, new_env)
        if isinstance(func, tuple) and func[0] == "defined_fn":
            _, param_names, body, def_env = func
            new_env = dict(def_env)
            for p, a in zip(param_names, args):
                new_env[p] = a
            return self.eval_expr(body, new_env)
        return torch.tensor(0.0)

    def eval_program(self, program_def: dict, env: dict[str, object]) -> torch.Tensor:
        prog_env = dict(env)
        for name, param_names, body in program_def["defines"]:
            prog_env[name] = ("defined_fn", param_names, body, prog_env)
        return self.eval_expr(program_def["body"], prog_env)


_HC_INTERP = HandCodedInterpreter()


def _hc_eval(spec: ProgramSpec, params: dict[str, nn.Parameter],
             x_val: torch.Tensor) -> torch.Tensor:
    env: dict[str, object] = {"x": x_val}
    for name, param in params.items():
        env[name] = param
    if spec.hc_program is not None:
        return _HC_INTERP.eval_program(spec.hc_program, env)
    return _HC_INTERP.eval_expr(spec.hc_expr, env)


def run_handcoded_interp(spec: ProgramSpec, cfg: ExpAConfig, seed: int) -> TrainResult:
    if spec.hc_expr is None and spec.hc_program is None:
        raise ValueError(f"No hand-coded AST for {spec.name}")

    params = _make_params(spec, seed)
    optimizer = torch.optim.Adam(list(params.values()), lr=cfg.lr)
    xs, ys = _generate_data(spec, cfg)

    lists = {"loss": [], "grad_norm": [], "wall_time": [],
             "params": {n: [] for n in spec.param_names}}
    conv_epoch = None
    t_start = time.perf_counter()

    for epoch in range(cfg.max_epochs):
        t0 = time.perf_counter()
        total_loss = torch.tensor(0.0)
        for x_val, y_val in zip(xs, ys):
            pred = _hc_eval(spec, params, x_val)
            total_loss = total_loss + (pred - y_val) ** 2

        optimizer.zero_grad()
        total_loss.backward()
        gn = _grad_norm(params)
        optimizer.step()
        wt = time.perf_counter() - t0

        loss_val = total_loss.item()
        _record(epoch, loss_val, params, spec, gn, wt, lists)
        if conv_epoch is None and loss_val < cfg.convergence_threshold:
            conv_epoch = epoch
        elif conv_epoch is not None and epoch >= conv_epoch + 50:
            break

    total_wt = time.perf_counter() - t_start
    return TrainResult(
        method="handcoded_interp", program=spec.name, seed=seed,
        converged=conv_epoch is not None, convergence_epoch=conv_epoch,
        final_loss=lists["loss"][-1],
        final_param_errors=_param_errors(params, spec),
        loss_history=lists["loss"], param_history=lists["params"],
        grad_norm_history=lists["grad_norm"],
        wall_time_history=lists["wall_time"], total_wall_time=total_wt,
    )


# ---------------------------------------------------------------------------
# Method 4: Finite differences
# ---------------------------------------------------------------------------

def _fd_loss_no_grad(graph, spec, params, xs, ys):
    with torch.no_grad():
        total = 0.0
        for x_val, y_val in zip(xs, ys):
            inputs = _build_tagged_inputs(spec, params, x_val)
            result = evaluate(graph, inputs)
            pred = unwrap_number(result)
            total += (pred.item() - y_val.item()) ** 2
    return total


def run_finite_diff(spec: ProgramSpec, cfg: ExpAConfig, seed: int) -> TrainResult:
    graph = _get_graph(
        f"compiled_interp_{spec.name}",  # reuse same graph
        spec.interp_source,
        _all_input_names(spec),
    )
    params = _make_params(spec, seed)
    xs, ys = _generate_data(spec, cfg)

    lists = {"loss": [], "grad_norm": [], "wall_time": [],
             "params": {n: [] for n in spec.param_names}}
    conv_epoch = None
    t_start = time.perf_counter()

    for epoch in range(cfg.max_epochs):
        t0 = time.perf_counter()

        # Central finite differences
        grads = {}
        for name, param in params.items():
            orig = param.item()
            param.data.fill_(orig + cfg.fd_epsilon)
            loss_plus = _fd_loss_no_grad(graph, spec, params, xs, ys)
            param.data.fill_(orig - cfg.fd_epsilon)
            loss_minus = _fd_loss_no_grad(graph, spec, params, xs, ys)
            param.data.fill_(orig)
            grads[name] = (loss_plus - loss_minus) / (2 * cfg.fd_epsilon)

        # SGD update
        for name, param in params.items():
            param.data -= cfg.fd_lr * grads[name]

        loss_val = _fd_loss_no_grad(graph, spec, params, xs, ys)
        gn = (sum(g ** 2 for g in grads.values())) ** 0.5
        wt = time.perf_counter() - t0

        _record(epoch, loss_val, params, spec, gn, wt, lists)
        if conv_epoch is None and loss_val < cfg.convergence_threshold:
            conv_epoch = epoch
        elif conv_epoch is not None and epoch >= conv_epoch + 50:
            break

    total_wt = time.perf_counter() - t_start
    return TrainResult(
        method="finite_diff", program=spec.name, seed=seed,
        converged=conv_epoch is not None, convergence_epoch=conv_epoch,
        final_loss=lists["loss"][-1],
        final_param_errors=_param_errors(params, spec),
        loss_history=lists["loss"], param_history=lists["params"],
        grad_norm_history=lists["grad_norm"],
        wall_time_history=lists["wall_time"], total_wall_time=total_wt,
    )


# ---------------------------------------------------------------------------
# Method 5: Evolution strategy (self-contained, no dependencies)
# ---------------------------------------------------------------------------

class SimpleES:
    """(mu, lambda) evolution strategy with sigma adaptation."""

    def __init__(self, x0: np.ndarray, sigma0: float, seed: int,
                 pop_size: int = 20):
        self.rng = np.random.RandomState(seed)
        self.mean = np.array(x0, dtype=np.float64)
        self.sigma = sigma0
        self.dim = len(x0)
        self.pop_size = pop_size
        self.mu = max(1, pop_size // 2)
        self.generation = 0
        self._n_better = 0

    def ask(self) -> list[np.ndarray]:
        return [self.mean + self.sigma * self.rng.randn(self.dim)
                for _ in range(self.pop_size)]

    def tell(self, solutions: list[np.ndarray], fitnesses: list[float]):
        idx = np.argsort(fitnesses)[:self.mu]
        new_mean = np.mean([solutions[i] for i in idx], axis=0)
        # 1/5 success rule for sigma adaptation
        best_fitness = fitnesses[idx[0]]
        if hasattr(self, "_prev_best") and best_fitness < self._prev_best:
            self._n_better += 1
        self._prev_best = best_fitness
        self.generation += 1
        if self.generation % 10 == 0 and self.generation > 0:
            ratio = self._n_better / 10
            if ratio > 0.2:
                self.sigma *= 1.2
            elif ratio < 0.2:
                self.sigma *= 0.83
            self._n_better = 0
        self.mean = new_mean

    @property
    def best(self):
        return self.mean


def run_evolution_strategy(spec: ProgramSpec, cfg: ExpAConfig, seed: int) -> TrainResult:
    graph = _get_graph(
        f"compiled_interp_{spec.name}",
        spec.interp_source,
        _all_input_names(spec),
    )
    xs, ys = _generate_data(spec, cfg)

    x0 = np.array([spec.init_values[n] for n in spec.param_names])
    es = SimpleES(x0, cfg.es_sigma0, seed, cfg.es_pop_size)

    lists = {"loss": [], "grad_norm": [], "wall_time": [],
             "params": {n: [] for n in spec.param_names}}
    conv_epoch = None
    t_start = time.perf_counter()

    fevals = 0
    # Map ES generations to "epochs" for comparable tracking
    max_gens = cfg.es_max_fevals // cfg.es_pop_size

    # Dummy params dict for helpers
    params = {name: nn.Parameter(torch.tensor(spec.init_values[name]))
              for name in spec.param_names}

    for gen in range(min(max_gens, cfg.max_epochs)):
        t0 = time.perf_counter()
        solutions = es.ask()
        fitnesses = []
        for sol in solutions:
            for i, name in enumerate(spec.param_names):
                params[name].data.fill_(sol[i])
            loss = _fd_loss_no_grad(graph, spec, params, xs, ys)
            fitnesses.append(loss)
        es.tell(solutions, fitnesses)
        fevals += len(solutions)

        # Set params to current ES mean for recording
        for i, name in enumerate(spec.param_names):
            params[name].data.fill_(es.mean[i])
        loss_val = _fd_loss_no_grad(graph, spec, params, xs, ys)
        wt = time.perf_counter() - t0

        _record(gen, loss_val, params, spec, float("nan"), wt, lists)
        if conv_epoch is None and loss_val < cfg.convergence_threshold:
            conv_epoch = gen
        elif conv_epoch is not None and gen >= conv_epoch + 50:
            break

    total_wt = time.perf_counter() - t_start
    return TrainResult(
        method="evolution_strategy", program=spec.name, seed=seed,
        converged=conv_epoch is not None, convergence_epoch=conv_epoch,
        final_loss=lists["loss"][-1],
        final_param_errors=_param_errors(params, spec),
        loss_history=lists["loss"], param_history=lists["params"],
        grad_norm_history=lists["grad_norm"],
        wall_time_history=lists["wall_time"], total_wall_time=total_wt,
    )


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

METHOD_RUNNERS = {
    "direct": run_direct,
    "compiled_interp": run_compiled_interp,
    "handcoded_interp": run_handcoded_interp,
    "finite_diff": run_finite_diff,
    "evolution_strategy": run_evolution_strategy,
}


def run_method(method: str, spec: ProgramSpec, cfg: ExpAConfig,
               seed: int) -> TrainResult:
    return METHOD_RUNNERS[method](spec, cfg, seed)
