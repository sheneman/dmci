############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# baselines.py: Four methods for Experiment C. 1. dmci — program as data through compiled self-hosted interpreter, autograd 2....
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Four methods for Experiment C.

1. dmci              — program as data through compiled self-hosted interpreter, autograd
2. direct_compiled   — same recursive program compiled directly, autograd
3. handcoded_pytorch — hand-coded PyTorch implementation, autograd
4. pure_mlp          — MLP learns the entire function from data, no structure

Key comparisons:
- dmci vs direct_compiled: interpretation overhead with identical gradient quality
- dmci vs handcoded_pytorch: automated compilation vs manual coding
- All structured methods vs pure_mlp: recursive inductive bias vs flat approximation
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn as nn

from neural_compiler.compiler import compile_program
from neural_compiler.evaluator import evaluate
from neural_compiler.runtime.tagged_value import make_float, unwrap_number

from .config import ExpCConfig, DEFAULT
from .models import ModelSpec, _all_input_names


@dataclass
class TrainResult:
    method: str
    model_name: str
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


_GRAPH_CACHE: dict[str, object] = {}


def _generate_data(model: ModelSpec, cfg: ExpCConfig):
    n_inputs = len(model.input_names)

    if n_inputs == 1:
        xs_raw = torch.linspace(model.x_range[0], model.x_range[1],
                                cfg.n_data_points)
        xs_list = [{model.input_names[0]: x.item()} for x in xs_raw]
    elif n_inputs == 2:
        n_per = max(4, int(cfg.n_data_points ** 0.5))
        v0 = torch.linspace(model.x_range[0], model.x_range[1], n_per)
        v1 = torch.linspace(model.x_range[0], model.x_range[1], n_per)
        xs_list = []
        for a in v0:
            for b in v1:
                xs_list.append({model.input_names[0]: a.item(),
                                model.input_names[1]: b.item()})
    else:
        raise ValueError(f"Unsupported input count: {n_inputs}")

    ys = []
    for xd in xs_list:
        y = model.ground_truth(**xd, **model.target_values)
        ys.append(torch.tensor(float(y)))

    return xs_list, ys


def _make_params(model: ModelSpec, seed: int) -> dict[str, nn.Parameter]:
    torch.manual_seed(seed)
    return {
        name: nn.Parameter(
            torch.tensor(model.init_values[name])
            + 0.3 * max(abs(model.init_values[name]), 0.1) * torch.randn(1).squeeze()
        )
        for name in model.param_names
    }


def _param_errors(params, model):
    return {
        name: abs(params[name].item() - model.target_values[name])
        for name in model.param_names
    }


def _grad_norm(params: dict[str, nn.Parameter]) -> float:
    grads = [p.grad.flatten() for p in params.values() if p.grad is not None]
    if not grads:
        return 0.0
    return torch.cat(grads).norm().item()


def _train_compiled(method_name, graph, model, cfg, seed):
    params = _make_params(model, seed)
    optimizer = torch.optim.Adam(list(params.values()), lr=cfg.lr)
    xs_list, ys = _generate_data(model, cfg)

    lists = {"loss": [], "grad_norm": [], "wall_time": [],
             "params": {n: [] for n in model.param_names}}
    conv_epoch = None
    t_start = time.perf_counter()

    for epoch in range(cfg.max_epochs):
        t0 = time.perf_counter()

        total_loss = torch.tensor(0.0)
        for xd, y_val in zip(xs_list, ys):
            tagged = {}
            for n, v in xd.items():
                tagged[n] = make_float(torch.tensor(v))
            for n, p in params.items():
                tagged[n] = make_float(p)
            result = evaluate(graph, tagged)
            pred = unwrap_number(result)
            total_loss = total_loss + (pred - y_val) ** 2

        optimizer.zero_grad()
        total_loss.backward()
        gn = _grad_norm(params)
        optimizer.step()
        wt = time.perf_counter() - t0

        loss_val = total_loss.item()
        lists["loss"].append(loss_val)
        lists["grad_norm"].append(gn)
        lists["wall_time"].append(wt)
        for name in model.param_names:
            lists["params"][name].append(params[name].item())

        if conv_epoch is None and loss_val < cfg.convergence_threshold:
            conv_epoch = epoch
        elif conv_epoch is not None and epoch >= conv_epoch + 50:
            break

    total_wt = time.perf_counter() - t_start
    return TrainResult(
        method=method_name, model_name=model.name, seed=seed,
        converged=conv_epoch is not None, convergence_epoch=conv_epoch,
        final_loss=lists["loss"][-1],
        final_param_errors=_param_errors(params, model),
        loss_history=lists["loss"], param_history=lists["params"],
        grad_norm_history=lists["grad_norm"],
        wall_time_history=lists["wall_time"], total_wall_time=total_wt,
    )


# ---------------------------------------------------------------------------
# Method 1: DMCI — program as data through compiled interpreter
# ---------------------------------------------------------------------------

def run_dmci(model: ModelSpec, cfg: ExpCConfig, seed: int) -> TrainResult:
    cache_key = f"dmci_{model.name}"
    if cache_key not in _GRAPH_CACHE:
        all_inputs = {n: None for n in _all_input_names(model)}
        _GRAPH_CACHE[cache_key] = compile_program(
            model.interp_source, inputs=all_inputs, prelude=True)
    graph = _GRAPH_CACHE[cache_key]
    return _train_compiled("dmci", graph, model, cfg, seed)


# ---------------------------------------------------------------------------
# Method 2: Direct compilation (no interpreter)
# ---------------------------------------------------------------------------

def run_direct_compiled(model: ModelSpec, cfg: ExpCConfig, seed: int) -> TrainResult:
    cache_key = f"direct_{model.name}"
    if cache_key not in _GRAPH_CACHE:
        all_inputs = {n: None for n in _all_input_names(model)}
        _GRAPH_CACHE[cache_key] = compile_program(
            model.direct_source, inputs=all_inputs, prelude=True)
    graph = _GRAPH_CACHE[cache_key]
    return _train_compiled("direct_compiled", graph, model, cfg, seed)


# ---------------------------------------------------------------------------
# Method 3: Hand-coded PyTorch
# ---------------------------------------------------------------------------

def _build_handcoded_fn(model: ModelSpec):
    def fn(inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        v = {**inputs}

        if model.name == "C01_lotka_volterra":
            x = v["x0"]
            y = torch.tensor(5.0)
            for _ in range(20):
                dx = 0.05 * (v["a"] * x - v["b"] * x * y)
                dy = 0.05 * (0.02 * x * y - 0.5 * y)
                x, y = x + dx, y + dy
            return x

        elif model.name == "C02_sir_epidemic":
            s, i = v["S0"], torch.tensor(0.01)
            for _ in range(30):
                ds = 0.1 * (-v["beta"] * s * i)
                di = 0.1 * (v["beta"] * s * i - v["gamma"] * i)
                s, i = s + ds, i + di
            return s

        elif model.name == "C03_decay_chain":
            a_val = v["A0"]
            b_val = torch.tensor(0.0)
            for _ in range(25):
                da = 0.1 * (-v["lam1"] * a_val)
                db = 0.1 * (v["lam1"] * a_val - v["lam2"] * b_val)
                a_val, b_val = a_val + da, b_val + db
            return b_val

        elif model.name == "C04_logistic_map":
            x = v["x0"]
            for _ in range(10):
                x = v["r"] * x * (1.0 - x)
            return x

        elif model.name == "C05_continued_fraction":
            result = torch.tensor(0.0)
            for _ in range(8):
                result = v["a"] * v["x"] / (1.0 + result)
            return result

        elif model.name == "C06_damped_pendulum":
            theta = v["theta0"]
            omega = torch.tensor(0.0)
            for _ in range(20):
                dtheta = 0.05 * omega
                domega = 0.05 * (-(v["gL"]) * torch.sin(theta)
                                 - v["b"] * omega)
                theta, omega = theta + dtheta, omega + domega
            return theta

        elif model.name == "C07_iir_filter":
            y_prev1 = torch.tensor(0.0)
            y_prev2 = torch.tensor(0.0)
            for _ in range(12):
                y = (v["a1"] * y_prev1 + v["a2"] * y_prev2
                     + v["b0"] * v["x"])
                y_prev2, y_prev1 = y_prev1, y
            return y_prev1

        elif model.name == "C08_cascaded_ema":
            y1 = torch.tensor(0.0)
            y2 = torch.tensor(0.0)
            for _ in range(10):
                ny1 = v["alpha"] * v["x"] + (1.0 - v["alpha"]) * y1
                y2 = v["beta"] * ny1 + (1.0 - v["beta"]) * y2
                y1 = ny1
            return y2

        else:
            raise ValueError(f"No hand-coded implementation for {model.name}")

    return fn


def run_handcoded_pytorch(model: ModelSpec, cfg: ExpCConfig, seed: int) -> TrainResult:
    fn = _build_handcoded_fn(model)
    params = _make_params(model, seed)
    optimizer = torch.optim.Adam(list(params.values()), lr=cfg.lr)
    xs_list, ys = _generate_data(model, cfg)

    lists = {"loss": [], "grad_norm": [], "wall_time": [],
             "params": {n: [] for n in model.param_names}}
    conv_epoch = None
    t_start = time.perf_counter()

    for epoch in range(cfg.max_epochs):
        t0 = time.perf_counter()

        total_loss = torch.tensor(0.0)
        for xd, y_val in zip(xs_list, ys):
            inputs = {}
            for n, val in xd.items():
                inputs[n] = torch.tensor(val)
            for n, p in params.items():
                inputs[n] = p
            pred = fn(inputs)
            total_loss = total_loss + (pred - y_val) ** 2

        optimizer.zero_grad()
        total_loss.backward()
        gn = _grad_norm(params)
        optimizer.step()
        wt = time.perf_counter() - t0

        loss_val = total_loss.item()
        lists["loss"].append(loss_val)
        lists["grad_norm"].append(gn)
        lists["wall_time"].append(wt)
        for name in model.param_names:
            lists["params"][name].append(params[name].item())

        if conv_epoch is None and loss_val < cfg.convergence_threshold:
            conv_epoch = epoch
        elif conv_epoch is not None and epoch >= conv_epoch + 50:
            break

    total_wt = time.perf_counter() - t_start
    return TrainResult(
        method="handcoded_pytorch", model_name=model.name, seed=seed,
        converged=conv_epoch is not None, convergence_epoch=conv_epoch,
        final_loss=lists["loss"][-1],
        final_param_errors=_param_errors(params, model),
        loss_history=lists["loss"], param_history=lists["params"],
        grad_norm_history=lists["grad_norm"],
        wall_time_history=lists["wall_time"], total_wall_time=total_wt,
    )


# ---------------------------------------------------------------------------
# Method 4: Pure MLP (no structure)
# ---------------------------------------------------------------------------

class SimpleMLP(nn.Module):
    def __init__(self, n_inputs: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_inputs, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def run_pure_mlp(model: ModelSpec, cfg: ExpCConfig, seed: int) -> TrainResult:
    torch.manual_seed(seed)
    n_inputs = len(model.input_names)
    mlp = SimpleMLP(n_inputs)
    optimizer = torch.optim.Adam(mlp.parameters(), lr=cfg.lr)
    xs_list, ys = _generate_data(model, cfg)

    x_tensor = torch.tensor([[xd[n] for n in model.input_names]
                              for xd in xs_list])
    y_tensor = torch.stack(ys)

    lists = {"loss": [], "grad_norm": [], "wall_time": [], "params": {}}
    conv_epoch = None
    t_start = time.perf_counter()

    for epoch in range(cfg.max_epochs):
        t0 = time.perf_counter()

        preds = mlp(x_tensor)
        loss = ((preds - y_tensor) ** 2).sum()

        optimizer.zero_grad()
        loss.backward()

        grads = [p.grad.flatten() for p in mlp.parameters()
                 if p.grad is not None]
        gn = torch.cat(grads).norm().item() if grads else 0.0
        optimizer.step()
        wt = time.perf_counter() - t0

        loss_val = loss.item()
        lists["loss"].append(loss_val)
        lists["grad_norm"].append(gn)
        lists["wall_time"].append(wt)

        if conv_epoch is None and loss_val < cfg.convergence_threshold:
            conv_epoch = epoch
        elif conv_epoch is not None and epoch >= conv_epoch + 50:
            break

    total_wt = time.perf_counter() - t_start
    return TrainResult(
        method="pure_mlp", model_name=model.name, seed=seed,
        converged=conv_epoch is not None, convergence_epoch=conv_epoch,
        final_loss=lists["loss"][-1],
        final_param_errors={},
        loss_history=lists["loss"], param_history=lists["params"],
        grad_norm_history=lists["grad_norm"],
        wall_time_history=lists["wall_time"], total_wall_time=total_wt,
    )


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

METHOD_RUNNERS = {
    "dmci": run_dmci,
    "direct_compiled": run_direct_compiled,
    "handcoded_pytorch": run_handcoded_pytorch,
    "pure_mlp": run_pure_mlp,
}


def run_method(method: str, model: ModelSpec, cfg: ExpCConfig,
               seed: int) -> TrainResult:
    return METHOD_RUNNERS[method](model, cfg, seed)
