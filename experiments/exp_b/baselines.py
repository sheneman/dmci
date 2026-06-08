############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# baselines.py: Four methods for Experiment B. 1. dmci — program as data through compiled self-hosted interpreter, autograd 2....
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Four methods for Experiment B.

1. dmci              — program as data through compiled self-hosted interpreter, autograd
2. direct_compiled   — same program compiled directly (no interpreter), autograd
3. handcoded_pytorch — hand-coded PyTorch implementation of the same model, autograd
4. pure_mlp          — MLP learns the entire function from data, no physics

Key comparisons:
- dmci vs direct_compiled: interpretation overhead (wall-clock) with same gradient quality
- dmci vs handcoded_pytorch: automated pipeline vs manual coding, both exact
- dmci vs pure_mlp: compiled physics vs learning from scratch
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch
import torch.nn as nn

from neural_compiler.compiler import compile_program
from neural_compiler.evaluator import evaluate
from neural_compiler.runtime.tagged_value import make_float, unwrap_number

from .config import ExpBConfig, DEFAULT
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


def _generate_data(model: ModelSpec, cfg: ExpBConfig):
    n_inputs = len(model.input_names)

    if n_inputs == 1:
        xs_raw = torch.linspace(model.x_range[0], model.x_range[1], cfg.n_data_points)
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
    elif n_inputs == 3:
        n_per = max(3, int(cfg.n_data_points ** (1.0 / 3)))
        vals = [torch.linspace(model.x_range[0], model.x_range[1], n_per)
                for _ in range(3)]
        xs_list = []
        for a in vals[0]:
            for b in vals[1]:
                for c in vals[2]:
                    xs_list.append({
                        model.input_names[0]: a.item(),
                        model.input_names[1]: b.item(),
                        model.input_names[2]: c.item(),
                    })
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


def _train_compiled(method_name, graph, model, cfg, seed, use_tagged):
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

def run_dmci(model: ModelSpec, cfg: ExpBConfig, seed: int) -> TrainResult:
    cache_key = f"dmci_{model.name}"
    if cache_key not in _GRAPH_CACHE:
        all_inputs = {n: None for n in _all_input_names(model)}
        _GRAPH_CACHE[cache_key] = compile_program(
            model.interp_source, inputs=all_inputs, prelude=True)
    graph = _GRAPH_CACHE[cache_key]
    return _train_compiled("dmci", graph, model, cfg, seed, use_tagged=True)


# ---------------------------------------------------------------------------
# Method 2: Direct compilation (no interpreter)
# ---------------------------------------------------------------------------

def run_direct_compiled(model: ModelSpec, cfg: ExpBConfig, seed: int) -> TrainResult:
    cache_key = f"direct_{model.name}"
    if cache_key not in _GRAPH_CACHE:
        all_inputs = {n: None for n in _all_input_names(model)}
        _GRAPH_CACHE[cache_key] = compile_program(
            model.direct_source, inputs=all_inputs, prelude=True)
    graph = _GRAPH_CACHE[cache_key]
    return _train_compiled("direct_compiled", graph, model, cfg, seed, use_tagged=True)


# ---------------------------------------------------------------------------
# Method 3: Hand-coded PyTorch
# ---------------------------------------------------------------------------

def _build_handcoded_fn(model: ModelSpec):
    def fn(inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        v = {**inputs}

        if model.name == "M01_coulomb":
            return v["k"] * v["q1"] * v["q2"] / (v["r"] ** 2)
        elif model.name == "M02_beer_lambert":
            return v["epsilon"] * v["c"] * v["l"]
        elif model.name == "M03_michaelis_menten":
            return v["Vmax"] * v["S"] / (v["Km"] + v["S"])
        elif model.name == "M04_arrhenius":
            return v["A"] * torch.exp(-v["Ea"] * v["T"])
        elif model.name == "M05_hookes_spring":
            return v["A"] * torch.exp(-v["b"] * v["t"]) * torch.cos(v["omega"] * v["t"])
        elif model.name == "M06_logistic_growth":
            return v["K"] / (1.0 + (v["K"] - 1.0) * torch.exp(-v["r"] * v["t"]))
        elif model.name == "M07_power_law":
            return v["a"] * torch.pow(v["x"], v["b"])
        elif model.name == "M08_euler_ode":
            y = v["x"]
            for _ in range(10):
                y = y + (-v["k"] * y) * 0.1
            return y
        elif model.name == "M09_taylor_exp":
            ax = v["a"] * v["x"]
            result = torch.tensor(0.0)
            for i in range(9):
                result = result + torch.pow(ax, i) / math.factorial(i)
            return result
        elif model.name == "M10_smooth_activation":
            return v["a"] * v["x"] * torch.sigmoid(v["b"] * v["x"])
        elif model.name == "M11_recursive_filter":
            y = torch.tensor(0.0)
            for _ in range(8):
                y = v["alpha"] * v["x"] + (1.0 - v["alpha"]) * y
            return y
        elif model.name == "M12_newton_sqrt":
            guess = v["a"] * v["x"]
            for _ in range(5):
                guess = (guess + v["x"] / guess) / 2.0
            return guess
        elif model.name == "M13_composed_transforms":
            return v["a"] * v["x"] + v["b"]
        elif model.name == "M14_anomaly_scorer":
            return v["w1"] * v["f1"] + v["w2"] * v["f2"] + v["w3"] * v["f3"]
        elif model.name == "M15_horner_eval":
            x = v["x"]
            return v["a0"] + x * (v["a1"] + x * (v["a2"] + x * v["a3"]))
        else:
            raise ValueError(f"No hand-coded implementation for {model.name}")

    return fn


def run_handcoded_pytorch(model: ModelSpec, cfg: ExpBConfig, seed: int) -> TrainResult:
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
# Method 4: Pure MLP (no physics)
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


def run_pure_mlp(model: ModelSpec, cfg: ExpBConfig, seed: int) -> TrainResult:
    torch.manual_seed(seed)
    n_inputs = len(model.input_names)
    mlp = SimpleMLP(n_inputs)
    optimizer = torch.optim.Adam(mlp.parameters(), lr=cfg.lr)
    xs_list, ys = _generate_data(model, cfg)

    x_tensor = torch.tensor([[xd[n] for n in model.input_names] for xd in xs_list])
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

        grads = [p.grad.flatten() for p in mlp.parameters() if p.grad is not None]
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


def run_method(method: str, model: ModelSpec, cfg: ExpBConfig, seed: int) -> TrainResult:
    return METHOD_RUNNERS[method](model, cfg, seed)
