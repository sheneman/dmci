############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# test_cli_serialize.py: Tests for the .ncg serializer, the nncompile CLI, and the torch nn.Module emitter.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Tests for the .ncg serializer, the nncompile CLI, and the torch nn.Module emitter."""
import importlib.util
import json

import pytest
import torch

from neural_compiler import cli
from neural_compiler.compiler import compile_program
from neural_compiler.emit import emit_torch_module
from neural_compiler.evaluator import evaluate
from neural_compiler.serialize import (
    graph_from_dict,
    graph_to_dict,
    load_compiled,
    save_compiled,
)

ROUNDTRIP_CASES = [
    ("(+ (* 3 4) 5)", {}, 17.0),
    ("(+ (* 3 x) (- y 1))", {"x": 4.0, "y": 7.0}, 18.0),
    ("(if (> x 0) 1 -1)", {"x": 5.0}, 1.0),
    ("(let ((a 2) (b 3)) (* a b))", {}, 6.0),
    ("(loop ((i 1) (acc 0)) (if (> i n) acc (recur (+ i 1) (+ acc i))))", {"n": 5.0}, 15.0),
    ("(letrec ((f (lambda (n) (if (<= n 1) 1 (* n (f (- n 1))))))) (f 5))", {}, 120.0),
    ("(letrec ((fib (lambda (n) (if (< n 2) n (+ (fib (- n 1)) (fib (- n 2))))))) (fib 8))", {}, 21.0),
]


def _compile(src, inputs):
    return compile_program(src, inputs={k: None for k in inputs} or None)


@pytest.mark.parametrize("src,inputs,expected", ROUNDTRIP_CASES)
def test_serialize_roundtrip_dict(src, inputs, expected):
    g = _compile(src, inputs)
    g2 = graph_from_dict(graph_to_dict(g))
    assert abs(float(evaluate(g2, inputs)) - expected) < 1e-9


@pytest.mark.parametrize("src,inputs,expected", ROUNDTRIP_CASES)
def test_serialize_roundtrip_file(tmp_path, src, inputs, expected):
    g = _compile(src, inputs)
    path = tmp_path / "m.ncg"
    save_compiled(g, path, source=src)
    g2 = load_compiled(path)
    assert abs(float(evaluate(g2, inputs)) - expected) < 1e-9


def test_ncg_is_valid_json_and_versioned(tmp_path):
    g = _compile("(* a x)", {"a": None, "x": None})
    path = tmp_path / "m.ncg"
    save_compiled(g, path, source="(* a x)")
    art = json.loads(path.read_text())
    assert art["format"] == "nncompile-graph"
    assert art["version"] == 1
    assert art["source"] == "(* a x)"


def test_load_rejects_bad_format(tmp_path):
    from neural_compiler.serialize import from_artifact

    with pytest.raises(ValueError):
        from_artifact({"format": "not-ncg", "version": 1, "graph": {}})


def test_cli_compile_run_info(tmp_path, capsys):
    scm = tmp_path / "p.scm"
    scm.write_text("(/ (* k (* q1 q2)) (* r r))")
    ncg = tmp_path / "p.ncg"

    assert cli.main(["compile", str(scm), "-o", str(ncg)]) == 0
    assert ncg.exists()

    capsys.readouterr()
    assert cli.main(["run", str(ncg), "--inputs", '{"k":2.0,"q1":3.0,"q2":4.0,"r":2.0}']) == 0
    assert abs(float(capsys.readouterr().out.strip()) - 6.0) < 1e-6

    capsys.readouterr()
    assert cli.main(["info", str(scm)]) == 0
    info = capsys.readouterr().out
    assert "inputs:" in info and "backends:" in info


def test_cli_run_numpy_backend(tmp_path, capsys):
    scm = tmp_path / "p.scm"
    scm.write_text("(+ (* 3 x) 1)")
    capsys.readouterr()
    assert cli.main(["run", str(scm), "--inputs", '{"x": 4.0}', "--backend", "numpy"]) == 0
    assert abs(float(capsys.readouterr().out.strip()) - 13.0) < 1e-6


def _load_emitted(path):
    spec = importlib.util.spec_from_file_location("emitted_model", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_emit_module_imports_and_optimizes(tmp_path):
    g = compile_program("(/ (* k (* q1 q2)) (* r r))",
                        inputs={k: None for k in ["k", "q1", "q2", "r"]})
    code = emit_torch_module(g, "(/ (* k (* q1 q2)) (* r r))",
                             params=["k"], data_inputs=["q1", "q2", "r"], module_name="emitted_model")
    path = tmp_path / "emitted_model.py"
    path.write_text(code)
    mod = _load_emitted(path)
    assert mod.PARAMS == ["k"] and mod.INPUTS == ["q1", "q2", "r"]

    model = mod.CompiledModel(k=0.1)
    opt = torch.optim.Adam(model.parameters(), lr=0.2)
    rs = torch.linspace(1.0, 3.0, 12)
    ys = 5.0 * 3.0 * 4.0 / (rs * rs)  # true k = 5.0
    for _ in range(400):
        pred = torch.stack([model(q1=3.0, q2=4.0, r=float(r)) for r in rs])
        loss = ((pred - ys) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert abs(model.params["k"].item() - 5.0) < 1e-2


def test_emit_rejects_unknown_param():
    g = compile_program("(* a x)", inputs={"a": None, "x": None})
    with pytest.raises(ValueError):
        emit_torch_module(g, "(* a x)", params=["zzz"], data_inputs=["x"])


# --------------------------------------------------------------------- batched loops
def test_batched_loop_divergent_bounds():
    """Per-element loop bounds: each batch element terminates on its own iteration."""
    from neural_compiler.compiler import compile_scheme
    from neural_compiler.evaluator import evaluate_batched

    g = compile_scheme(
        "(loop ((i 0) (acc 0)) (if (>= i n) acc (recur (+ i 1) (+ acc i))))", inputs={"n": None})
    ns = [3, 5, 7, 1, 10]
    out = evaluate_batched(g, {"n": torch.tensor([float(k) for k in ns])})
    assert out.tolist() == [float(sum(range(k))) for k in ns]


def test_batched_loop_gradient():
    from neural_compiler.compiler import compile_scheme
    from neural_compiler.evaluator import evaluate_batched

    g = compile_scheme(
        "(* a (loop ((i 0) (acc 0)) (if (>= i n) acc (recur (+ i 1) (+ acc i)))))",
        inputs={"a": None, "n": None})
    ns = torch.tensor([3., 5., 7., 4.])
    ys = 2.5 * torch.tensor([float(sum(range(int(k)))) for k in ns])
    a = torch.nn.Parameter(torch.tensor(0.1))
    opt = torch.optim.Adam([a], lr=0.3)
    for _ in range(300):
        loss = ((evaluate_batched(g, {"a": a, "n": ns}) - ys) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    assert abs(a.item() - 2.5) < 1e-2


# --------------------------------------------------------------------- DMCI
def test_compile_dmci_matches_direct_and_is_differentiable():
    from neural_compiler.dmci import compile_dmci
    from neural_compiler.runtime.tagged_value import make_float, unwrap_number

    src = "(/ (* k (* q1 q2)) (* r r))"
    g = compile_dmci(src)
    assert g.uses_tagged_values  # ran through the meta-circular interpreter
    feed = {n: make_float(torch.tensor(v)) for n, v in {"k": 2., "q1": 3., "q2": 4., "r": 2.}.items()}
    assert abs(float(unwrap_number(evaluate(g, feed))) - 6.0) < 1e-5  # == direct compilation

    k = torch.nn.Parameter(torch.tensor(0.1))
    opt = torch.optim.Adam([k], lr=0.2)
    rs = torch.linspace(1., 3., 8); ys = 5.0 * 3 * 4 / (rs * rs)
    for _ in range(400):
        loss = torch.tensor(0.0)
        for r, y in zip(rs, ys):
            pred = unwrap_number(evaluate(g, {
                "k": make_float(k), "q1": make_float(torch.tensor(3.)),
                "q2": make_float(torch.tensor(4.)), "r": make_float(r)}))
            loss = loss + (pred - y) ** 2
        opt.zero_grad(); loss.backward(); opt.step()
    assert abs(k.item() - 5.0) < 1e-2


def test_cli_dmci_run(tmp_path, capsys):
    scm = tmp_path / "p.scm"
    scm.write_text("(/ (* k (* q1 q2)) (* r r))")
    capsys.readouterr()
    assert cli.main(["run", str(scm), "--dmci",
                     "--inputs", '{"k":2.0,"q1":3.0,"q2":4.0,"r":2.0}']) == 0
    assert abs(float(capsys.readouterr().out.strip()) - 6.0) < 1e-5


def test_evaluate_respects_max_heap():
    """``evaluate(..., max_heap=N)`` sets the tagged-value heap cap (a no-GC bump-allocator
    limit); the ``--max-heap`` CLI flag exposes it for genuinely large list/recursion workloads.
    Exercised here with a fixed cons-chain."""
    from neural_compiler.compiler import compile_scheme
    from neural_compiler.runtime.tagged_value import is_pair

    g = compile_scheme("(cons 1 (cons 2 (cons 3 (cons 4 (cons 5 (quote ()))))))", inputs=None)
    assert g.uses_tagged_values
    with pytest.raises(RuntimeError, match="Heap overflow"):
        evaluate(g, {}, max_heap=8)            # room for 4 cells; the 5th cons overflows
    out = evaluate(g, {}, max_heap=10_000)     # roomy: succeeds
    assert bool(is_pair(out).item() > 0.5)


# --------------------------------------------------------------------- JAX emit
def test_emit_jax_module_optimizes(tmp_path):
    pytest.importorskip("jax")
    import jax
    import jax.numpy as jnp
    from neural_compiler.emit import emit_jax_module

    g = compile_program("(/ (* k (* q1 q2)) (* r r))",
                        inputs={k: None for k in ["k", "q1", "q2", "r"]})
    code = emit_jax_module(g, "(/ (* k (* q1 q2)) (* r r))",
                           params=["k"], data_inputs=["q1", "q2", "r"], module_name="emitted_jax")
    path = tmp_path / "emitted_jax.py"
    path.write_text(code)
    mod = _load_emitted(path)

    rs = jnp.linspace(1., 3., 8); ys = 5.0 * 3 * 4 / (rs * rs)

    def loss(p):
        pred = jax.vmap(lambda r: mod.apply(p, q1=jnp.float32(3.), q2=jnp.float32(4.), r=r))(rs)
        return jnp.mean((pred - ys) ** 2)

    p = mod.init_params(k=0.1)
    for _ in range(2000):
        gr = jax.grad(loss)(p)
        p = {"k": p["k"] - 0.002 * gr["k"]}
    assert abs(float(p["k"]) - 5.0) < 1e-2


def test_emit_jax_rejects_tagged():
    from neural_compiler.dmci import compile_dmci
    from neural_compiler.emit import emit_jax_module

    g = compile_dmci("(* a x)")  # tagged (DMCI)
    with pytest.raises(ValueError):
        emit_jax_module(g, "(* a x)", params=["a"], data_inputs=["x"])
