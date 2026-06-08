############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# test_soft_choice.py: Unit tests for the soft-choice AST node and evaluation pipeline.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Unit tests for the soft-choice AST node and evaluation pipeline."""

import pytest
import torch
from neural_compiler.parser import parse
from neural_compiler.parser.ast_nodes import SoftChoice, Var, App
from neural_compiler.anf import to_anf
from neural_compiler.anf.anf_nodes import ANFSoftChoice
from neural_compiler.graph import build_graph
from neural_compiler.evaluator.engine import (
    _eval_graph,
    set_soft_choice_gumbel,
    set_soft_choice_tau,
)


@pytest.fixture(autouse=True)
def _deterministic_softmax():
    """Disable Gumbel noise and reset tau for reproducible tests."""
    set_soft_choice_gumbel(False)
    set_soft_choice_tau(1.0)
    yield
    set_soft_choice_gumbel(True)
    set_soft_choice_tau(1.0)


def _compile(source, inputs=None):
    inputs = inputs or {}
    input_decl = {k: None for k in inputs}
    anf = to_anf(parse(source))
    graph = build_graph(anf, inputs=input_decl)
    return graph


def _eval_tensor(source, inputs):
    graph = _compile(source, inputs)
    tensor_inputs = {k: torch.tensor(v, dtype=torch.float32) if not isinstance(v, torch.Tensor) else v
                     for k, v in inputs.items()}
    values = _eval_graph(graph, tensor_inputs)
    return values[graph.root_id]


class TestParseSoftChoice:
    def test_basic_parse(self):
        ast = parse("(soft-choice ((+ x x) (* x x)) w)")
        assert isinstance(ast, SoftChoice)
        assert len(ast.options) == 2
        assert isinstance(ast.weights, Var)
        assert ast.weights.name == "w"

    def test_three_options(self):
        ast = parse("(soft-choice ((+ x 1) (- x 1) (* x 2)) w)")
        assert isinstance(ast, SoftChoice)
        assert len(ast.options) == 3

    def test_requires_two_args(self):
        with pytest.raises(SyntaxError, match="options list and weights"):
            parse("(soft-choice ((+ x x)))")

    def test_requires_at_least_two_options(self):
        with pytest.raises(SyntaxError, match="at least 2"):
            parse("(soft-choice ((+ x x)) w)")


class TestANFSoftChoice:
    def test_anf_produces_soft_choice_node(self):
        ast = parse("(soft-choice ((+ x x) (* x x)) w)")
        anf = to_anf(ast)
        assert isinstance(anf, ANFSoftChoice)
        assert len(anf.options) == 2


class TestEvalSoftChoice:
    def test_equal_logits_averages(self):
        result = _eval_tensor(
            "(soft-choice ((+ x x) (* x x)) w)",
            {"x": 3.0, "w": torch.tensor([0.0, 0.0])},
        )
        assert result.item() == pytest.approx(7.5, abs=1e-5)

    def test_strong_first_selects_first(self):
        result = _eval_tensor(
            "(soft-choice ((+ x x) (* x x)) w)",
            {"x": 3.0, "w": torch.tensor([10.0, 0.0])},
        )
        assert result.item() == pytest.approx(6.0, abs=0.01)

    def test_strong_second_selects_second(self):
        result = _eval_tensor(
            "(soft-choice ((+ x x) (* x x)) w)",
            {"x": 3.0, "w": torch.tensor([0.0, 10.0])},
        )
        assert result.item() == pytest.approx(9.0, abs=0.01)

    def test_four_options(self):
        result = _eval_tensor(
            "(soft-choice ((+ x x) (- x x) (* x x) (/ x x)) w)",
            {"x": 4.0, "w": torch.tensor([0.0, 0.0, 10.0, 0.0])},
        )
        assert result.item() == pytest.approx(16.0, abs=0.01)

    def test_low_tau_sharpens(self):
        set_soft_choice_tau(0.01)
        result = _eval_tensor(
            "(soft-choice ((+ x x) (* x x)) w)",
            {"x": 3.0, "w": torch.tensor([1.0, 0.0])},
        )
        assert result.item() == pytest.approx(6.0, abs=1e-4)


class TestSoftChoiceGradient:
    def test_gradient_flows_to_weights(self):
        graph = _compile("(soft-choice ((+ x x) (* x x)) w)", {"x", "w"})
        w = torch.tensor([0.0, 0.0], requires_grad=True)
        x = torch.tensor(3.0)
        values = _eval_graph(graph, {"x": x, "w": w})
        result = values[graph.root_id]
        result.backward()
        assert w.grad is not None
        assert w.grad.shape == (2,)
        assert w.grad[0].item() == pytest.approx(-0.75, abs=1e-4)
        assert w.grad[1].item() == pytest.approx(0.75, abs=1e-4)

    def test_gradient_flows_to_x(self):
        graph = _compile("(soft-choice ((+ x x) (* x x)) w)", {"x", "w"})
        w = torch.tensor([0.0, 0.0], requires_grad=True)
        x = torch.tensor(3.0, requires_grad=True)
        values = _eval_graph(graph, {"x": x, "w": w})
        result = values[graph.root_id]
        result.backward()
        assert x.grad is not None
        assert x.grad.item() == pytest.approx(4.0, abs=1e-4)

    def test_operator_recovery_converges(self):
        """Core Experiment E.1 scenario: recover * from {+, -, *, /}."""
        source = "(soft-choice ((+ x x) (- x x) (* x x) (/ x x)) w)"
        graph = _compile(source, {"x", "w"})

        torch.manual_seed(0)
        xs = torch.randn(16) * 2
        w = torch.tensor([0.0, 0.0, 0.0, 0.0], requires_grad=True)
        optimizer = torch.optim.Adam([w], lr=0.1)

        for epoch in range(150):
            tau = max(0.1, 1.0 - epoch * 0.006)
            set_soft_choice_tau(tau)
            loss = torch.tensor(0.0)
            for xi in xs:
                values = _eval_graph(graph, {"x": xi, "w": w})
                pred = values[graph.root_id]
                loss = loss + (pred - xi * xi) ** 2
            loss = loss / len(xs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        selected = torch.argmax(w).item()
        assert selected == 2, f"Expected * (idx 2), got idx {selected}"
