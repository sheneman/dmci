############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# test_tagged_value.py: Tests for tagged value representation.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Tests for tagged value representation."""

import pytest
import torch

from neural_compiler.runtime.tagged_value import (
    TAG_DIM, PAYLOAD_DIM, VALUE_DIM,
    NIL, BOOL, INT, FLOAT, CHAR, SYMBOL, PAIR, STRING, CLOSURE, VECTOR,
    make_nil, make_bool, make_int, make_float, make_char, make_symbol,
    make_pair, make_string, make_closure, make_vector,
    extract_tag, extract_payload, type_index, type_name,
    is_type, is_nil, is_pair, is_number, is_symbol, is_closure,
    unwrap_number, unwrap_bool, unwrap_pair_addrs, unwrap_closure,
    tagged_if, soft_select, from_scalar, to_scalar,
)


class TestTaggedValueLayout:
    def test_dimensions(self):
        assert TAG_DIM == 10
        assert PAYLOAD_DIM == 4
        assert VALUE_DIM == 14

    def test_nil_shape(self):
        v = make_nil()
        assert v.shape == (VALUE_DIM,)

    def test_all_constructors_produce_correct_shape(self):
        values = [
            make_nil(),
            make_bool(True),
            make_int(42),
            make_float(3.14),
            make_char(65),
            make_symbol(0),
            make_pair(0.0, 1.0),
            make_string(0.0, 5.0),
            make_closure(0, 0.0),
            make_vector(0.0, 3.0),
        ]
        for v in values:
            assert v.shape == (VALUE_DIM,), f"Bad shape for {v}"


class TestTaggedValueTypes:
    def test_nil_type(self):
        v = make_nil()
        assert type_index(v) == NIL
        assert type_name(v) == "nil"
        assert is_nil(v).item() == 1.0

    def test_bool_true(self):
        v = make_bool(True)
        assert type_index(v) == BOOL
        assert unwrap_bool(v).item() == 1.0

    def test_bool_false(self):
        v = make_bool(False)
        assert type_index(v) == BOOL
        assert unwrap_bool(v).item() == 0.0

    def test_int(self):
        v = make_int(42)
        assert type_index(v) == INT
        assert unwrap_number(v).item() == 42.0

    def test_float(self):
        v = make_float(3.14)
        assert type_index(v) == FLOAT
        assert unwrap_number(v).item() == pytest.approx(3.14)

    def test_char(self):
        v = make_char(65)
        assert type_index(v) == CHAR
        assert unwrap_number(v).item() == 65.0

    def test_symbol(self):
        v = make_symbol(7)
        assert type_index(v) == SYMBOL
        assert unwrap_number(v).item() == 7.0
        assert is_symbol(v).item() == 1.0

    def test_pair(self):
        v = make_pair(10.0, 11.0)
        assert type_index(v) == PAIR
        car_addr, cdr_addr = unwrap_pair_addrs(v)
        assert car_addr.item() == 10.0
        assert cdr_addr.item() == 11.0
        assert is_pair(v).item() == 1.0

    def test_closure(self):
        v = make_closure(3, 100.0)
        assert type_index(v) == CLOSURE
        func_id, env_addr = unwrap_closure(v)
        assert func_id.item() == 3.0
        assert env_addr.item() == 100.0
        assert is_closure(v).item() == 1.0


class TestTagExtraction:
    def test_one_hot_tags(self):
        """Each type constructor should produce exactly one 1.0 in the tag."""
        values = [
            (make_nil(), NIL),
            (make_bool(True), BOOL),
            (make_int(1), INT),
            (make_float(1.0), FLOAT),
            (make_char(65), CHAR),
            (make_symbol(0), SYMBOL),
            (make_pair(0, 1), PAIR),
            (make_string(0, 0), STRING),
            (make_closure(0, 0), CLOSURE),
            (make_vector(0, 0), VECTOR),
        ]
        for v, expected_type in values:
            tag = extract_tag(v)
            assert tag.sum().item() == 1.0, f"Tag not one-hot for type {expected_type}"
            assert tag[expected_type].item() == 1.0

    def test_is_number_int(self):
        v = make_int(5)
        assert is_number(v).item() == 1.0

    def test_is_number_float(self):
        v = make_float(5.0)
        assert is_number(v).item() == 1.0

    def test_is_number_nil(self):
        v = make_nil()
        assert is_number(v).item() == 0.0


class TestTaggedIf:
    def test_true_branch(self):
        test = make_bool(True)
        then_val = make_int(10)
        else_val = make_int(20)
        result = tagged_if(test, then_val, else_val)
        assert unwrap_number(result).item() == pytest.approx(10.0)

    def test_false_branch(self):
        test = make_bool(False)
        then_val = make_int(10)
        else_val = make_int(20)
        result = tagged_if(test, then_val, else_val)
        assert unwrap_number(result).item() == pytest.approx(20.0)

    def test_numeric_truth(self):
        test = make_int(42)
        then_val = make_float(1.0)
        else_val = make_float(2.0)
        result = tagged_if(test, then_val, else_val)
        assert unwrap_number(result).item() == pytest.approx(1.0)

    def test_zero_is_false(self):
        test = make_int(0)
        then_val = make_float(1.0)
        else_val = make_float(2.0)
        result = tagged_if(test, then_val, else_val)
        assert unwrap_number(result).item() == pytest.approx(2.0)


class TestSoftSelect:
    def test_dispatch_by_type(self):
        int_val = make_int(5)
        branches = {
            INT: make_int(100),
            FLOAT: make_int(200),
        }
        result = soft_select(int_val, branches)
        assert unwrap_number(result).item() == pytest.approx(100.0)

    def test_dispatch_float(self):
        float_val = make_float(5.0)
        branches = {
            INT: make_int(100),
            FLOAT: make_int(200),
        }
        result = soft_select(float_val, branches)
        assert unwrap_number(result).item() == pytest.approx(200.0)


class TestFromScalar:
    def test_bool(self):
        v = from_scalar(True)
        assert type_index(v) == BOOL
        assert unwrap_bool(v).item() == 1.0

    def test_int(self):
        v = from_scalar(42)
        assert type_index(v) == INT
        assert unwrap_number(v).item() == 42.0

    def test_float(self):
        v = from_scalar(3.14)
        assert type_index(v) == FLOAT
        assert unwrap_number(v).item() == pytest.approx(3.14)

    def test_tensor(self):
        v = from_scalar(torch.tensor(7.0))
        assert type_index(v) == FLOAT
        assert unwrap_number(v).item() == 7.0

    def test_to_scalar_roundtrip(self):
        for val in [0, 1, -5, 42, 3.14, 0.0]:
            tv = from_scalar(val)
            assert to_scalar(tv) == pytest.approx(float(val))


class TestTensorInputs:
    """Test that constructors work with tensor inputs (for gradient flow)."""

    def test_make_float_tensor(self):
        x = torch.tensor(5.0, requires_grad=True)
        v = make_float(x)
        assert v.shape == (VALUE_DIM,)
        result = unwrap_number(v)
        result.backward()
        assert x.grad is not None
        assert x.grad.item() == 1.0

    def test_make_int_tensor(self):
        x = torch.tensor(3.0, requires_grad=True)
        v = make_int(x)
        result = unwrap_number(v)
        result.backward()
        assert x.grad is not None

    def test_make_bool_tensor(self):
        x = torch.tensor(1.0)
        v = make_bool(x)
        assert unwrap_bool(v).item() == 1.0

    def test_tagged_if_gradient(self):
        """Gradient flows through tagged_if MUX."""
        x = torch.tensor(3.0, requires_grad=True)
        then_val = make_float(x)
        else_val = make_float(torch.tensor(0.0))
        test = make_bool(True)
        result = tagged_if(test, then_val, else_val)
        loss = unwrap_number(result)
        loss.backward()
        assert x.grad is not None
        assert x.grad.item() == 1.0
