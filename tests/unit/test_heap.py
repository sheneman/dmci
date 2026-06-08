############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# test_heap.py: Tests for the tensor heap (cons cells, read/write, gradient flow).
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Tests for the tensor heap (cons cells, read/write, gradient flow)."""

import pytest
import torch

from neural_compiler.runtime.heap import TensorHeap
from neural_compiler.runtime.tagged_value import (
    VALUE_DIM, NIL, INT, FLOAT, PAIR,
    make_nil, make_int, make_float, make_pair,
    type_index, unwrap_number, unwrap_pair_addrs,
    is_nil, is_pair,
)
from neural_compiler.runtime.symbols import SymbolTable


class TestHeapBasics:
    def test_initial_state(self):
        heap = TensorHeap(max_size=1024)
        assert heap.allocated() == 0
        assert heap.max_size == 1024

    def test_cons_allocates_two_slots(self):
        heap = TensorHeap()
        car = make_int(1)
        cdr = make_int(2)
        pair = heap.cons(car, cdr)
        assert heap.allocated() == 2
        assert type_index(pair) == PAIR

    def test_cons_car_cdr_roundtrip(self):
        heap = TensorHeap()
        car_val = make_int(42)
        cdr_val = make_float(3.14)
        pair = heap.cons(car_val, cdr_val)

        got_car = heap.car(pair)
        got_cdr = heap.cdr(pair)
        assert unwrap_number(got_car).item() == 42.0
        assert unwrap_number(got_cdr).item() == pytest.approx(3.14)

    def test_nested_cons(self):
        heap = TensorHeap()
        inner = heap.cons(make_int(1), make_int(2))
        outer = heap.cons(inner, make_int(3))

        got_car = heap.car(outer)
        assert type_index(got_car) == PAIR
        got_inner_car = heap.car(got_car)
        assert unwrap_number(got_inner_car).item() == 1.0

    def test_multiple_pairs(self):
        heap = TensorHeap()
        p1 = heap.cons(make_int(1), make_nil())
        p2 = heap.cons(make_int(2), make_nil())
        p3 = heap.cons(make_int(3), make_nil())
        assert heap.allocated() == 6
        assert unwrap_number(heap.car(p1)).item() == 1.0
        assert unwrap_number(heap.car(p2)).item() == 2.0
        assert unwrap_number(heap.car(p3)).item() == 3.0


class TestHeapList:
    def test_empty_list(self):
        heap = TensorHeap()
        lst = heap.build_list([])
        assert type_index(lst) == NIL

    def test_single_element(self):
        heap = TensorHeap()
        lst = heap.build_list([make_int(42)])
        assert type_index(lst) == PAIR
        assert unwrap_number(heap.car(lst)).item() == 42.0
        assert type_index(heap.cdr(lst)) == NIL

    def test_three_elements(self):
        heap = TensorHeap()
        lst = heap.build_list([make_int(1), make_int(2), make_int(3)])

        assert unwrap_number(heap.car(lst)).item() == 1.0
        rest1 = heap.cdr(lst)
        assert unwrap_number(heap.car(rest1)).item() == 2.0
        rest2 = heap.cdr(rest1)
        assert unwrap_number(heap.car(rest2)).item() == 3.0
        assert type_index(heap.cdr(rest2)) == NIL

    def test_list_walk(self):
        """Walk a list and collect all values."""
        heap = TensorHeap()
        elements = [make_int(i) for i in range(5)]
        lst = heap.build_list(elements)

        collected = []
        current = lst
        while type_index(current) != NIL:
            collected.append(unwrap_number(heap.car(current)).item())
            current = heap.cdr(current)
        assert collected == [0.0, 1.0, 2.0, 3.0, 4.0]


class TestHeapWrite:
    def test_write_and_read(self):
        heap = TensorHeap()
        pair = heap.cons(make_int(1), make_int(2))
        car_addr, _ = unwrap_pair_addrs(pair)

        heap.write(car_addr, make_int(99))
        assert unwrap_number(heap.car(pair)).item() == 99.0

    def test_write_preserves_other_cells(self):
        heap = TensorHeap()
        p1 = heap.cons(make_int(1), make_nil())
        p2 = heap.cons(make_int(2), make_nil())

        car_addr, _ = unwrap_pair_addrs(p1)
        heap.write(car_addr, make_int(99))

        assert unwrap_number(heap.car(p1)).item() == 99.0
        assert unwrap_number(heap.car(p2)).item() == 2.0


class TestHeapOverflow:
    def test_overflow_raises(self):
        heap = TensorHeap(max_size=4)
        heap.cons(make_int(1), make_int(2))
        with pytest.raises(RuntimeError, match="Heap overflow"):
            heap.cons(make_int(3), make_int(4))
            heap.cons(make_int(5), make_int(6))

    def test_read_out_of_bounds(self):
        heap = TensorHeap()
        with pytest.raises(IndexError):
            heap.read(0)


class TestHeapReset:
    def test_reset_clears_state(self):
        heap = TensorHeap()
        heap.cons(make_int(1), make_int(2))
        assert heap.allocated() == 2
        heap.reset()
        assert heap.allocated() == 0


class TestHeapGradientFlow:
    def test_gradient_through_cons_car(self):
        """Gradient flows: x -> make_float(x) -> cons -> car -> unwrap -> loss."""
        heap = TensorHeap()
        x = torch.tensor(5.0, requires_grad=True)
        val = make_float(x)
        pair = heap.cons(val, make_nil())
        got = heap.car(pair)
        loss = unwrap_number(got)
        loss.backward()
        assert x.grad is not None
        assert x.grad.item() == 1.0

    def test_gradient_through_cons_cdr(self):
        heap = TensorHeap()
        x = torch.tensor(7.0, requires_grad=True)
        val = make_float(x)
        pair = heap.cons(make_nil(), val)
        got = heap.cdr(pair)
        loss = unwrap_number(got)
        loss.backward()
        assert x.grad is not None
        assert x.grad.item() == 1.0

    def test_gradient_through_nested_list(self):
        """Gradient flows through a 3-element list to the second element."""
        heap = TensorHeap()
        x = torch.tensor(3.0, requires_grad=True)
        elems = [make_float(torch.tensor(1.0)), make_float(x), make_float(torch.tensor(5.0))]
        lst = heap.build_list(elems)

        second = heap.car(heap.cdr(lst))
        loss = unwrap_number(second)
        loss.backward()
        assert x.grad is not None
        assert x.grad.item() == 1.0


class TestSymbolTable:
    def test_intern_new(self):
        st = SymbolTable()
        idx = st.intern("foo")
        assert idx == 0
        assert st.name(0) == "foo"

    def test_intern_existing(self):
        st = SymbolTable()
        idx1 = st.intern("foo")
        idx2 = st.intern("foo")
        assert idx1 == idx2

    def test_multiple_symbols(self):
        st = SymbolTable()
        a = st.intern("a")
        b = st.intern("b")
        c = st.intern("c")
        assert a != b != c
        assert st.name(a) == "a"
        assert st.name(b) == "b"
        assert st.name(c) == "c"

    def test_len(self):
        st = SymbolTable()
        assert len(st) == 0
        st.intern("x")
        st.intern("y")
        st.intern("x")
        assert len(st) == 2

    def test_contains(self):
        st = SymbolTable()
        st.intern("hello")
        assert st.contains("hello")
        assert not st.contains("world")

    def test_unknown_id_raises(self):
        st = SymbolTable()
        with pytest.raises(KeyError):
            st.name(999)
