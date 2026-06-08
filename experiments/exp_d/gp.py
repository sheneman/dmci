############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# gp.py: Minimal GP framework for symbolic regression. ASTs are built from {+, -, *, /, sin, cos, x, const_0, const_1,...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Minimal GP framework for symbolic regression.

ASTs are built from {+, -, *, /, sin, cos, x, const_0, const_1, ...}.
Each AST has embedded numeric constants that are learnable.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass


# Terminal and function sets
FUNCTIONS_BINARY = ["+", "-", "*", "/"]
FUNCTIONS_UNARY = ["sin", "cos"]
FUNCTIONS = FUNCTIONS_BINARY + FUNCTIONS_UNARY
TERMINALS = ["x"]  # plus const_N nodes generated dynamically


@dataclass
class GPNode:
    kind: str  # "func", "var", "const"
    value: str  # operator name, variable name, or const_N
    children: list[GPNode]

    def depth(self) -> int:
        if not self.children:
            return 1
        return 1 + max(c.depth() for c in self.children)

    def size(self) -> int:
        return 1 + sum(c.size() for c in self.children)

    def copy(self) -> GPNode:
        return copy.deepcopy(self)


class ConstCounter:
    """Track how many constants have been allocated in a tree."""

    def __init__(self):
        self.count = 0

    def next(self) -> str:
        name = f"const_{self.count}"
        self.count += 1
        return name


def random_tree(depth: int, counter: ConstCounter, method: str = "grow") -> GPNode:
    """Generate a random expression tree.

    method='full': always pick functions until max depth, then terminals.
    method='grow': randomly pick functions or terminals at each non-leaf level.
    """
    if depth <= 1:
        return _random_terminal(counter)

    if method == "full" or (method == "grow" and random.random() < 0.7):
        op = random.choice(FUNCTIONS)
        if op in FUNCTIONS_UNARY:
            child = random_tree(depth - 1, counter, method)
            return GPNode("func", op, [child])
        else:
            left = random_tree(depth - 1, counter, method)
            right = random_tree(depth - 1, counter, method)
            return GPNode("func", op, [left, right])
    else:
        return _random_terminal(counter)


def _random_terminal(counter: ConstCounter) -> GPNode:
    if random.random() < 0.5:
        return GPNode("var", "x", [])
    else:
        return GPNode("const", counter.next(), [])


def collect_consts(tree: GPNode) -> list[str]:
    """Collect all constant names in the tree, in order."""
    result = []
    _collect_consts_impl(tree, result)
    return result


def _collect_consts_impl(node: GPNode, acc: list[str]):
    if node.kind == "const":
        acc.append(node.value)
    for c in node.children:
        _collect_consts_impl(c, acc)


def to_scheme(node: GPNode) -> str:
    """Convert a GP tree to a Scheme expression string."""
    if node.kind == "var":
        return node.value
    if node.kind == "const":
        return node.value
    op = node.value
    if op in FUNCTIONS_UNARY:
        return f"({op} {to_scheme(node.children[0])})"
    else:
        return f"({op} {to_scheme(node.children[0])} {to_scheme(node.children[1])})"


def _all_nodes(node: GPNode) -> list[GPNode]:
    """Flat list of all nodes in the tree (preorder)."""
    result = [node]
    for c in node.children:
        result.extend(_all_nodes(c))
    return result


def subtree_mutation(tree: GPNode, max_depth: int) -> GPNode:
    """Replace a random subtree with a new random subtree."""
    tree = tree.copy()
    nodes = _all_nodes(tree)
    if len(nodes) <= 1:
        counter = ConstCounter()
        return random_tree(max_depth, counter)

    target_idx = random.randint(1, len(nodes) - 1)
    parent, child_idx = _find_parent(tree, nodes[target_idx])
    if parent is None:
        counter = ConstCounter()
        return random_tree(max_depth, counter)

    counter = ConstCounter()
    counter.count = len(collect_consts(tree))
    remaining_depth = max(2, max_depth - _depth_of(tree, parent))
    new_subtree = random_tree(remaining_depth, counter)
    parent.children[child_idx] = new_subtree
    return _renumber_consts(tree)


def subtree_crossover(parent1: GPNode, parent2: GPNode) -> GPNode:
    """Swap a random subtree from parent2 into parent1."""
    child = parent1.copy()
    donor = parent2.copy()

    child_nodes = _all_nodes(child)
    donor_nodes = _all_nodes(donor)

    if len(child_nodes) <= 1 or len(donor_nodes) <= 1:
        return child

    cx_point = random.randint(1, len(child_nodes) - 1)
    donor_point = random.randint(0, len(donor_nodes) - 1)

    parent, child_idx = _find_parent(child, child_nodes[cx_point])
    if parent is None:
        return child

    parent.children[child_idx] = donor_nodes[donor_point].copy()
    return _renumber_consts(child)


def tournament_select(population: list[tuple[GPNode, float]],
                      k: int) -> GPNode:
    """Select the best individual from a random tournament of size k."""
    contestants = random.sample(population, min(k, len(population)))
    best = min(contestants, key=lambda x: x[1])
    return best[0].copy()


def _find_parent(root: GPNode, target: GPNode) -> tuple[GPNode | None, int]:
    """Find the parent of target node and the child index."""
    for i, c in enumerate(root.children):
        if c is target:
            return root, i
        result = _find_parent(c, target)
        if result[0] is not None:
            return result
    return None, -1


def _depth_of(root: GPNode, target: GPNode, current: int = 0) -> int:
    """Find the depth of target node within root."""
    if root is target:
        return current
    for c in root.children:
        d = _depth_of(c, target, current + 1)
        if d >= 0:
            return d
    return -1


def _renumber_consts(tree: GPNode) -> GPNode:
    """Renumber all const nodes sequentially (const_0, const_1, ...)."""
    counter = [0]

    def _visit(node: GPNode):
        if node.kind == "const":
            node.value = f"const_{counter[0]}"
            counter[0] += 1
        for c in node.children:
            _visit(c)

    _visit(tree)
    return tree


def make_initial_population(pop_size: int, min_depth: int,
                            max_depth: int) -> list[GPNode]:
    """Ramped half-and-half initialization."""
    pop = []
    for i in range(pop_size):
        depth = min_depth + (i % (max_depth - min_depth + 1))
        method = "full" if i % 2 == 0 else "grow"
        counter = ConstCounter()
        tree = random_tree(depth, counter, method)
        pop.append(tree)
    return pop
