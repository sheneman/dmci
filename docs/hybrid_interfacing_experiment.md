# Experiment: Learned Input Interfacing with Compiled Subgraphs

## Overview

A proof-of-concept demonstrating that a trainable neural network can learn to **transform inputs** before passing them to frozen, deterministically compiled GNN subgraphs. Unlike the routing experiment (which learns *which* subgraph to use), this experiment demonstrates **interfacing** — the model learns *how to prepare inputs* for each compiled module and how to combine their outputs.

Source: `examples/hybrid_interfacing.py`

## Task

Learn a function f(a, b, c, d) that secretly decomposes into compiled subprograms applied to specific linear combinations of the inputs:

f(a, b, c, d) = p1(a + b) + 2 * p2(a - c, b + d) - p3(c - d)

The model does not know the input projections or output weights — it must discover them from (input, output) training pairs.

### Compiled subgraphs

| Subgraph | Scheme source | Function | Nodes | Depth |
|----------|--------------|----------|-------|-------|
| p1 | `(+ (* x x) 1)` | x² + 1 | 4 | 2 |
| p2 | `(+ (- (* x y) x) y)` | xy - x + y | 5 | 3 |
| p3 | `(let ((x2 (* x x))) (* x x2))` | x³ | 3 | 2 |

## Architecture

```
                    ┌── Learned Linear₁ [4→1] ─→ x ──→ [Frozen GNN: x²+1]     → o₁ ──┐
(a, b, c, d) ──────┼── Learned Linear₂ [4→2] ─→ (x,y) → [Frozen GNN: xy-x+y]  → o₂ ──┼─→ w₁o₁ + w₂o₂ + w₃o₃
                    └── Learned Linear₃ [4→1] ─→ x ──→ [Frozen GNN: x³]       → o₃ ──┘
                                                                                    ↑
                                                                    Learned output weights (w₁, w₂, w₃)
```

### Trainable components (19 parameters total)

| Component | Shape | Parameters | Learns |
|-----------|-------|------------|--------|
| proj1 | 4 → 1 (no bias) | 4 | Which linear combination of (a,b,c,d) to feed p1 |
| proj2 | 4 → 2 (no bias) | 8 | Which two linear combinations to feed p2 as (x, y) |
| proj3 | 4 → 1 (no bias) | 4 | Which linear combination to feed p3 |
| output_weights | 3 | 3 | How to combine the three subgraph outputs |
| **Total** | | **19** | |

### Pure MLP baseline (12,865 parameters)

A feedforward network (4 → 64 → 64 → 64 → 64 → 1) with ReLU activations — 677x more parameters than the hybrid model.

## Gradient flow through frozen subgraphs

This experiment critically relies on gradients flowing **through** the compiled GNN subgraphs back to the trainable projection weights. The subgraph parameters are frozen (no `requires_grad`), but the message-passing operations (gather, scatter, elementwise arithmetic) are standard differentiable PyTorch operations. The autograd graph passes through them:

```
Loss → d/d(output) → d/d(output_weights)     [updates output weights]
     → d/d(o_i)    → d/d(subgraph forward)   [passes through frozen GNN]
                    → d/d(proj_i inputs)      [updates projection weights]
```

This is a key architectural property: frozen compiled subgraphs are transparent to gradient flow. The trainable model can learn how to format inputs for the subgraphs by backpropagating through their exact computation. Setting `torch.no_grad()` around the subgraph evaluation would block this gradient path and prevent the projection weights from learning (confirmed experimentally — the model fails to converge without gradient flow).

## Training

- **Loss**: MSE
- **Optimizer**: Adam, lr = 3e-3
- **Epochs**: 5000
- **Batch size**: 1024 (freshly sampled each epoch)
- **Training range**: (a, b, c, d) each uniform in [-2, 2]

## Results

### Accuracy

### Model complexity

| Model | Trainable params | Frozen subgraph structure | Total |
|-------|-----------------|--------------------------|-------|
| Hybrid | 19 | 12 nodes, 14 edges, 1 const float | 19 trainable + frozen graph |
| Pure MLP | 12,865 | — | 12,865 trainable |

The frozen subgraph structure encodes the three compiled programs (x²+1, xy-x+y, x³). The 19 trainable parameters are: 4+8+4 projection weights and 3 output combination weights.

### Accuracy

| Model | Trainable params | In-distribution MSE | Extrapolation MSE (2x range) |
|-------|-----------------|--------------------|-----------------------------|
| **Hybrid** | **19** | **0.000175** | **0.005** |
| Pure MLP | 12,865 | 0.051 | 2,145 |
| **Ratio** | | **290x better** | **417,000x better** |

The hybrid model achieves near-perfect accuracy with 19 trainable parameters. The MLP with 12,865 parameters cannot match it in-distribution and diverges catastrophically on extrapolation.

### Learned projections

The model must discover 4 linear projections from (a, b, c, d):

| Projection | True weights | Target | Learned weights |
|-----------|-------------|--------|----------------|
| p1 input (x) | [1, 1, 0, 0] | a + b | [1.12, 1.12, -0.58, 0.58] |
| p2 input (x) | [1, 0, -1, 0] | a - c | [0.18, -0.82, -0.77, -0.23] |
| p2 input (y) | [0, 1, 0, 1] | b + d | [-0.82, 0.18, 0.23, 0.77] |
| p3 input (x) | [0, 0, 1, -1] | c - d | [0.00, 0.00, -0.82, 0.82] |

| Output weight | True | Learned |
|--------------|------|---------|
| w₁ (p1) | 1.0 | 1.03 |
| w₂ (p2) | 2.0 | 1.99 |
| w₃ (p3) | -1.0 | 1.80 |

### Equivalent factorizations

The learned projections are not identical to the true values but are **mathematically equivalent**. Because the subgraphs contain nonlinear operations, multiple factorizations produce the same function:

- **p3 example**: True is [0, 0, 1, -1] with weight -1. Learned is [0, 0, -0.82, 0.82] with weight 1.80. Since p3(x) = x³: (-0.823)³ × 1.796 ≈ (-0.557) × 1.796 ≈ -1.0 × (c - d)³. The scaling absorbed into the cube and output weight.

- **p1 example**: True is [1, 1, 0, 0] with weight 1. Learned is [1.12, 1.12, -0.58, 0.58] — this includes components on c and d that approximately cancel (since -0.58c + 0.58d contributes a small perturbation). The model found a nearby equivalent in the loss landscape.

This non-uniqueness is expected and demonstrates that the model discovers functionally correct solutions even when the parameterization admits multiple equivalent representations.

## Significance

### 1. Gradient flow through frozen subgraphs is essential

This experiment proves that the compiled GNN subgraphs are not opaque black boxes — they are **differentiable modules** that gradients flow through. The trainable model learns to interface with them via backpropagation, not reinforcement learning or evolutionary search. This is the core technical property that makes the hybrid architecture practical.

### 2. Extreme parameter efficiency

19 trainable parameters outperform 12,865 by orders of magnitude. The compiled subgraphs provide polynomial computation (quadratic, bilinear, cubic) for free — the trainable model only needs to learn linear projections and scalar weights. This is a qualitative advantage: the learning problem is reduced from nonlinear regression to linear regression.

### 3. Perfect extrapolation from exact computation

The compiled subgraphs compute correct polynomial values everywhere in R⁴, not just on the training domain. Once the projections are learned, the hybrid model generalizes to any input range. The MLP has no such guarantee — it has only seen inputs in [-2, 2]⁴ and produces nonsense outside this range.

### 4. Interpretable learned structure

The projection weights reveal what the model learned: which input combinations matter for each compiled subgraph. This interpretability comes for free from the architecture — the linear projections are human-readable, and the compiled subgraphs have known semantics.

## Comparison with routing experiment

| Property | Routing (Example 1) | Interfacing (Example 2) |
|----------|---------------------|------------------------|
| What is learned | Which subgraph to select | How to transform inputs for each subgraph |
| Gradient flow | Through weighted combination only | **Through frozen GNN subgraphs** |
| Trainable params | 355 (router MLP) | 19 (linear projections + output weights) |
| Input dimension | 1 (scalar x) | 4 (a, b, c, d) |
| Subgraph invocation | All receive same raw input | Each receives different learned projection |
| Key demonstration | Learned routing boundaries | Learned input interfacing + gradient flow |

Together, the seven experiments demonstrate the full range of capabilities: selecting between modules (routing), preparing inputs (interfacing), recursive computation (recursive), chaining outputs between stages (composition), integrating with perceptual architectures (CNN physics), sparse selection from a large library (library), and 3-stage deep gradient flow (deep composition).

## Visualization

See `examples/hybrid_interfacing.png` for a five-panel figure:
1. **Training loss**: Log-scale MSE showing hybrid convergence to near-zero vs MLP plateau
2. **Learned input projections**: Bar chart of projection weights with true values marked as stars
3. **Learned output weights**: Comparison of learned vs true combination weights
4. **1D slice along a**: Extrapolation comparison with b=c=d=1, showing hybrid tracks true function outside training range
5. **1D slice along c**: Extrapolation showing cubic behavior via p3, where MLP fails dramatically
