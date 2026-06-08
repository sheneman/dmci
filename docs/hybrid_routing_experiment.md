# Experiment: Hybrid Routing with Compiled Subgraphs

## Overview

A proof-of-concept demonstrating that a trainable neural network can incorporate frozen, deterministically compiled GNN subgraphs and learn to route inputs between them. The trainable model only needs to learn routing boundaries — the exact computation within each piece is provided for free by the compiled subgraphs.

Source: `examples/hybrid_routing.py`

## Task

Learn an unknown piecewise function f(x) where each piece is a distinct mathematical formula:

| Region | Function | Compiled Scheme source |
|--------|----------|----------------------|
| x < -1 | x² + 1 | `(+ (* x x) 1)` |
| -1 ≤ x ≤ 2 | 2x - 3 | `(- (* 2 x) 3)` |
| x > 2 | -x² + 3x | `(+ (- 0 (* x x)) (* 3 x))` |

The model receives (x, f(x)) training pairs sampled uniformly from x ∈ [-5, 5]. It does not know the boundaries at x = -1 and x = 2 — it must learn them from data.

## Compiled Subgraphs

Each Scheme program is compiled to a frozen GNN subgraph via the standard pipeline (Scheme → AST → ANF → Dataflow DAG → PyG HeteroData → message passing). The compiled subgraphs are small:

| Program | Nodes | Depth (MP rounds) |
|---------|-------|--------------------|
| x² + 1 | 4 | 2 |
| 2x - 3 | 5 | 2 |
| -x² + 3x | 7 | 3 |

All subgraph parameters are frozen (`requires_grad = False`). The subgraphs use batched evaluation (`forward_batch`) to process all training samples simultaneously.

## Architecture

### Hybrid model

```
Input x ──┬──→ [Frozen GNN: x² + 1]    → o₁ ──┐
           ├──→ [Frozen GNN: 2x - 3]    → o₂ ──┤──→ weighted sum → output
           ├──→ [Frozen GNN: -x² + 3x]  → o₃ ──┘        ↑
           │                                              │
           └──→ [Trainable MLP Router]  → softmax(w₁, w₂, w₃)
```

- **Subgraphs**: All three receive the same input x and produce exact outputs o₁, o₂, o₃ (evaluated inside `torch.no_grad()`)
- **Router**: A small MLP (1 → 16 → 16 → 3) with ReLU activations that maps x to three routing logits, passed through softmax with temperature annealing
- **Output**: Weighted combination: output = w₁o₁ + w₂o₂ + w₃o₃
- **Trainable parameters**: 355 (all in the router MLP; subgraphs contribute zero)

### Pure MLP baseline

A standard feedforward network (1 → 16 → 16 → 16 → 1) with ReLU activations and 593 trainable parameters — intentionally given more capacity than the hybrid model.

## Training

- **Loss**: MSE between predicted and true f(x)
- **Optimizer**: Adam, lr = 1e-3
- **Epochs**: 3000
- **Batch size**: 512 (freshly sampled each epoch)
- **Temperature annealing**: The softmax temperature in the router is linearly annealed from 1.0 to 0.2 over training, encouraging progressively sharper routing decisions

### Gradient flow

Gradients flow through the weighted combination (w₁o₁ + w₂o₂ + w₃o₃) back to the router MLP. The subgraph outputs o₁, o₂, o₃ act as fixed coefficients in this multiplication — the router learns which coefficient to weight up for each input region. No gradients need to flow through the frozen GNN subgraphs themselves.

## Results

### Model complexity

| Model | Trainable params | Frozen subgraph structure | Total |
|-------|-----------------|--------------------------|-------|
| Hybrid (compiled + router) | 355 | 16 nodes, 16 edges, 5 const floats | 355 trainable + frozen graph |
| Pure MLP | 593 | — | 593 trainable |

The frozen subgraph structure (node op types, edge topology, constant values) encodes the three compiled programs. These are analogous to architecture choices rather than learned weights — they define *what* is computed, while the 355 trainable parameters learn *when* to invoke each computation.

### In-distribution accuracy (x ∈ [-5, 5])

| Model | Test MSE |
|-------|----------|
| Hybrid (compiled + router) | **0.033** |
| Pure MLP | 0.168 |

The hybrid model achieves **5.1x lower error**.

### Extrapolation (x ∈ [-10, 10])

| Model | Test MSE | vs Hybrid |
|-------|----------|-----------|
| Hybrid (compiled + router) | **0.016** | — |
| Pure MLP | 186.3 | 11,358x worse |

The hybrid model's extrapolation MSE actually *decreases* compared to in-distribution (0.016 vs 0.033) because the routing boundaries at x = -1 and x = 2 fall within the training range — once past these boundaries, the compiled subgraphs provide exact computation everywhere. The MLP has no such guarantee and diverges catastrophically outside the training domain.

### Learned routing weights

The router learns near-one-hot routing with sharp transitions at the correct boundaries:

- For x < -1: w₁ ≈ 1 (selects x² + 1)
- For -1 ≤ x ≤ 2: w₂ ≈ 1 (selects 2x - 3)
- For x > 2: w₃ ≈ 1 (selects -x² + 3x)

Temperature annealing from 1.0 → 0.2 encourages this sharpening. Residual error comes primarily from the narrow transition regions around x = -1 and x = 2 where routing is not yet fully one-hot.

### Convergence

The hybrid model converges faster in the early epochs (the subgraphs immediately provide correct function values — only the routing needs to be learned). The MLP must simultaneously learn three different mathematical functions and their boundaries.

## Significance

This experiment demonstrates the core thesis of the paper:

1. **Compiled subgraphs reduce the learning problem.** The hybrid model only needs to learn 2 decision boundaries (a classification task), not 3 continuous functions (a regression task). This is a qualitative reduction in learning difficulty.

2. **Exact computation provides free generalization.** The compiled subgraphs compute correct values everywhere in R, not just on the training domain. The model generalizes perfectly in each piece — only routing must extrapolate.

3. **Fewer trainable parameters, better results.** 355 parameters (all routing) beats 593 parameters (all learned). The compiled subgraphs provide "free capacity" that doesn't count against the parameter budget.

4. **The architecture works end-to-end with standard training.** No special machinery is needed — MSE loss, Adam optimizer, and backpropagation through the weighted combination are sufficient to train the router.

## Limitations and future work

- **Routing only, no input transformation.** The current example passes x directly to all subgraphs. A more powerful architecture would also learn to transform inputs before passing them to compiled modules.

- **Independent evaluation.** All subgraphs are evaluated for every input. A gating mechanism that skips unselected subgraphs would improve efficiency for larger libraries of compiled programs.

- **Soft routing.** The softmax-based routing is inherently soft — in the transition regions, the model blends outputs from multiple subgraphs rather than hard-switching. This is adequate for continuous piecewise functions but may not suit discontinuous targets.

- **Single input variable.** Extending to multi-variable functions with multi-variable compiled subgraphs is straightforward but not yet demonstrated.

- **No composition.** Subgraph outputs are not fed into other subgraphs. See the composition experiment for multi-stage pipelines.

## Visualization

See `examples/hybrid_routing.png` for a three-panel figure:
1. **Function comparison**: True piecewise function, individual subgraph outputs, hybrid prediction, and MLP prediction over x ∈ [-7, 7]
2. **Routing weights**: Learned w₁(x), w₂(x), w₃(x) showing near-one-hot selection at correct boundaries
3. **Training loss**: Log-scale MSE over 3000 epochs for both models
