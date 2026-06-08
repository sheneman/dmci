# Experiment: Recursive Compiled Subgraphs in Hybrid Architecture

## Overview

A proof-of-concept demonstrating that **recursive programs** compiled to GNN subgraphs can participate in a hybrid trainable architecture with batched execution. The compiler's tail-call optimization (TCO) converts the recursive definition to loop/recur, and a new batched loop evaluator (padded iteration with masking) enables parallel evaluation across the batch.

This experiment required implementing batched loop execution — a new feature in the compiler. Prior to this work, `forward_batch` only supported DAG programs.

Source: `examples/hybrid_recursive.py`

## Task

Learn a function f(n, x) that combines a recursive computation (factorial) with polynomial terms:

f(n, x) = 0.01 * factorial(n) + x² + 2x - 1

where n ∈ {0, 1, 2, ..., 7} is a non-negative integer and x ∈ [-2, 2] is continuous.

The scaling factor 0.01 on factorial tames the dynamic range while keeping the factorial term significant: 0.01 * 7! = 50.4, comparable to the polynomial terms at the edges of the x range.

### Compiled subgraphs

| Subgraph | Scheme source | Type | Nodes |
|----------|--------------|------|-------|
| factorial | `(letrec ((fact (lambda (n acc) (if (= n 0) acc (fact (- n 1) (* acc n)))))) (fact n 1))` | Recursive (TCO → loop/recur) | 3 (outer) + 9 (loop body) |
| x² | `(* x x)` | DAG | 2 |

The factorial program is written as a tail-recursive `letrec`. The compiler's TCO pass detects the self-tail-call and converts it to an equivalent `loop/recur` structure. The batched loop evaluator then executes this loop across all batch elements simultaneously, with a boolean mask tracking which elements have terminated.

## Architecture

```
n ──→ round + clamp [0,7] ──→ [Frozen GNN: factorial] → o_fact ──┐
                                                                   ├──→ Linear(3→1) → output
x ──→ [Frozen GNN: x²] ──────────────────────────────→ o_sq   ──┤
                                                                   │
x ─────────────────────────────────────────────────────────────────┘
```

### Trainable components (4 parameters)

| Component | Shape | Parameters | Learns |
|-----------|-------|------------|--------|
| Linear weight | 3 → 1 | 3 | Combination weights: w(fact), w(x²), w(x) |
| Linear bias | 1 | 1 | Additive constant |
| **Total** | | **4** | |

The model receives three features — factorial(n), x², and x — and learns a single linear combination. There are no hidden layers; the subgraphs provide all the nonlinear computation.

### Pure MLP baseline (8,577 parameters)

A feedforward network (2 → 64 → 64 → 64 → 1) with ReLU activations — 2,144x more parameters than the hybrid model.

## Batched Loop Execution (new feature)

This experiment motivated implementing batched loop execution in the compiler. The approach:

1. **Initialization**: All batch elements start with their respective loop parameters (e.g., different n values for factorial, all with acc=1).

2. **Parallel body evaluation**: Each iteration evaluates the loop body DAG with feature tensors of shape `[N_body_nodes, batch_size]`, using the same message-passing machinery as DAG batching.

3. **Masked termination**: The body's root `if` node produces a condition vector. Elements where the condition triggers the non-recur branch are marked as terminated. Their results are stored and they are excluded from future iterations via a boolean mask.

4. **Parameter update**: For elements that recur, the new loop parameters are extracted from the body evaluation and written back (masked to avoid overwriting terminated elements).

5. **Convergence**: The loop exits when all batch elements have terminated, or raises an error after `max_iter` iterations.

For factorial with n ∈ {0..7}, the maximum iteration count is 7. Elements with smaller n terminate earlier and are frozen via the mask.

## Training

- **Loss**: MSE
- **Optimizer**: Adam, lr = 1e-2
- **Epochs**: 3000
- **Batch size**: 512
- **n values**: uniformly sampled integers from {0, 1, ..., 7}
- **x values**: uniformly sampled from [-2, 2]

Note: The factorial subgraph requires integer inputs. The model applies `round(clamp(n, 0, 7))` before passing to the subgraph. Since n is already provided as an integer in the training data, this is a no-op during training but ensures robustness.

## Results

### Learned weights

| Parameter | True value | Learned value |
|-----------|-----------|---------------|
| w(factorial) | 0.01 | 0.0100 |
| w(x²) | 1.0 | 0.9901 |
| w(x) | 2.0 | 2.0000 |
| bias | -1.0 | -0.9779 |

All four parameters converged to within 0.03 of the true values. With only 4 trainable parameters learning a linear combination, the optimization landscape is convex and convergence is fast.

### Accuracy

### Model complexity

| Model | Trainable params | Frozen subgraph structure | Total |
|-------|-----------------|--------------------------|-------|
| Hybrid | 4 | 14 nodes, 15 edges, 3 const floats | 4 trainable + frozen graph |
| Pure MLP | 8,577 | — | 8,577 trainable |

The frozen structure encodes the factorial program (3 outer nodes + 9 loop body nodes, including the recursive loop logic) and x² (2 nodes). The 4 trainable parameters are 3 linear combination weights and 1 bias.

### Accuracy

| Model | Trainable params | Test MSE |
|-------|-----------------|----------|
| **Hybrid** | **4** | **0.000208** |
| Pure MLP | 8,577 | 0.868 |
| **Ratio** | | **4,167x better** |

### Why the MLP struggles

The MLP must learn factorial — a function that maps {0,1,2,3,4,5,6,7} to {1,1,2,6,24,120,720,5040} — from data using ReLU activations. This is fundamentally hard:

1. **Super-exponential growth**: The values span 4 orders of magnitude. ReLU networks approximate smooth functions well but struggle with the explosive growth of factorial.

2. **Discrete input**: n takes only 8 integer values. The MLP must interpolate between them, producing arbitrary (wrong) values for non-integer n.

3. **Memorization vs. generalization**: Even if the MLP memorizes the 8 factorial values perfectly, it must simultaneously fit the continuous polynomial terms in x — competing objectives for a finite-capacity network.

The compiled factorial subgraph sidesteps all of this: it computes exact factorial values by executing the recursive program via loop unrolling.

## Significance

### 1. Recursive programs work in the hybrid architecture

This is the first demonstration that compiled programs containing recursion (converted to loops via TCO) can participate in batched training. The factorial subgraph executes a variable number of loop iterations per batch element, handled transparently by the masked batched evaluator.

### 2. The compiler pipeline is end-to-end

The path from Scheme source to trainable hybrid component is:
```
Scheme source → letrec → TCO → loop/recur → compile → GNN subgraph → batched loop eval → hybrid model
```
No manual intervention is needed. The same compilation pipeline that handles DAG programs extends to recursive programs.

### 3. Extreme parameter efficiency for exact computation

4 trainable parameters achieve 4,167x lower error than 8,577. The factorial computation (which dominates the function's complexity) is provided for free by the compiled subgraph. The trainable model only learns a linear combination — the simplest possible learning task.

### 4. Factorial is impossible to learn from data

No amount of MLP capacity can reliably learn factorial from a finite training set. The function's super-exponential growth, combined with the discrete input domain, makes it a pathological case for function approximation. The compiled subgraph computes it exactly by definition.

## Comparison across all seven experiments

| Experiment | Capability | Trainable | Frozen structure | Baseline params | In-dist. | Extrap. |
|-----------|-----------|-----------|-----------------|----------------|---------|---------|
| Routing | Learned subgraph selection (3 of 3) | 355 | 16 nodes, 16 edges, 5 consts | 593 | 5.1x | 11,358x |
| Interfacing | Learned input projections + gradient flow | 19 | 12 nodes, 14 edges, 1 const | 12,865 | 290x | 417,000x |
| Recursive | Batched recursive subgraphs | 4 | 14 nodes, 15 edges, 3 consts | 8,577 | 4,167x | — |
| Composition | Multi-stage subgraph chaining | 7 | 17 nodes, 20 edges, 1 const | 8,577 | ~280,000x | ~75,000,000x |
| CNN Physics | Perceptual input + exact physics | 55,686 | 11 nodes, 10 edges, 2 consts | 60,164 | Arch. demo | — |
| Library | Sparse selection (3 of 16) | 49 | 59 nodes, 70 edges, 8 consts | 12,737 | 30x | 98,000x |
| Deep Composition | 3-stage pipeline gradient flow | 6 | 8 nodes, 8 edges, 1 const | 8,577 | ~620,000x | ~29,000,000,000x |

The frozen subgraph structure (node op types, edge topology, constant values) encodes the compiled programs. These are analogous to architecture choices — they define what is computed — while the trainable parameters learn how to use the compiled modules. The MLP must encode both what to compute and how to combine in its trainable weights alone.

The seven experiments progressively demonstrate:
1. **Routing**: the model selects which compiled module to use
2. **Interfacing**: the model transforms inputs for compiled modules, with gradients flowing through frozen subgraphs
3. **Recursive**: compiled modules can contain loops/recursion, executing with variable iteration counts across the batch
4. **Composition**: compiled module outputs feed into other compiled modules, with gradient flow through multiple frozen subgraphs in series
5. **CNN Physics**: compiled modules receive inputs from a CNN, with gradients flowing through frozen subgraphs to train convolutional filters
6. **Library**: the model discovers which programs to use from a large library via L1-regularized sparse selection
7. **Deep Composition**: gradients flow through 3 frozen subgraphs in series, resolving hard constraints imposed by intermediate constants

Together they establish that compiled GNN subgraphs integrate with diverse neural architectures — from linear layers to multi-stage pipelines to convolutional networks to library-scale selection to 3-stage deep gradient flow — with standard gradient-based training throughout.

## Visualization

See `examples/hybrid_recursive.png` for a four-panel figure:
1. **Training loss**: Hybrid converges to near-zero; MLP plateaus with periodic instability
2. **Learned weights**: Bar chart comparing learned vs true combination weights (near-perfect match)
3. **f(n, x=1)**: Bar chart showing factorial growth — hybrid matches exactly at all n, MLP fails at large n
4. **f(5, x) slice**: Extrapolation comparison showing hybrid tracks the true curve outside training range
