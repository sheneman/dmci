# Experiment: Learned Subgraph Selection from a Program Library

## Overview

A proof-of-concept demonstrating that a trainable model can discover which compiled subgraphs to use from a **library of 16 programs**, learn input projections for each, and combine selected outputs — with L1 regularization driving sparse selection. The model correctly identifies the 3 active subgraphs from 16 candidates, learns appropriate input projections, and achieves 98,000x better extrapolation than a pure MLP.

This experiment combines **routing** (which subgraph to select) with **interfacing** (how to prepare inputs) at library scale, directly motivating the LLM future work direction: an LLM populates the library, the trainable network selects and wires the programs.

Source: `examples/hybrid_library.py`

## Task

Learn a function f(x, y) that secretly decomposes into 3 compiled subprograms selected from a library of 16:

f(x, y) = (x+y)³ + 2((x-y)² + 1) + 3(x+y) - 1

Expanded: f(x, y) = (x+y)³ + 2(x-y)² + 3(x+y) + 1, a degree-3 polynomial in two variables.

The model does not know which subgraphs are needed, what inputs to feed them, or how to combine their outputs — it must discover all of this from (input, output) training pairs.

### Program library (16 compiled subgraphs)

All programs take a single input x, compiled from Scheme source:

| Index | Name | Scheme source | Function | Nodes | Role |
|-------|------|--------------|----------|-------|------|
| 0 | x² | `(* x x)` | x² | 2 | distractor |
| **1** | **x³** | **`(let ((x2 (* x x))) (* x x2))`** | **x³** | **3** | **active** |
| **2** | **x²+1** | **`(+ (* x x) 1)`** | **x²+1** | **4** | **active** |
| 3 | x²-1 | `(- (* x x) 1)` | x²-1 | 4 | distractor |
| 4 | x⁴ | `(let ((x2 (* x x))) (* x2 x2))` | x⁴ | 3 | distractor |
| **5** | **2x** | **`(* 2 x)`** | **2x** | **3** | **active** |
| 6 | x+1 | `(+ x 1)` | x+1 | 3 | distractor |
| 7 | x-1 | `(- x 1)` | x-1 | 3 | distractor |
| 8 | x²+x | `(+ (* x x) x)` | x²+x | 3 | distractor |
| 9 | x²-x | `(- (* x x) x)` | x²-x | 3 | distractor |
| 10 | x⁴+x² | `(let ((x2 (* x x))) (+ (* x2 x2) x2))` | x⁴+x² | 4 | distractor |
| 11 | x⁴-x² | `(let ((x2 (* x x))) (- (* x2 x2) x2))` | x⁴-x² | 4 | distractor |
| 12 | 3x²+2x | `(+ (* 3 (* x x)) (* 2 x))` | 3x²+2x | 7 | distractor |
| 13 | x²-2x | `(- (* x x) (* 2 x))` | x²-2x | 5 | distractor |
| 14 | x⁶ | `(let ((x2 (* x x))) (* x2 (* x2 x2)))` | x⁶ | 4 | distractor |
| 15 | x³+x² | `(let ((x2 (* x x))) (+ (* x x2) x2))` | x³+x² | 4 | distractor |

The library includes "tempting" distractors: x²-1 is close to x²+1, x²+x contains both needed terms, 3x²+2x combines quadratic and linear components. The model must avoid these and find the minimal active set.

### Canonical decomposition (ground truth)

| Subgraph | Input projection | Output weight | Contribution |
|----------|-----------------|---------------|-------------|
| x³ (index 1) | [1, 1] → (x+y) | 1.0 | (x+y)³ |
| x²+1 (index 2) | [1, -1] → (x-y) | 2.0 | 2((x-y)²+1) |
| 2x (index 5) | [1, 1] → (x+y) | 1.5 | 3(x+y) |
| — | — | bias = -1.0 | -1 |

## Architecture

```
          ┌── proj₀  [2→1] ──→ [Frozen GNN: x²]     → o₀  ──┐
          ├── proj₁  [2→1] ──→ [Frozen GNN: x³]     → o₁  ──┤
          ├── proj₂  [2→1] ──→ [Frozen GNN: x²+1]   → o₂  ──┤
          ├── proj₃  [2→1] ──→ [Frozen GNN: x²-1]   → o₃  ──┤
(x, y) ──┼── ...                                             ├──→ Σ wᵢoᵢ + bias → output
          ├── proj₁₃ [2→1] ──→ [Frozen GNN: x²-2x]  → o₁₃ ──┤
          ├── proj₁₄ [2→1] ──→ [Frozen GNN: x⁶]     → o₁₄ ──┤
          └── proj₁₅ [2→1] ──→ [Frozen GNN: x³+x²]  → o₁₅ ──┘
              16 projections           16 frozen subgraphs          Linear(16→1)
              (trainable)              (frozen, exact)               (L1-regularized)
```

### Trainable components (49 parameters)

| Component | Shape | Parameters | Learns |
|-----------|-------|------------|--------|
| 16 input projections | 2 → 1 (no bias) each | 32 | Which linear combination of (x,y) to feed each subgraph |
| Output combination | 16 → 1 | 17 | Which subgraph outputs to use and how to combine them |
| **Total** | | **49** | |

### Pure MLP baseline (12,737 parameters)

A feedforward network (2 → 64 → 64 → 64 → 64 → 1) with ReLU activations — 260x more parameters than the hybrid model.

## Sparse Selection via L1 Regularization

The output combination layer has 16 weights (one per subgraph) plus bias. L1 regularization on these weights drives sparse selection:

**Loss** = MSE + λ · Σ|wᵢ|

where λ ramps linearly from 0 to 0.5 over the first 20% of training (warmup), then holds at 0.5. The warmup lets the model find the function before sparsity pressure kicks in.

### Selection trajectory

| Epoch | Active subgraphs | L1 | λ |
|-------|------------------|----|---|
| 0 | all 16 | 1.74 | 0.000 |
| 1000 | 15 | 5.20 | 0.250 |
| 2000 | 10 | 3.22 | 0.500 |
| 3000 | 7 | 2.08 | 0.500 |
| 4500 | 5 | 1.39 | 0.500 |
| **6000** | **3** | **0.92** | **0.500** |
| 10000 | 3 | 0.42 | 0.500 |

The pruning is gradual: the model first eliminates high-degree distractors (x⁶, x⁴, x⁴±x²), then composite functions (3x²+2x, x²-2x, x³+x²), and finally simple distractors (x+1, x-1). By epoch 6000, only the 3 true subgraphs remain — exact match with ground truth.

## Training

- **Loss**: MSE + λ · L1(output weights)
- **L1 schedule**: λ ramps 0 → 0.5 over first 2000 epochs, then holds
- **Optimizer**: Adam, lr = 3e-3
- **Epochs**: 10,000
- **Batch size**: 1024
- **Training range**: x, y each uniform in [-2, 2]

## Results

### Library selection

| Index | Name | Output weight | Status |
|-------|------|--------------|--------|
| **1** | **x³** | **+0.075** | **correctly selected** |
| **2** | **x²+1** | **+0.050** | **correctly selected** |
| **5** | **2x** | **+0.291** | **correctly selected** |
| all others | — | < 0.001 | correctly pruned |
| — | bias | +0.951 | — |

**Exact match**: the discovered active set {1, 2, 5} equals the true active set {1, 2, 5}. All 13 distractors are pruned to effectively zero weight.

### Learned projections

| Subgraph | True projection | Learned projection | True weight | Learned weight |
|----------|-----------------|-------------------|-------------|---------------|
| x³ | [1, 1] | [+2.38, +2.38] | 1.0 | +0.075 |
| x²+1 | [1, -1] | [-6.30, +6.30] | 2.0 | +0.050 |
| 2x | [1, 1] | [+5.09, +5.09] | 1.5 | +0.291 |

The projections point in the correct directions but with inflated magnitudes. The output weights are correspondingly small. This is an L1-induced distortion: the penalty minimizes output weight magnitudes, so the model shifts computation into the unpenalized projection scaling.

### Equivalent factorizations

The learned weights differ from canonical values but are **mathematically equivalent**:

**Cubic path** (x³): proj ≈ 2.38·[1, 1], so input = 2.38(x+y). Then:
- x³(2.38(x+y)) = 2.38³·(x+y)³ = 13.42·(x+y)³
- w · 13.42 = 0.075 · 13.42 = **1.001 ≈ 1.0** ✓

The cubic's degree-3 scaling gives a continuous one-parameter family: any α with w·α³ = 1 works.

**Quadratic+constant path** (x²+1): proj ≈ -6.30·[1, -1], so input = -6.30(x-y). Then:
- x²+1 at -6.30(x-y) = 39.69(x-y)² + 1
- w · 39.69 = 0.050 · 39.69 = **1.99 ≈ 2.0** ✓ (quadratic coefficient)
- w + bias = 0.050 + 0.951 = **1.001 ≈ 1.0** ✓ (constant term)

The constant term in x²+1 creates a constraint: w must absorb both the quadratic scaling and the constant contribution. With the bias as an additional degree of freedom, the model satisfies both constraints by making w small (minimizing L1) and compensating with a large projection scale and adjusted bias.

**Linear path** (2x): proj ≈ 5.09·[1, 1], so input = 5.09(x+y). Then:
- 2x at 5.09(x+y) = 10.19(x+y)
- w · 10.19 = 0.291 · 10.19 = **2.97 ≈ 3.0** ✓

### Accuracy

| Model | Trainable params | In-distribution MSE | Extrapolation MSE (2x range) |
|-------|-----------------|--------------------|-----------------------------|
| **Hybrid** | **49** | **0.002** | **0.13** |
| Pure MLP | 12,737 | 0.065 | 12,728 |
| **Ratio** | | **30x better** | **98,000x better** |

The in-distribution ratio (30x) is lower than other experiments because L1 regularization sacrifices some accuracy for sparsity. The extrapolation ratio (98,000x) is the more relevant metric — it demonstrates that the selected compiled subgraphs generalize exactly outside the training domain.

### Model complexity

| Model | Trainable params | Frozen subgraph structure | Total |
|-------|-----------------|--------------------------|-------|
| Hybrid | 49 | 59 nodes, 70 edges, 8 const floats (16 programs) | 49 trainable + frozen graph |
| Pure MLP | 12,737 | — | 12,737 trainable |

The hybrid model is 260x smaller in trainable parameters. The frozen library (59 nodes across 16 programs) encodes all polynomial computation — the trainable model only learns selection, projection, and combination.

## Significance

### 1. Sparse selection from a large library

The model correctly identifies 3 active subgraphs from 16 candidates, with all 13 distractors pruned to effectively zero weight. This is a qualitatively harder problem than the routing experiment (3 of 3): the model must search over a combinatorial space of possible subsets.

### 2. Combined routing and interfacing

Each selected subgraph receives a learned input projection (Linear 2→1). The model simultaneously discovers **which** subgraphs to use and **what inputs** to feed them — combining the capabilities demonstrated separately in the routing and interfacing experiments.

### 3. L1 regularization drives interpretable sparsity

The L1 warmup schedule (0 → 0.5 over 20% of training) produces a clean pruning trajectory: from 16 active subgraphs down to exactly 3 by epoch 6000. The selection is not an artifact of thresholding — the pruned weights are 3 orders of magnitude smaller than the active weights.

### 4. L1 distorts but preserves equivalence

L1 regularization drives output weights small and shifts computation into unpenalized projection scaling. The learned factorization is mathematically equivalent to the canonical solution (verified by the scaling analysis) but with a characteristic distortion: small weights, large projections, bias absorbing constants. This is a predictable and well-understood effect of L1 regularization.

### 5. Motivates library-scale architectures

This experiment establishes the building block for library-scale hybrid architectures: given a library of compiled programs (potentially populated by an LLM writing Scheme code), a trainable network can discover which programs to use and how to wire them. The 16-program library is a proof of concept; the architecture scales naturally to larger libraries.

## Comparison across all seven experiments

| Experiment | Capability | Trainable | Frozen structure | Baseline params | In-dist. | Extrap. |
|-----------|-----------|-----------|-----------------|----------------|---------|---------|
| Routing | Learned subgraph selection (3 of 3) | 355 | 16 nodes, 16 edges, 5 consts | 593 | 5.1x | 11,358x |
| Interfacing | Learned input projections | 19 | 12 nodes, 14 edges, 1 const | 12,865 | 290x | 417,000x |
| Recursive | Batched recursive subgraphs | 4 | 14 nodes, 15 edges, 3 consts | 8,577 | 4,167x | — |
| Composition | Multi-stage subgraph chaining | 7 | 17 nodes, 20 edges, 1 const | 8,577 | ~280,000x | ~75,000,000x |
| CNN Physics | Perceptual input + exact physics | 55,686 | 11 nodes, 10 edges, 2 consts | 60,164 | Arch. demo | — |
| **Library** | **Sparse selection (3 of 16)** | **49** | **59 nodes, 70 edges, 8 consts** | **12,737** | **30x** | **98,000x** |
| Deep Composition | 3-stage pipeline gradient flow | 6 | 8 nodes, 8 edges, 1 const | 8,577 | ~620,000x | ~29,000,000,000x |

The seven experiments progressively demonstrate:
1. **Routing**: the model selects which compiled module to use
2. **Interfacing**: the model transforms inputs for compiled modules, with gradients flowing through frozen subgraphs
3. **Recursive**: compiled modules can contain loops/recursion
4. **Composition**: compiled module outputs feed into other compiled modules in multi-stage pipelines
5. **CNN Physics**: compiled modules receive inputs from a CNN, with gradients flowing through frozen subgraphs to train convolutional filters
6. **Library**: the model discovers which programs to use from a large library via L1-regularized sparse selection
7. **Deep Composition**: gradients flow through 3 frozen subgraphs in series, resolving hard constraints imposed by intermediate constants

Together they establish that compiled GNN subgraphs integrate with diverse neural architectures — from linear layers to multi-stage pipelines to convolutional networks to library-scale selection to 3-stage deep gradient flow — with standard gradient-based training throughout.

## Visualization

See `examples/hybrid_library.png` for a four-panel figure:
1. **Training loss**: Log-scale MSE showing hybrid convergence below MLP
2. **Library weights**: Bar chart of output weights — 3 green (correctly selected), 13 gray (correctly pruned)
3. **f(x, 0.5) slice**: Extrapolation comparison showing hybrid tracks true cubic function outside training range
4. **f(0.5, y) slice**: Extrapolation showing MLP divergence while hybrid remains exact
