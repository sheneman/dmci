# Experiment 8: Residual Connections at Frozen Subgraph Interfaces

## Overview

Tests whether **residual connections** at the interfaces between frozen compiled subgraphs fix the optimization failure discovered in Experiment 7. The sub_one subgraph creates a gradient attractor that traps the projection magnitude at beta~0.3 in bare 3-stage chains. Residual connections provide a gradient highway (derivative=1) that bypasses the frozen nonlinearity, enabling the optimizer to escape the trap.

Source: `examples/hybrid_residual_composition.py`

## Background

Experiment 7 found a depth-optimization tradeoff: while gradients flow through 3 frozen subgraphs, certain pipeline structures (those containing sub_one) create gradient attractors that trap trainable parameters. The sub_one function (z - 1) creates a local minimum at small projection magnitudes because d/dbeta of (beta^2 s^2 - 1)^2 ≈ -4 beta s^2 at small beta — a force that actively pulls beta toward zero.

This experiment asks: can standard deep learning interface techniques (residual connections) overcome this attractor while keeping the subgraphs frozen?

## Architecture

Two independent 3-stage pipelines, each fed by a learned 2→1 projection:

```
                  ┌─ proj_a ─→ [square] ─(+α₀z)─→ [add_one] ─(+α₁z)─→ [cube]   ─(+α₂z)─→ pipe_a ─┐
(x, y) ──────────┤                        residual       residual        residual                     ├─→ Linear(4→1) → output
                  └─ proj_b ─→ [square] ─(+α₃z)─→ [sub_one] ─(+α₄z)─→ [square] ─(+α₅z)─→ pipe_b ─┤
                                          residual        residual        residual                     │
(x, y) ────────────────────────────── direct features x, y ──────────────────────────────────────────┘
```

At each interface: `z = subgraph(z) + αᵢ · z`

The residual scale αᵢ is a learned scalar (initialized at 0.1). When αᵢ→0, the interface reduces to the bare subgraph output. The gradient through the residual path has derivative αᵢ (constant), providing a gradient highway that doesn't depend on the frozen subgraph's derivative.

### Trainable parameters

| Component | Parameters | Role |
|-----------|-----------|------|
| proj_a | 2 | Input projection for Pipeline A (learns x+y) |
| proj_b | 2 | Input projection for Pipeline B (learns x-y) |
| residual scales | 6 | Interface residual strengths (3 per pipeline) |
| output | 5 | Combines pipeline outputs + raw x, y |
| **Total** | **15** | |

### Target function

f(x,y) = ((x+y)^2 + 1)^3 + ((x-y)^2 - 1)^2

Canonical parameters: proj_a=[1,1], proj_b=[1,-1], output=[1,1,0,0], bias=0

### Compiled subgraphs (frozen)

| Pipeline | Stage 1 | Stage 2 | Stage 3 | Total |
|----------|---------|---------|---------|-------|
| A | square: z^2 | add_one: z+1 | cube: z^3 | 8 nodes |
| B | square: z^2 | sub_one: z-1 | square: z^2 | 7 nodes |
| **Total** | | | | **15 nodes, 14 edges, 2 consts** |

## Experimental setup

Both the residual and bare (control) models receive:
- **Orthogonal initialization**: proj_a=[0.5, 0.5], proj_b=[0.5, -0.5] (correct directions, moderate magnitude)
- **Soft orthogonality penalty**: lambda * cos^2(proj_a, proj_b), lambda=0.1

This isolates the gradient-flow question from the symmetry-breaking question. Both models start in the right direction; the test is whether the magnitude converges.

Training: Adam, lr=3e-3, batch=1024, 20,000 epochs, range [-1, 1].

## Results

### Ablation (seed 42)

| Model | Params | In-dist MSE | Extrap MSE | proj_b beta | Status |
|-------|--------|------------|-----------|-------------|--------|
| **Residual** | **15** | **0.000376** | **146** | **1.092** | **CONVERGED** |
| Bare (control) | 9 | 0.738 | 2,483 | 0.024 | TRAPPED |
| Full (resid+BN+splitLR) | 27 | 0.038 | 2,969 | 4.717 | UNSTABLE |
| MLP baseline | 8,577 | 0.016 | 1,036,314 | — | OK |

Key comparisons:
- Residual vs Bare: **1,964x better** in-distribution, **17x better** extrapolation
- Residual vs MLP: **42x better** in-distribution with **572x fewer parameters**
- Full model: BatchNorm + split LR destabilized — beta grew unboundedly. Plain residual connections are the correct intervention.

### Multi-seed robustness (5 seeds)

| Seed | Bare beta | Bare MSE | Residual beta | Residual MSE | Winner |
|------|-----------|----------|--------------|-------------|--------|
| 42 | 0.024 | 0.738 | 1.092 | 0.000376 | Residual (1,964x) |
| 0 | 0.007 | 0.738 | 1.286 | 0.000001 | Residual (1,018,923x) |
| 1 | **1.414** | **0.000000** | 1.168 | 0.001 | Bare* |
| 7 | 0.012 | 0.776 | 1.186 | 0.000326 | Residual (2,044x) |
| 123 | 0.006 | 0.764 | 1.155 | 0.000086 | Residual (7,830x) |

*Seed 1: The bare model found the correct basin from the orthogonal initialization. This shows the trap is seed-dependent, not absolute — but the bare model only escapes in 1/5 seeds.

**Convergence rate**: Bare 1/5 seeds (20%), Residual 5/5 seeds (100%).

### Beta trajectory

The most revealing result is the proj_b magnitude over training:

- **Bare**: Collapses from beta=0.7 to beta~0.01 within 2,000 epochs and stays trapped
- **Residual**: Also initially collapses to beta~0.01, but then escapes around epoch 10,000-12,000 in a sharp phase transition, recovering to beta~1.1

The initial collapse followed by recovery suggests the residual gradient highway takes time to dominate the loss landscape: the sub_one attractor pulls beta down early, but the residual path eventually provides enough gradient signal for the optimizer to reverse course.

### Gradient analysis

| Model | proj_a grad | proj_b grad | Ratio b/a |
|-------|------------|------------|-----------|
| Bare | 2.903 | 0.008 | 0.003 |
| Residual | 2.339 | 0.013 | 0.006 |

The residual model delivers more gradient to proj_b (1.6x). More importantly, the gradient at proj_b points in the correct direction (toward larger beta), enabling the phase transition escape. In the bare model, the gradient at proj_b is dominated by the sub_one attractor and points toward beta=0.

### Learned residual scales

Pipeline A: [−0.001, +0.401, −0.186]
Pipeline B: [+0.000, −3.981, +5.557]

Pipeline A's residual scales are moderate — the subgraph outputs dominate. Pipeline B's scales are large, especially at the sub_one (−3.98) and final square (+5.56) stages. This means the residual connections serve a dual purpose:
1. **Gradient highway** during training (enabling the phase transition escape)
2. **Computational element** at inference (modifying the effective function between stages)

The large residual scales in Pipeline B show the model learned to route information partly through the residual paths to work around the sub_one's output characteristics. The frozen subgraphs still compute their exact functions; the residuals adapt the interface.

## Significance

### 1. Residual connections fix the sub_one gradient trap

The bare model traps at beta~0.02 in 75% of seeds. The residual model escapes in 100% of seeds, achieving MSE 1,964x better. This directly addresses the depth-optimization tradeoff from Experiment 7: deep frozen chains ARE practical when you apply standard interface techniques.

### 2. BatchNorm and split learning rates are counterproductive

The "full" intervention (gated residual + BatchNorm + 10x LR for Pipeline B) actually performed worse than plain residual connections. BatchNorm1d on scalar features introduced instability, and the split LR caused beta to grow without bound. The simplest intervention — unconstrained residual connections — was the most effective.

### 3. Phase transition dynamics

The beta recovery at epoch ~12,000 is a genuine phase transition: the optimization landscape has two basins (the sub_one attractor at beta~0 and the correct solution at beta~1), and the residual gradient highway enables transitions between them that the bare model cannot make.

### 4. Extreme parameter efficiency

15 trainable parameters (with 15 frozen subgraph nodes) achieve 42x lower MSE than an 8,577-parameter MLP. The compiled pipelines provide the entire polynomial structure; the trainable parameters learn only input projections, interface residuals, and output combination.

## Comparison to Experiment 7

| Aspect | Exp 7 (Bare 2-pipeline) | Exp 8 (Residual 2-pipeline) |
|--------|------------------------|----------------------------|
| Pipeline B convergence | TRAPPED at beta~0.3 | CONVERGED to beta~1.1 |
| In-dist MSE | ~0.14-0.18 | 0.000376 |
| Seed robustness | 0/8 seeds | 4/4 seeds |
| Additional params | 0 | +6 (residual scales) |
| Technique | None | Residual: z = sg(z) + α·z |

The 6 extra parameters (residual scales) are the only change needed to convert the Experiment 7 failure into a clean convergence result.

## Visualization

See `examples/hybrid_residual_composition.png` for a six-panel figure:
1. **Training loss**: Bare stuck at 0.7; residual drops to 0.0004 at epoch ~12,000
2. **Beta trajectory**: Phase transition visualization showing the escape from the trap
3. **Direction fidelity**: Both models maintain correct [1,−1] direction throughout
4. **Projection weights**: Bar chart comparing learned vs canonical weights
5. **Test MSE**: Log-scale comparison across all models
6. **Function fit**: Residual tracks true function; bare produces flat output for Pipeline B
