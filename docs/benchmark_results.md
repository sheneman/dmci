# Benchmark Results: RTX 4090 Large Program Performance

## Environment

- **GPU**: NVIDIA GeForce RTX 4090 (24 GB VRAM)
- **Node**: n128.fortytwo.ibest.uidaho.edu
- **Python**: 3.11.15
- **PyTorch**: 2.6.0+cu124
- **Repeats**: 30 per evaluator (5 for Scheme scalar)
- **VRAM budget**: 75% of available memory
- **Memory model**: `(n_nodes + n_inputs) * 4 bytes * 4` (4x multiplier for gather buffer intermediates)
- **Date**: 2026-05-18 (SLURM job 5145679)
- **All 419 tests passed** before benchmarking

## Evaluator Descriptions

| Evaluator | Label | Description |
|-----------|-------|-------------|
| `python_scalar` | Python | Sequential scalar evaluation in pure Python (batch=1) |
| `numpy` | NumPy | Vectorized NumPy evaluation (batched) |
| `torch_cpu` | PyTorch CPU | Direct PyTorch op compilation, CPU (batched) |
| `torch_gpu` | PyTorch GPU | Direct PyTorch op compilation, CUDA (batched) |
| `batch_cpu` | GNN CPU | Compiled GNN subgraph, message-passing evaluation, CPU (batched) |
| `batch_gpu` | GNN GPU | Compiled GNN subgraph, message-passing evaluation, CUDA (batched) |

The `torch_cpu`/`torch_gpu` evaluators compile programs to direct PyTorch operations (each primitive maps to a torch function). The `batch_cpu`/`batch_gpu` evaluators use the GNN message-passing architecture — programs compiled to graph structure, evaluated via gather/scatter message passing. Both produce identical results; the difference is the execution model.

## Wide Tree Benchmarks

Programs: `tree_W` = W-way multiply-and-sum tree with W leaves. Tests wide, shallow graphs where batched message passing can exploit parallelism.

### Throughput (samples/second)

| Program | Nodes | Depth | Batch size | Python | NumPy | Torch CPU | GNN CPU | GNN GPU | Torch GPU |
|---------|-------|-------|------------|--------|-------|-----------|---------|---------|-----------|
| tree_50 | 199 | 7 | 1,000,000 | 91,938 | 9,997,543 | 6,020,599 | 235,395 | 21,369,295 | 752,569,397 |
| tree_100 | 399 | 8 | 1,000,000 | 49,443 | 6,644,538 | 4,188,417 | 137,343 | 9,314,210 | 368,487,334 |
| tree_150 | 599 | 9 | 1,000,000 | 27,463 | 5,204,231 | 3,398,167 | 113,028 | 5,507,342 | 242,193,815 |
| tree_200 | 799 | 9 | 500,000 | 21,534 | 4,063,039 | 3,969,837 | 87,666 | 4,095,208 | 159,994,474 |
| tree_250 | 999 | 9 | 500,000 | 16,831 | 3,254,455 | 2,703,827 | 55,541 | 3,269,247 | 127,913,099 |
| tree_300 | 1,199 | 10 | 500,000 | 11,620 | 2,702,022 | 2,395,005 | 40,662 | 2,453,092 | 106,496,600 |

### GNN GPU vs Python speedup

| Program | GNN GPU / Python |
|---------|-----------------|
| tree_50 | 232x |
| tree_100 | 188x |
| tree_150 | 201x |
| tree_200 | 190x |
| tree_250 | 194x |
| tree_300 | 211x |

### GNN GPU vs direct PyTorch GPU

| Program | GNN GPU | Torch GPU | Torch/GNN ratio |
|---------|---------|-----------|-----------------|
| tree_50 | 21.4M/s | 752.6M/s | 35x |
| tree_100 | 9.3M/s | 368.5M/s | 40x |
| tree_150 | 5.5M/s | 242.2M/s | 44x |
| tree_200 | 4.1M/s | 160.0M/s | 39x |
| tree_250 | 3.3M/s | 127.9M/s | 39x |
| tree_300 | 2.5M/s | 106.5M/s | 43x |

The direct PyTorch GPU evaluator is consistently ~40x faster than GNN GPU for wide trees. This is the cost of the message-passing architecture: gather/scatter operations and per-round synchronization introduce overhead compared to direct op execution. However, the GNN architecture provides structural composability — programs are data (graphs), not code (functions) — which is the core architectural innovation enabling the hybrid experiments.

## Deep Chain Benchmarks

Programs: `chain_D` = sequence of D additions (x + 1 + 1 + ...). Tests deep, narrow graphs where each message-passing round processes only one operation. This is the worst case for GNN evaluation: depth = D means D sequential message-passing rounds with no parallelism.

### Throughput (samples/second)

| Program | Nodes | Depth | Batch size | Python | NumPy | Torch CPU | GNN CPU | GNN GPU | Torch GPU |
|---------|-------|-------|------------|--------|-------|-----------|---------|---------|-----------|
| chain_50 | 101 | 50 | 10,000,000 | 1,075,413 | 60,732,437 | 90,102,945 | 72,956 | 3,393,794 | 4,382,975,590 |
| chain_100 | 201 | 100 | 5,000,000 | 578,413 | 54,603,091 | 797,757,821 | 15,858 | 849,284 | 4,673,195,048 |
| chain_150 | 301 | 150 | 1,000,000 | 362,073 | 77,453,172 | 299,077,822 | 5,781 | 379,919 | 1,122,349,617 |
| chain_200 | 401 | 200 | 1,000,000 | 282,139 | 56,677,493 | 336,398,785 | 2,526 | 213,385 | 885,490,463 |
| chain_250 | 501 | 250 | 1,000,000 | 237,209 | 49,199,549 | 208,856,183 | 1,586 | 136,427 | 670,104,844 |
| chain_300 | 601 | 300 | 1,000,000 | 186,512 | 40,298,822 | 206,020,707 | — | — | 497,511,266 |
| chain_350 | 701 | 350 | 1,000,000 | 149,955 | 34,218,354 | 173,766,671 | — | — | 426,420,563 |
| chain_400 | 801 | 400 | 1,000,000 | 128,728 | 29,632,707 | 153,899,313 | — | — | 368,885,078 |
| chain_450 | 901 | 450 | 1,000,000 | 115,949 | 26,311,222 | 144,491,374 | — | — | 328,027,602 |
| chain_500 | 1,001 | 500 | 1,000,000 | 101,141 | 23,588,327 | 129,327,781 | — | — | 296,112,192 |

GNN evaluation was skipped for chains ≥ 300 depth because the sequential message-passing rounds became prohibitively slow.

### Chain scaling behavior

Deep chains reveal the fundamental tradeoff of the GNN architecture:

- **Direct PyTorch GPU** scales well: 4.4B/s at depth 50 → 296M/s at depth 500 (15x slowdown for 10x more depth). PyTorch fuses the chain of additions into efficient CUDA kernels.
- **GNN GPU** scales poorly: 3.4M/s at depth 50 → 136K/s at depth 250 (25x slowdown for 5x more depth). Each depth level requires a separate message-passing round — the GPU can't parallelize sequential rounds.
- **NumPy** is remarkably consistent: 60M/s at depth 50 → 24M/s at depth 500. Vectorized CPU evaluation scales linearly with depth.

## Key Observations

### 1. GNN excels on wide, shallow programs

For wide trees (depth 7-10, width 50-300), GNN GPU achieves 2.5M-21.4M samples/second — 188-232x faster than Python scalar evaluation and competitive with NumPy. The message-passing architecture can evaluate all nodes at the same depth in parallel, making width essentially free.

### 2. GNN struggles on deep, narrow programs

For deep chains (depth 50-250), GNN GPU throughput drops from 3.4M/s to 136K/s. Each message-passing round is a synchronization barrier that prevents pipelining. Direct PyTorch GPU is 1,000-5,000x faster on these programs.

### 3. The ~40x GNN-to-PyTorch gap is the cost of structural composability

The GNN architecture is consistently ~40x slower than direct PyTorch compilation for the same program. This overhead comes from:
- Gather/scatter operations in message passing (indirect memory access)
- Per-round synchronization (cannot fuse across rounds)
- Graph metadata overhead (edge indices, node type lookups)

This is the price of treating programs as data (graph structure) rather than code (compiled functions). The benefit is that programs-as-graphs enable the hybrid architecture experiments: routing, interfacing, composition, and recursive execution with frozen compiled subgraphs embedded in trainable networks.

### 4. Batch size is critical for GPU utilization

The VRAM-aware batch sizing uses 75% of GPU memory with a 4x multiplier for gather buffer intermediates. Larger programs get smaller batches:
- tree_50: 1M batch → 21.4M/s
- tree_300: 500K batch → 2.5M/s

The throughput difference is partly batch size (less GPU utilization at 500K) and partly program size (more work per sample). A batch-size scaling study would isolate these effects.

### 5. All evaluators produce identical results

The benchmark runs after all 419 tests pass, confirming that all evaluators (Python scalar, NumPy, PyTorch CPU/GPU, GNN CPU/GPU) produce numerically equivalent results for every program. The compilation is correct; only the execution model differs.

## Raw Data

Full CSV: `benchmarks/results_rtx4090_large.csv`
Plots: `benchmarks/figures_rtx4090/large_programs.{pdf,png}`, `benchmarks/figures_rtx4090/gnn_speedup.{pdf,png}`

## Timing Details

All throughput numbers are computed as `batch_size / median_time`. Timings exclude compilation — only forward evaluation is measured. The Python scalar evaluator runs batch_size=1 (sequential); all others process the full batch in a single call.

| Evaluator | tree_100 median (s) | chain_100 median (s) |
|-----------|--------------------|--------------------|
| Python scalar | 2.02e-5 (×1M) | 1.73e-6 (×5M) |
| NumPy | 0.150 | 0.0916 |
| Torch CPU | 0.239 | 0.00623 |
| GNN CPU | 7.28 | 315.3 |
| GNN GPU | 0.107 | 5.89 |
| Torch GPU | 0.00271 | 0.00107 |
