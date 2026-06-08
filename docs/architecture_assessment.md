# Neural Compiler: Embedding Deterministic Programs as Subgraphs in Trainable Graph Networks

## Architectural Assessment

## Core Concept

Trainable neural networks containing deterministically programmed (frozen) subcomponents compiled from a high-level language (Scheme). The overall model includes these as internal tools — the compiled subgraphs are frozen, but the encapsulating trainable model learns when and how to invoke them. A GNN serves as the compilation target, making compiled programs first-class citizens of the learned representation space.

## Why GNN (not direct PyTorch compilation)

A Scheme program can be compiled to either a GNN subgraph or direct PyTorch tensor operations. Direct PyTorch is ~40x faster for execution, but the GNN formulation is architecturally superior for the hybrid trainable/compiled vision:

### Programs as data, not code

Direct PyTorch compiled modules are opaque functions — the outer model calls them, gets results, gradients flow through, but architecturally they are black boxes. A frozen `nn.Module` is standard practice, not a new architecture.

GNN compiled modules are **data** — graphs with nodes, edges, and features living in the same space as the trainable model. This is the fundamental distinction.

### Homogeneous interface

The frozen compiled subgraph and the trainable graph share the same representation. Message passing flows naturally across the frozen/trainable boundary. The trainable region literally surrounds the frozen region in one unified graph — no special calling convention, no interface mismatch.

### Structural composability

The outer model can learn to connect, compose, or route between multiple compiled subgraphs by manipulating edges — not by writing code, but by learning graph structure. Multiple compiled programs can coexist as subgraphs within a single larger graph.

### Inspectability

The trainable model can "see" the structure of the compiled module (node types, connectivity, depth) and learn to reason about it. Direct PyTorch gives only input→output.

### Intermediate observability

The message-passing rounds create intermediate representations at every node in the compiled subgraph. The trainable parts of the network can observe and learn from these intermediates. Direct PyTorch exposes only the final output.

### Performance overhead reframed

The 40x overhead vs direct compilation matters less in this context because the compiled subgraph is a component of a larger model, not the whole execution engine. Training loop overhead already dominates wall-clock time.

## What the trainable outer model learns

- **Routing**: When to send inputs to which compiled module
- **Interfacing**: How to format inputs for and interpret outputs from compiled subgraphs
- **Composition**: How to chain or combine multiple compiled modules
- **Glue computation**: What learned computation should fill the gaps between exact compiled operations

## Novelty Assessment

### Closest prior work

| Work | Relationship | Key difference |
|------|-------------|----------------|
| Saldyt & Kambhampati (2024) — Algorithmic LMs with Neurally Compiled Libraries | Compiles algorithms into differentiable modules inside LLaMA3 | Restricted to parallel, differentiable algorithms on register machines. Not arbitrary programs, not graph-structured, not from a general-purpose language |
| Weber et al. (ICML 2024) — Learning to Compile Programs to Neural Networks | Compiles C programs to neural network weights | Produces learned *approximations*, not exact deterministic implementations. Standalone surrogates, not embedded subgraphs in a trainable network |
| Andreas et al. (2016) — Neural Module Networks | Compositional neural modules assembled per input | All modules are *learned*, not compiled from source code. Fixed vocabulary of module types |
| Reed & de Freitas (2016) — Neural Programmer-Interpreter | Learns to compose subroutines for program execution | Everything learned from demonstrations. No deterministically compiled subcomponents |
| Trask et al. (2018) — NALU | Hard-coded arithmetic structure in neural networks | Only basic arithmetic primitives (add/multiply). No compilation from a language, no arbitrary programs |
| Neuro-symbolic AI (broad field) | Embeds symbolic reasoning in neural architectures | Typically logic/rules/constraints, not arbitrary imperative programs compiled from a general-purpose language |
| Toolformer / tool-augmented LLMs (2023-2025) | Models learn to invoke tools | Tools are *external* APIs called at inference time via text. Not embedded within the model's computational graph |
| Mixture of Experts | Learned routing to specialized subnetworks | All experts are learned. No deterministic/programmatic experts |

### What is novel

1. **Compilation from a general-purpose language (Scheme) into frozen subgraphs embedded in trainable networks.** No prior work does this.
2. **GNN as compilation target** — program semantics represented as graph structure that coexists with trainable graph regions. Programs become data in the same representation space as the learned model.
3. **The "learned router + frozen exact module" pattern** where modules are arbitrary compiled programs, not just primitive operations.
4. **Homogeneous graph architecture** where the boundary between compiled (frozen) and learned (trainable) computation is a property of nodes/edges within a single graph, not an API boundary.

## Technical Considerations

### Gradient flow

Gradients flow through frozen GNN subgraphs during backpropagation — the forward computation is differentiable even with frozen weights. The outer model learns via backprop through the compiled subgraph's message-passing rounds. Parameters in the frozen region do not update; learning occurs only in the trainable regions.

### Routing mechanism

Soft routing via attention-based gating is differentiable and tractable. Hard discrete selection (which compiled module to invoke) is not differentiable and requires reinforcement learning or straight-through estimators.

### Scaling

Message passing through large frozen subgraphs is expensive. Practical systems should compile critical subroutines (not entire programs) as subgraphs. The compiled modules serve as exact computational primitives within a larger learned system.

### Compilation target properties

The GNN compilation produces:
- A fixed graph topology (`edge_index`) encoding program dataflow
- Fixed node features encoding operation types
- Deterministic MLP weights that implement exact operations (add, multiply, etc.)
- A known depth (number of message-passing rounds) determined by program structure

All of these are frozen. The trainable model wraps around this structure.

## Benchmark Context

Current benchmarks characterize the compilation pipeline and execution performance:

- **GNN GPU (RTX 4090)**: 9.3M samples/s for a 399-node tree program at 5M batch
- **Direct PyTorch GPU**: 392.8M samples/s for the same program (~42x faster)
- **Python scalar**: 38.5K samples/s (interpreter baseline)
- **GNN GPU vs interpreter**: 241x speedup from batched parallel evaluation

The 42x gap between GNN and direct PyTorch quantifies the cost of the graph-based representation — the price of making compiled programs structurally inspectable and composable within a trainable architecture.

## References

- Andreas, J., Rohrbach, M., Darrell, T., & Klein, D. (2016). Neural Module Networks. CVPR.
- Reed, S. & de Freitas, N. (2016). Neural Programmer-Interpreters. ICLR.
- Trask, A., Hill, F., Reed, S., Rae, J., Dyer, C., & Blunsom, P. (2018). Neural Arithmetic Logic Units. NeurIPS.
- Gaunt, A. L., Brockschmidt, M., Kushman, N., & Tarlow, D. (2017). Differentiable Programs with Neural Libraries. ICML.
- Saldyt, L. & Kambhampati, S. (2024). Algorithmic Language Models with Neurally Compiled Libraries. arXiv:2407.04899.
- Weber, M., et al. (2024). Learning to Compile Programs to Neural Networks. ICML. arXiv:2407.15078.
