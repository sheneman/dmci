# `nncompile` — command-line compiler

`nncompile` is the command-line front-end for the Neural Compiler. It compiles a Scheme `.scm`
program to a portable compiled artifact (`.ncg`), emits a ready-to-import differentiable
`torch.nn.Module`, evaluates a program/artifact on any supported backend, and inspects a
compiled graph.

It is installed as a console script by `pip install -e .` (entry point
`neural_compiler.cli:main`). Equivalently, run `python -m neural_compiler.cli`.

```
nncompile <command> [options]

  compile  SOURCE.scm  [-o OUT.ncg]  [--prelude]
  emit     SOURCE.scm  [-o OUT.py]   [--params a,b]  [--backend torch]  [--prelude]
  run      SOURCE.scm | ARTIFACT.ncg   [--inputs JSON]  [--backend BACKEND]  [--prelude]
  info     SOURCE.scm | ARTIFACT.ncg   [--prelude]
```

**Input auto-detection.** You never declare a program's inputs. The free variables of the
program are discovered automatically (the compiler reports each undefined variable; the CLI
adds it as an input and retries), so `nncompile compile model.scm` just works.

**`--prelude`** prepends the standard-library prelude (`map`, `filter`, `fold-left`, …) for
programs that use those higher-order list utilities.

---

## `compile` — Scheme → portable `.ncg` artifact

```bash
nncompile compile model.scm -o model.ncg
```

Produces a `.ncg` file — the compiled program *as data*. A `.ncg` is **backend-agnostic**: the
backend (PyTorch / JAX / NumPy / CuPy) is chosen when you *run* it, not baked into the file.
One artifact runs on every backend — *compile once, differentiate everywhere*.

### The `.ncg` format

A `.ncg` is a small, human-readable **JSON** file (not a pickle — loading one cannot execute
arbitrary code):

```json
{
  "format": "nncompile-graph",
  "version": 1,
  "source": "(* a x)",
  "graph": { "root_id": ..., "input_names": ["a", "x"], "nodes": {...}, "loops": {...},
             "functions": {...}, "uses_tagged_values": false }
}
```

The `graph` object mirrors the `ComputeGraph` dataclass (`neural_compiler/graph/builder.py`).
Recursive programs share one function registry across every body graph; it is serialized once
at the top level and re-shared on load. Load it back with:

```python
from neural_compiler import load_compiled
from neural_compiler.evaluator import evaluate

g = load_compiled("model.ncg")
evaluate(g, {"a": 2.0, "x": 4.0})                  # 8.0 (torch, default)
evaluate(g, {"a": 2.0, "x": 4.0}, backend="numpy") # forward-only
```

---

## `emit` — Scheme → standalone `torch.nn.Module`

```bash
nncompile emit model.scm --params a,b -o model.py
```

Writes a self-contained Python file exposing `CompiledModel(torch.nn.Module)`:

- the input names passed to `--params` become learnable `torch.nn.Parameter`s
  (the constants you want to optimize), accessible as `model.params[name]`;
- the remaining declared inputs become `forward()` **keyword arguments** (the data).

If `--params` is omitted, the module has no learnable constants (a fixed function).

The emitted file embeds the compiled graph and depends only on `neural_compiler` and `torch`.
Use it in any optimization or search loop:

```python
# model.scm:  (/ (* k (* q1 q2)) (* r r))     # Coulomb's law; k learnable
import torch
from model import CompiledModel

model = CompiledModel(k=0.1)                          # initial guess for k
opt = torch.optim.Adam(model.parameters(), lr=0.2)
for _ in range(400):
    pred = torch.stack([model(q1=3.0, q2=4.0, r=float(r)) for r in radii])
    loss = ((pred - targets) ** 2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
print(model.params["k"].item())                       # recovers the true k
```

`forward()` picks the correct differentiable path automatically: arithmetic programs use the
batched evaluator (so scalar **or** `(B,)` data both work and keep the autograd graph);
heap/list programs (`uses_tagged_values`) use the tagged evaluator.

### Emit a JAX module

```bash
nncompile emit model.scm --backend jax --params k -o model.py
```

JAX is functional, so the emitted file exposes `apply(params, **data)` and `init_params(...)`
instead of an `nn.Module`. Optimize with `jax.grad`; `apply` is a scalar function, so `jax.vmap`
it over a batch of data:

```python
import jax, jax.numpy as jnp
from model import apply, init_params

def loss(params):
    pred = jax.vmap(lambda r: apply(params, q1=jnp.float32(3.), q2=jnp.float32(4.), r=r))(radii)
    return jnp.mean((pred - targets) ** 2)

params = init_params(k=0.1)
for _ in range(2000):
    g = jax.grad(loss)(params)
    params = {"k": params["k"] - 0.002 * g["k"]}   # or use optax
```

The JAX backend handles arithmetic/math programs whose control flow does not depend on the
differentiated constants; it does not support heap/list (tagged) programs.

---

## `run` — evaluate on a backend

```bash
nncompile run model.ncg --inputs '{"x": 4.0}'                   # torch (default)
nncompile run model.scm --inputs '{"x": 4.0}' --backend numpy   # also compiles on the fly
```

`--inputs` is a JSON object of input values; all of the program's inputs must be supplied.
The result is printed (scalar programs print a number; heap programs print the unwrapped value).
`--max-heap N` raises the tagged-value heap cap; needed only for genuinely huge list/recursion
workloads (a typical recursive DMCI program uses very little heap, so an overflow at the default
usually signals a non-terminating program instead).

---

## `info` — inspect a compiled program

```bash
nncompile info model.scm
```

```
source:        (/ (* k (* q1 q2)) (* r r))
inputs:        q1, q2, k, r
nodes:         8
graph depth:   3
has loops:     False
has functions: False   (recursion / closures)
tagged values: False   (heap-allocated pairs/lists/closures)
batchable:     yes (straight-line)
backends:      autograd = torch, jax ; forward-only = numpy, cupy
```

---

## `--dmci` — compile via the meta-circular interpreter

`compile`, `emit`, `run`, and `info` accept `--dmci`. Instead of compiling your program
directly, it compiles the **self-hosted Scheme evaluator** (`bootstrap/compiler.scm`) and runs
your program as quoted **data** through it — the method behind Paper 2. Gradients flow from the
loss, through the compiled interpreter, to your program's constants; a new program is just new
data for the same compiled evaluator.

```bash
nncompile run  model.scm --dmci --inputs '{"k":2.0,"q1":3.0,"q2":4.0,"r":2.0}'   # same value as direct
nncompile info model.scm --dmci      # tagged values: True  (runs through the interpreter)
nncompile emit model.scm --dmci --params k -o model.py   # torch module that interprets your program
```

A `--dmci` graph computes the **same value** as direct compilation (Theorem 1) but runs your
program as data through the interpreter, so it is heap-backed (`tagged: True`) and larger. It now
covers nearly the full language under the interpreter — scalar arithmetic/equation models,
recursion, and batched, vectorized vector/matrix programs (`dot`, `matmul`, `det`, `inv`, …) —
with gradients throughout.

**Recursion is trampolined.** The tagged evaluator bounces tail calls — including the
self-hosted evaluator's own `scheme-eval → eval-apply → scheme-eval` loop — in a flat Python
loop instead of nesting a Python frame per interpreted step. So tail-recursive programs run in
constant Python stack (deep recursion needs no inflated recursion limit), and structurally-
recursive programs (e.g. naive factorial) consume stack proportional only to the *interpreted
program's* genuine non-tail depth. A recursive program is just more data for the same compiled
evaluator — e.g. a tail-recursive factorial of 20 evaluates correctly in well under a second.

**Memory.** The interpreter conses into a no-GC bump-allocated heap, but a *terminating*
recursive program uses very little — only tens of cells per recursion level — so the default heap
covers thousands of levels. A sudden `Heap overflow` therefore almost always means a genuinely
**non-terminating** recursion rather than one that is merely deep. (Using an operator the
interpreter does not implement no longer silently misbehaves: the compile step runs a prescan that
raises `UnsupportedOperatorError` up front.) The interpreter implements variadic `+ - * /`,
`= < > <= >=`, `min/max/modulo/remainder`, the scalar math ops (`sin cos exp sqrt log abs pow`),
and the batched tensor vector/matrix ops (`vec ref dot cross norm normalize vsum vlen scale mat
matvec matmul transpose trace det inv outer eye zeros ones`). For a genuinely huge
list/recursion workload, raise the cap with `--max-heap N` (the heap is dict-backed, so a larger
cap is free until used).

`emit --backend jax --dmci` is rejected, because the JAX backend does not support the
heap-backed interpreter graph.

## Backend support

| Backend | Gradients | Notes |
|---|---|---|
| `torch` (default) | ✅ | the only path with batching and the full evaluator |
| `jax` | ✅ (functional) | straight-line control flow w.r.t. the differentiated variable |
| `numpy` | ❌ | reference forward evaluation |
| `cupy` | ❌ | GPU-accelerated forward-only |

## Limitations

- **`emit`** targets **torch** and **jax**; use `compile` + `evaluate(..., backend=...)` for
  NumPy/CuPy (forward-only).
- **Batched loops** now run per element: each batch element terminates on its own iteration via
  masked padded iteration, so data-dependent `loop`/`recur` bounds are correct in `evaluate_batched`.
  General (non-tail) **recursive function calls** in batched mode are still inlined uniformly —
  for those, evaluate per item. `info` reports a program's batchability.
- **`--dmci`** recursion is trampolined: tail calls (including the interpreter's own eval loop)
  bounce in a flat Python loop, so deep recursion runs in constant Python stack. A *terminating*
  recursive program also uses very little heap (tens of cells per level); a `Heap overflow`
  almost always signals a genuinely non-terminating recursion, not mere depth. (Unsupported
  operators are rejected up front by the compile-time prescan, so they cannot silently cause it.)
- A `.ncg` records the graph, not the training data or learned parameter values.
