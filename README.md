# FP-Arena

A rapid-prototyping environment for testing custom floating-point types in
[DaCe](https://github.com/spcl/dace). FP-Arena adds new FP types (and the C++
that implements them) as a **plugin** that registers itself into DaCe at import
time — no fork, no DaCe source changes.


## Install

FP-Arena tracks the latest DaCe [`extended`](https://github.com/spcl/dace/commits/extended/):

```bash
pip install git+https://github.com/spcl/FP-Arena.git
```

Already have a DaCe checkout (any `extended`-based branch) you want to use? Install
without pulling DaCe:

```bash
pip install --no-deps git+https://github.com/spcl/FP-Arena.git   # or: pip install --no-deps -e .
```

## Quick start

Importing `fp_arena` registers the types *and* auto-enables any SDFG that uses
them, so there is nothing else to call:

```python
import numpy as np
import dace
import fp_arena                       # registers types + auto-enables on compile

@dace.program
def axpy(a: fp_arena.float32sr[1024], b: fp_arena.float32sr[1024],
         c: fp_arena.float32sr[1024]):
    for i in dace.map[0:1024]:
        c[i] = a[i] * b[i] + c[i]     # arithmetic rounds stochastically

sdfg = axpy.to_sdfg()
a = np.full(1024, 1.0, np.float32)
b = np.full(1024, 1.0, np.float32)
c = np.zeros(1024, np.float32)
sdfg(a=a, b=b, c=c)                    # headers injected + fast-math stripped automatically
```


## Notes

- Auto-enable wraps `SDFG.compile` (installed on import) and only touches SDFGs
  that actually use an FP-Arena type. Turn it off with
  `fp_arena.disable_auto_extensions()`; enable a single SDFG explicitly with
  `sdfg.enable_fp_arena_extensions()`.
- The SR types are header-only, allocation-free, and `__host__`/`__device__`
  capable (GPU codegen works; the device RNG is clock-seeded).
- Stochastic rounding needs exact IEEE rounding, so enabling removes
  `-ffast-math` / `/fp:fast` / `--use_fast_math` from DaCe's compiler flags
  (otherwise on by default). Auto-enable scopes this to the SR compile; the
  explicit `enable_fp_arena_extensions` makes it a persistent default.
- Both types capture each operation's result exactly (error-free), which is what
  matters for error analysis. `float32sr` evaluates each op in `double` (exact
  for +/−/×) and rounds by perturbing the dropped mantissa bits; `float64sr`
  uses error-free transforms (TwoSum, FMA TwoProd) since there is no wider native
  type, then rounds against the exact residual (`ulp` is a power of two, so the
  probability compare is exact too).
