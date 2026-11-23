# Changelog

## 0.1.0 (2026-11-23)

The first release, published on PyPI as `fp-arena`.

- Custom floating-point types for DaCe (reduced precision, stochastic rounding), registered as a plugin at import
  time: no DaCe fork and no DaCe source changes.
- `change_and_propagate_fp_types` retypes an SDFG's data and symbols to a chosen format and propagates it.
- Experiments for error, performance, and perturbation, and precision searches on heat3d, hdiff, vadv, and CLOUDSC.
- The benchmark kernels moved from a top-level `corpus` package to `fp_arena.corpus`.
- DaCe extended comes from PyPI as `dace-extended>=1.0.0`. FP-Arena requires Python 3.12 or newer.
