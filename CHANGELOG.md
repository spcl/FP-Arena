# Changelog

## 0.1.0 (2026-11-23)

The first release, published on PyPI as `fp-arena`.

- Custom floating-point types for DaCe (reduced precision, stochastic rounding), registered as a plugin at import
  time: no DaCe fork and no DaCe source changes.
- `change_and_propagate_fp_types` retypes an SDFG's data and symbols to a chosen format and propagates it.
- Experiments for error, performance, and perturbation, and precision searches on heat3d, hdiff, vadv, and CLOUDSC.
- The benchmark kernels moved from a top-level `corpus` package to `fp_arena.corpus`.
- DaCe comes through an extra: `fp-arena[extended]` (dace-extended>=1.0.0, all of FP-Arena) or `fp-arena[dace]`
  (dace>=2.0.0b2: the types and `change_and_propagate_fp_types`; `fp_arena.search` needs DaCe extended). FP-Arena
  requires Python 3.12 or newer.
