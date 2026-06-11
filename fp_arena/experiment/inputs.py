# Copyright 2019-2026 ETH Zurich and the FP-Arena authors. All rights reserved.
"""
What values go into a run: per-array input generators, optional additive noise,
and the materialisation of a run's call arguments.

An input is specified in one of two ways:

* **any SciPy distribution** (anything exposing ``rvs(size=, random_state=)``,
  which covers the whole ``scipy.stats`` catalogue.)
* **an initialisation function** ``(shape, rng) -> ndarray``.

As a default, any read array is sampled from a uniform distribution on [0, 1] without noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    AbstractSet,
    Any,
    Callable,
    Dict,
    Optional,
    Protocol,
    Sequence,
    TYPE_CHECKING,
    Union,
    runtime_checkable,
)

import numpy as np
from scipy import stats as _stats

import dace

if TYPE_CHECKING:
    from fp_arena.experiment.config import ExperimentConfig

#: Default generator for unspecified read arrays (uniform on [0, 1]).
_DEFAULT_INPUT = _stats.uniform(0.0, 1.0)


@runtime_checkable
class SupportsRVS(Protocol):
    """Structural type for SciPy distributions: a ``rvs`` sampling method."""

    def rvs(self, *args, **kwargs) -> Any: ...


class InputDistribution:
    """Internal base: produce one array realisation of a given shape/dtype."""

    def sample(
        self, shape: Sequence[int], dtype: np.dtype, rng: np.random.Generator
    ) -> np.ndarray:
        """Return an array of ``shape``/``dtype`` drawn from ``rng``."""
        raise NotImplementedError


@dataclass
class _ScipyDistribution(InputDistribution):
    """Adapter around a SciPy distribution (frozen or unfrozen)."""

    dist: SupportsRVS

    def sample(self, shape, dtype, rng):
        return np.asarray(self.dist.rvs(size=tuple(shape), random_state=rng)).astype(
            dtype
        )


@dataclass
class _CallableDistribution(InputDistribution):
    """Adapter around a user ``(shape, rng) -> ndarray`` initialisation function."""

    fn: Callable[[Sequence[int], np.random.Generator], np.ndarray]

    def sample(self, shape, dtype, rng):
        return np.asarray(self.fn(tuple(shape), rng)).astype(dtype)


DistributionLike = Union[
    SupportsRVS, Callable[[Sequence[int], np.random.Generator], np.ndarray]
]


def as_distribution(spec: DistributionLike) -> InputDistribution:
    """Coerce a SciPy distribution or ``(shape, rng) -> ndarray`` fn into one."""
    if isinstance(spec, InputDistribution):
        return spec
    if isinstance(spec, SupportsRVS):
        return _ScipyDistribution(spec)
    if callable(spec):
        return _CallableDistribution(spec)
    raise TypeError(
        f"Cannot interpret {spec!r} as an input generator; pass a scipy.stats "
        f"distribution or a (shape, rng)->ndarray initialisation function"
    )


@dataclass
class Noise:
    """
    Additive perturbation on an input array: ``x + relative*|x|*R + absolute*A``,
    where ``R``/``A`` are draws from ``relative_dist``/``absolute_dist``. Attach
    per array via an analysis config's ``noise`` field.
    """

    relative: float = 0.0
    absolute: float = 0.0
    relative_dist: Optional[DistributionLike] = None
    absolute_dist: Optional[DistributionLike] = None

    def apply(self, arr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        out = np.array(arr, copy=True)
        if self.relative and self.relative_dist is not None:
            draw = as_distribution(self.relative_dist).sample(out.shape, out.dtype, rng)
            out = out + self.relative * np.abs(out) * draw
        if self.absolute and self.absolute_dist is not None:
            draw = as_distribution(self.absolute_dist).sample(out.shape, out.dtype, rng)
            out = out + self.absolute * draw
        return out


def materialize_shape(shape, symbols: Dict[str, int]) -> tuple:
    """Resolve a (possibly symbolic) shape to a tuple of ints using ``symbols``."""
    out = []
    for dim in shape:
        if isinstance(dim, (int, np.integer)):
            out.append(int(dim))
            continue
        try:
            out.append(int(dace.symbolic.evaluate(dim, symbols)))
        except Exception as exc:
            raise ValueError(
                f"Cannot resolve dimension {dim!r} from symbols {symbols}; "
                f"provide it in ExperimentConfig.symbols"
            ) from exc
    return tuple(out)


def make_call_args(
    sdfg: dace.SDFG,
    experiment: "ExperimentConfig",
    rng: np.random.Generator,
    noise: Optional[Dict[str, Noise]] = None,
    reads: Optional[AbstractSet[str]] = None,
) -> Dict[str, Any]:
    """
    Build the kwargs for one run: every non-transient array (read arrays sampled
    from their distribution then optionally perturbed by ``noise``) plus the symbol and scalar values.
    """
    noise = noise or {}
    if reads is None:
        reads, _ = sdfg.read_and_write_sets()
    args: Dict[str, Any] = {}
    for name, desc in sdfg.arrays.items():
        if desc.transient or not isinstance(desc, dace.data.Array):
            continue
        shape = materialize_shape(desc.shape, experiment.symbols)
        np_dtype = desc.dtype.as_numpy_dtype()
        if name in reads:
            dist = as_distribution(experiment.inputs.get(name, _DEFAULT_INPUT))
            arr = dist.sample(shape, np_dtype, rng)
            n = noise.get(name)
            if n is not None:
                arr = n.apply(arr, rng)
            args[name] = arr
        else:
            args[name] = np.zeros(shape, np_dtype)
    args.update(experiment.symbols)
    args.update(experiment.scalar_args)
    return args
