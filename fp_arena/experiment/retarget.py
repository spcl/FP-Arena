# Copyright 2019-2026 ETH Zurich and the FP-Arena authors. All rights reserved.
"""
The bridge between an :class:`ExperimentConfig` and DaCe: build a fresh SDFG and retarget its precision using :func:`change_and_propagate_fp_types`.
"""

import copy
from typing import List

import dace

from fp_arena.transformations.change_and_propagate_fp_types import (
    change_and_propagate_fp_types,
)
from fp_arena.experiment import registry
from fp_arena.experiment.config import ExperimentConfig, PrecisionMap


def _is_fp(tc: dace.dtypes.typeclass) -> bool:
    """:returns: whether ``tc`` is a registry float (native, SR, or MPFR)."""
    try:
        registry.key_of(tc)
        return True
    except ValueError:
        return False


def fresh_sdfg(experiment: ExperimentConfig) -> dace.SDFG:
    """
    A deep copy of the experiment's SDFG -- lowering mutates in place, so every precision point needs its own.
    """
    if not isinstance(experiment.program, dace.SDFG):
        raise TypeError(
            f"program must be a dace.SDFG, got {type(experiment.program).__name__}"
        )
    return copy.deepcopy(experiment.program)


def candidate_fp_arrays(sdfg: dace.SDFG) -> List[str]:
    """:returns: non-transient floating-point array names (the boundary fp arrays)."""
    return sorted(
        name
        for name, desc in sdfg.arrays.items()
        if not desc.transient
        and _is_fp(desc.dtype)
        and isinstance(desc, dace.data.Array)
    )


def _validate_pins(sdfg: dace.SDFG, pin_map: PrecisionMap) -> None:
    for name in pin_map:
        if name not in sdfg.arrays:
            raise ValueError(f"Pinned array {name!r} not found in SDFG {sdfg.name!r}")
        if not _is_fp(sdfg.arrays[name].dtype):
            raise ValueError(f"Pinned array {name!r} is not a floating-point array")


def _ensure_mpfr_linked() -> None:
    """Add the MPFR library to DaCe's CPU link line."""
    libs = dace.Config.get("compiler", "cpu", "libs") or ""
    if "mpfr" not in libs.split():
        dace.Config.append("compiler", "cpu", "libs", value=" mpfr")


def apply_precision(sdfg: dace.SDFG, pin_map: PrecisionMap, promotion_rules) -> None:
    """
    Retarget ``sdfg`` in place to the precisions in ``pin_map`` via :func:`change_and_propagate_fp_types`
    """
    if not pin_map:
        return
    _validate_pins(sdfg, pin_map)
    if any(registry.is_mpfr(key) for key in pin_map.values()):
        _ensure_mpfr_linked()
    typed = {name: registry.to_typeclass(key) for name, key in pin_map.items()}
    change_and_propagate_fp_types(sdfg, typed, promotion_rules)


def apply_reference(sdfg: dace.SDFG, reference, promotion_rules) -> None:
    """
    Lower ``sdfg`` in place to the reference precision (a single key for every fp array, or a per-array ``{name: key}`` map).
    """
    if isinstance(reference, str):
        ref_map = {name: reference for name in candidate_fp_arrays(sdfg)}
    else:
        ref_map = dict(reference)
    apply_precision(sdfg, ref_map, promotion_rules)


def apply_target(sdfg: dace.SDFG, target: str) -> None:
    """Retarget ``sdfg`` in place for the experiment's execution target."""
    if target == "cpu":
        return
    if target == "gpu":
        sdfg.apply_gpu_transformations()
        return
    raise ValueError(f"Unknown target {target!r}; expected 'cpu' or 'gpu'")
