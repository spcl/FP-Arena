# Copyright 2019-2026 ETH Zurich and the FP-Arena authors. All rights reserved.
"""
The bridge between an :class:`ExperimentConfig` and DaCe: build a fresh SDFG and retarget its precision using :func:`change_and_propagate_fp_types`.
"""

from __future__ import annotations

import copy

import dace
from dace.transformation.passes.insert_explicit_copies import InsertExplicitCopies
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.vectorize_gpu import VectorizeGPU

from fp_arena.experiment import registry
from fp_arena.experiment.config import CONSTANTS_KEY, ExperimentConfig, PrecisionMap
from fp_arena.transformations.change_and_propagate_fp_types import (
    _add_fusion_barrier,
    change_and_propagate_fp_types,
)


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


def candidate_fp_arrays(sdfg: dace.SDFG) -> list[str]:
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


def apply_precision(
    sdfg: dace.SDFG,
    pin_map: PrecisionMap,
    promotion_rules,
) -> None:
    """
    Retarget ``sdfg`` in place to the precisions in ``pin_map`` via :func:`change_and_propagate_fp_types`
    """
    if not pin_map:
        return
    pins = dict(pin_map)
    constants_key = pins.pop(CONSTANTS_KEY, None)
    _validate_pins(sdfg, pins)
    typed = {name: registry.to_typeclass(key) for name, key in pins.items()}
    change_and_propagate_fp_types(
        sdfg,
        typed,
        promotion_rules,
        constant_type=(
            None if constants_key is None else registry.to_typeclass(constants_key)
        ),
    )


def apply_reference(sdfg: dace.SDFG, reference, promotion_rules) -> None:
    """
    Lower ``sdfg`` in place to the reference precision (a single key for every fp array, or a per-array ``{name: key}`` map).
    """
    if isinstance(reference, str):
        ref_map = {name: reference for name in candidate_fp_arrays(sdfg)}
    else:
        ref_map = dict(reference)
    apply_precision(sdfg, ref_map, promotion_rules)


#: Device storage types; an edge crossing this boundary is a host<->device copy.
_GPU_STORAGE = (
    dace.dtypes.StorageType.GPU_Global,
    dace.dtypes.StorageType.GPU_Shared,
)


def _transfer_direction(sdfg: dace.SDFG, state) -> str | None:
    """``"h2d"``/``"d2h"`` if ``state`` copies across the host/device boundary, else ``None``."""
    for e in state.edges():
        src, dst = e.src, e.dst
        if isinstance(src, dace.nodes.AccessNode) and isinstance(
            dst, dace.nodes.AccessNode
        ):
            src_dev = sdfg.arrays[src.data].storage in _GPU_STORAGE
            dst_dev = sdfg.arrays[dst.data].storage in _GPU_STORAGE
            if src_dev != dst_dev:
                return "h2d" if dst_dev else "d2h"
    return None


#: The vectorization config used when ``gpu_vectorize`` is on but no config is given.
DEFAULT_GPU_VECTORIZE_CONFIG = VectorizeConfig(
    widths=(2,), remainder_strategy="branched_tail"
)


def resolve_vectorize_config(
    target: str,
    gpu_vectorize: bool,
    gpu_vectorize_config: VectorizeConfig | None = None,
) -> VectorizeConfig | None:
    """:returns: the ``VectorizeConfig`` :func:`apply_target` applies, or ``None`` if nothing is vectorized."""
    if target != "gpu" or not gpu_vectorize:
        return None
    return gpu_vectorize_config or DEFAULT_GPU_VECTORIZE_CONFIG


def _set_gpu_block_size(sdfg: dace.SDFG, gpu_block_size: list[int]) -> None:
    """
    Set ``gpu_block_size`` on every GPU kernel map of ``sdfg``.
    """

    InsertExplicitCopies().apply_pass(sdfg, {})
    sdfg.expand_library_nodes(recursive=True)

    # Recursive: expansion puts the copy kernel inside a nested SDFG.
    for node, _ in sdfg.all_nodes_recursive():
        if (
            isinstance(node, dace.nodes.MapEntry)
            and node.map.schedule == dace.dtypes.ScheduleType.GPU_Device
        ):
            node.map.gpu_block_size = list(gpu_block_size)


def apply_target(
    sdfg: dace.SDFG,
    target: str,
    gpu_block_size: list[int] | None = None,
    gpu_vectorize: bool = False,
    gpu_vectorize_config: VectorizeConfig | None = None,
) -> None:
    """
    Retarget ``sdfg`` in place for the execution target.
    """
    if target == "cpu":
        return
    if target == "gpu":
        sdfg.apply_gpu_transformations(simplify=False)
        for state in sdfg.all_states():
            if _transfer_direction(sdfg, state) is not None:
                _add_fusion_barrier(state)
        sdfg.simplify()
        config = resolve_vectorize_config(target, gpu_vectorize, gpu_vectorize_config)
        if config is not None:
            VectorizeGPU(config).apply_pass(sdfg, {})
        if gpu_block_size is not None:
            _set_gpu_block_size(sdfg, gpu_block_size)
        return
    raise ValueError(f"Unknown target {target!r}; expected 'cpu' or 'gpu'")
