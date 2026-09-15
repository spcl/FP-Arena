# Copyright 2019-2026 ETH Zurich and the FP-Arena authors. All rights reserved.
"""
Phase timing by inserted timer tasklets.

"""

from __future__ import annotations

import ctypes
import re

import dace
import numpy as np

from fp_arena.environments import Timers, TimersGPU
from fp_arena.experiment.retarget import _transfer_direction

#: Phases in pipeline order; each maps to a ``<phase>_times`` field of PerfResult.
PHASES = ("h2d", "cast_in", "kernel", "cast_out", "d2h")

_STOP_LABEL = "__fp_timer_stop"
_START_LABEL = "__fp_timer_start"
_BEGIN_LABEL = "__fp_timer_begin"
#: Recovers (slot, phase) from a stop-state label, tolerating a uniquifying suffix.
_STOP_RE = re.compile(rf"^{_STOP_LABEL}_(\d+)__({'|'.join(PHASES)})")


def _has_cast_map(state) -> bool:
    """Whether ``state`` contains a precision-cast map."""
    return any(
        isinstance(n, dace.nodes.MapEntry) and n.map.label.startswith("cast_map_")
        for n in state.nodes()
    )


def _classify_state(sdfg: dace.SDFG, state) -> str:
    """Assign ``state`` to a timing phase: h2d/d2h, cast_in/cast_out, or kernel."""
    direction = _transfer_direction(sdfg, state)
    if direction is not None:
        return direction
    if _has_cast_map(state):
        return "cast_out" if state.label.startswith("copy_out") else "cast_in"
    return "kernel"


def _homogeneous_category(sdfg: dace.SDFG, block) -> str | None:
    """The block's one phase, or ``None`` if it's a region spanning several (the caller recurses into those)."""
    if isinstance(block, dace.SDFGState):
        return _classify_state(sdfg, block)
    cats = {_classify_state(sdfg, s) for s in block.all_states()}
    return next(iter(cats)) if len(cats) == 1 else None


def _timer_tasklet(state, body: str, target: str) -> None:
    """Add an isolated, side-effecting CPP tasklet carrying one timer call."""
    tasklet = state.add_tasklet(
        name="timer", inputs={}, outputs={}, code=body, language=dace.Language.CPP
    )
    envs = {Timers.full_class_path()}
    if target == "gpu":
        envs.add(TimersGPU.full_class_path())
    tasklet.environments = envs
    tasklet.side_effects = True


def _bracket(
    region, block, slot: int, category: str, target: str, is_start: bool
) -> None:
    """Splice a ``start(slot)`` state before ``block`` and a ``stop(slot)`` after it."""
    pre = region.add_state_before(
        block, label=f"{_START_LABEL}_{slot}", is_start_block=is_start
    )
    _timer_tasklet(pre, f"fp_arena::timer::start({slot});", target)
    post = region.add_state_after(block, label=f"{_STOP_LABEL}_{slot}__{category}")
    _timer_tasklet(post, f"fp_arena::timer::stop({slot});", target)


def _bracket_region(
    sdfg: dace.SDFG, region, target: str, categories: list[str]
) -> None:
    """Bracket every phase-homogeneous child of ``region``; recurse into the rest."""
    start_block = region.start_block
    for block in list(region.nodes()):
        category = _homogeneous_category(sdfg, block)
        if category is None:
            _bracket_region(sdfg, block, target, categories)
            continue
        slot = len(categories)
        categories.append(category)
        _bracket(region, block, slot, category, target, block is start_block)


def insert_timers(sdfg: dace.SDFG, target: str) -> list[str]:
    """
    Bracket each phase with start/stop timer tasklets; prepend ``begin_invocation``.

    Must run last, after retargeting -- otherwise a later fusion pass would move the timer states.
    """
    categories: list[str] = []
    _bracket_region(sdfg, sdfg, target, categories)
    begin = sdfg.add_state_before(
        sdfg.start_block, label=_BEGIN_LABEL, is_start_block=True
    )
    _timer_tasklet(
        begin, f"fp_arena::timer::begin_invocation({len(categories)});", target
    )
    return categories


def timer_categories(sdfg: dace.SDFG) -> list[str]:
    """Recover the slot -> phase list from a timed ``sdfg``'s stop-state labels."""
    by_slot: dict[int, str] = {}
    for state in sdfg.all_states():
        match = _STOP_RE.match(state.label)
        if match:
            by_slot[int(match.group(1))] = match.group(2)
    return [by_slot[slot] for slot in sorted(by_slot)]


def reset_timers(csdfg) -> None:
    """Clear the compiled SDFG's timer buffer before a fresh timed run."""
    reset = csdfg.get_exported_function("fp_arena_timer_reset")
    if reset is None:
        raise RuntimeError(
            "fp_arena_timer_reset not found; the SDFG was not built with timers"
        )
    reset()


def read_timers(csdfg, n_slots: int) -> np.ndarray:
    """Read the timer buffer back as a ``[n_invocations, n_slots]`` array of ms."""
    count = csdfg.get_exported_function("fp_arena_timer_count", restype=ctypes.c_long)
    copy = csdfg.get_exported_function("fp_arena_timer_copy", restype=None)
    if count is None or copy is None:
        raise RuntimeError("timer readback symbols not found in the compiled SDFG")
    n = int(count())
    if n_slots <= 0 or n == 0:
        return np.zeros((0, max(n_slots, 0)))
    buf = (ctypes.c_double * n)()
    copy(buf)
    return np.frombuffer(buf, dtype=np.float64).reshape(-1, n_slots)


def phase_breakdown(categories: list[str], buf: np.ndarray, n_reps: int) -> dict:
    """
    Reduce the timer buffer to the per-rep :class:`PerfResult` series.

    Leading rows are warmups and dropped; ``total`` is the per-rep sum over the
    phases present.
    """
    empty = {"total_times": [], **{f"{p}_times": [] for p in PHASES}}
    if buf.size == 0:
        return empty
    reps = buf[-n_reps:]
    cats = np.asarray(categories)
    out: dict[str, list[float]] = {}
    present: list[np.ndarray] = []
    for phase in PHASES:
        cols = np.flatnonzero(cats == phase)
        if cols.size:
            series = reps[:, cols].sum(axis=1)
            out[f"{phase}_times"] = series.tolist()
            present.append(series)
        else:
            out[f"{phase}_times"] = []
    total = np.sum(present, axis=0).tolist() if present else []
    return {"total_times": total, **out}
