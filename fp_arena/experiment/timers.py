# Copyright 2019-2026 ETH Zurich and the FP-Arena authors. All rights reserved.
"""
Phase timing by inserted timer tasklets.

"""

from __future__ import annotations

import ctypes
import re

import dace
import numpy as np
from dace.sdfg import utils as sdfg_utils

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
    """The block's one phase, or ``None`` if it's a region spanning several."""
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


def _bracket(region, first, last, slot: int, category: str, target: str) -> None:
    """Splice a ``start(slot)`` state before ``first`` and a ``stop(slot)`` after ``last``."""
    pre = region.add_state_before(
        first,
        label=f"{_START_LABEL}_{slot}",
        is_start_block=first is region.start_block,
    )
    _timer_tasklet(pre, f"fp_arena::timer::start({slot});", target)
    post = region.add_state_after(last, label=f"{_STOP_LABEL}_{slot}__{category}")
    _timer_tasklet(post, f"fp_arena::timer::stop({slot});", target)


def _phase_sequence(sdfg: dace.SDFG) -> list[tuple[object, str]]:
    """The top-level blocks in execution order, each with its phase."""
    blocks = list(sdfg_utils.dfs_topological_sort(sdfg))
    cats = [_homogeneous_category(sdfg, b) or "kernel" for b in blocks]
    head = 0
    while head < len(cats) and cats[head] in ("h2d", "cast_in"):
        head += 1
    tail = len(cats)
    while tail > head and cats[tail - 1] in ("cast_out", "d2h"):
        tail -= 1
    for i in range(head, tail):
        cats[i] = "kernel"
    return list(zip(blocks, cats))


def insert_timers(sdfg: dace.SDFG, target: str) -> list[str]:
    """
    Bracket each phase with start/stop timer tasklets.

    Must run last, after retargeting -- otherwise a later fusion pass would move the timer states.
    """
    categories: list[str] = []
    slot_of: dict[str, int] = {}
    runs: list[list] = []  # [first, last, category]
    for block, category in _phase_sequence(sdfg):
        prev = runs[-1] if runs else None
        if (
            prev is not None
            and prev[2] == category
            and sdfg.out_degree(prev[1]) == 1
            and sdfg.in_degree(block) == 1
            and sdfg.out_edges(prev[1])[0].dst is block
        ):
            prev[1] = block
        else:
            runs.append([block, block, category])
    for first, last, category in runs:
        if category not in slot_of:
            slot_of[category] = len(categories)
            categories.append(category)
        _bracket(sdfg, first, last, slot_of[category], category, target)
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
