# Copyright 2019-2026 ETH Zurich and the FP-Arena authors. All rights reserved.
"""
CloudSC's temporal discretization noise floor on the advanced state, as a variance SNR.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import dace
import numpy as np
from corpus.cloudsc.cloudsc_reference_data import (
    REFERENCE_KLEV,
    REFERENCE_KLON,
    reference_free_symbols,
    reference_input_distributions,
    reference_scalar_args,
)

from fp_arena.search import ExperimentConfig
from fp_arena.search.inputs import make_call_args
from fp_arena.search.runner import (
    _accumulate,
    _copy_args,
    _finalize,
    _new_acc,
    _output_arrays,
    _sample_rngs,
    compile_reference,
)


@dataclass(frozen=True)
class NextState:
    """The state ``state`` after one call: ``state + ptsphy * (forcing + kernel)``."""

    state: str
    forcing: str
    kernel: str

    def __call__(self, args: dict[str, Any]) -> np.ndarray:
        return args[self.state] + args["ptsphy"] * (
            args[self.forcing] + args[self.kernel]
        )


CLOUDSC_STATES: dict[str, NextState] = {
    "pt_new": NextState("pt", "tendency_tmp_t", "tendency_loc_t"),
    "pq_new": NextState("pq", "tendency_tmp_q", "tendency_loc_q"),
    "pa_new": NextState("pa", "tendency_tmp_a", "tendency_loc_a"),
    "pclv_new": NextState("pclv", "tendency_tmp_cld", "tendency_loc_cld"),
}


def _run(csdfg, args: dict[str, Any], dt: float) -> dict[str, Any]:
    call = _copy_args(args)
    call["ptsphy"] = dt
    csdfg(**call)
    return call


def _snr_var(ref: np.ndarray, out: np.ndarray) -> float:
    acc = _new_acc()
    _accumulate(acc, ref, out)
    return _finalize(acc).snr_var


def timestep_snr_floor(
    program: dace.SDFG,
    nsub: int = 2,
    dt: float | None = None,
    target: str = "cpu",
    outputs: bool = False,
) -> dict[str, float]:
    """``{name: snr_var in dB}`` between one full step and ``nsub`` substeps."""
    experiment = ExperimentConfig(
        name="cloudsc_noise_floor",
        program=program,
        symbols=reference_free_symbols(
            program, klon=REFERENCE_KLON, klev=REFERENCE_KLEV
        ),
        scalar_args=reference_scalar_args(program),
        inputs=reference_input_distributions(program, klon=REFERENCE_KLON),
        target=target,
    )
    reads, _ = program.read_and_write_sets()
    args = make_call_args(
        program, experiment, _sample_rngs(experiment.seed, 1)[0], reads=reads
    )
    dt = float(args["ptsphy"]) if dt is None else float(dt)
    csdfg = compile_reference(experiment, "fp64")

    single = _run(csdfg, args, dt)

    written = _output_arrays(program) if outputs else []
    mean = {name: np.zeros(np.shape(single[name])) for name in written}
    state = {
        s.state: np.array(args[s.state], copy=True) for s in CLOUDSC_STATES.values()
    }
    for _ in range(nsub):
        step = _run(csdfg, {**args, **state}, dt / nsub)
        for name in written:
            mean[name] += step[name] / nsub
        for s in CLOUDSC_STATES.values():
            state[s.state] = s(step)
    csdfg.finalize()

    floor = {
        name: _snr_var(s(single), state[s.state]) for name, s in CLOUDSC_STATES.items()
    }
    strictest = max(db for db in floor.values() if math.isfinite(db))
    for name in written:
        db = _snr_var(single[name], mean[name])
        # Independent of the step: no error allowed, which no rounding can meet.
        if db == math.inf and np.any(single[name]):
            db = strictest
        floor[name] = db
    return floor


if __name__ == "__main__":
    from pathlib import Path

    sdfg = dace.SDFG.from_file(Path(__file__).resolve().parent / "CLOUDSC-CPU.sdfg")
    for name, db in timestep_snr_floor(sdfg, outputs=True).items():
        print(f"  noise floor {name:18s} {db:8.2f} dB", flush=True)
