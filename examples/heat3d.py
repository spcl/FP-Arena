# Copyright 2019-2026 ETH Zurich and the FP-Arena authors. All rights reserved.
"""Run the heat3d stencil in reduced precision."""

import dace as dc
import numpy as np

from fp_arena.transformations.change_and_propagate_fp_types import (
    change_and_propagate_fp_types,
)

GRID_N, TSTEPS = 40, 20

N = dc.symbol("N", dtype=dc.int64)


@dc.program
def kernel(TSTEPS: dc.int64, A: dc.float64[N, N, N], B: dc.float64[N, N, N]):
    for t in range(1, TSTEPS):
        B[1:-1, 1:-1, 1:-1] = (
            0.125 * (A[2:, 1:-1, 1:-1] - 2.0 * A[1:-1, 1:-1, 1:-1] + A[:-2, 1:-1, 1:-1])
            + 0.125
            * (A[1:-1, 2:, 1:-1] - 2.0 * A[1:-1, 1:-1, 1:-1] + A[1:-1, :-2, 1:-1])
            + 0.125
            * (A[1:-1, 1:-1, 2:] - 2.0 * A[1:-1, 1:-1, 1:-1] + A[1:-1, 1:-1, :-2])
            + A[1:-1, 1:-1, 1:-1]
        )
        A[1:-1, 1:-1, 1:-1] = (
            0.125 * (B[2:, 1:-1, 1:-1] - 2.0 * B[1:-1, 1:-1, 1:-1] + B[:-2, 1:-1, 1:-1])
            + 0.125
            * (B[1:-1, 2:, 1:-1] - 2.0 * B[1:-1, 1:-1, 1:-1] + B[1:-1, :-2, 1:-1])
            + 0.125
            * (B[1:-1, 1:-1, 2:] - 2.0 * B[1:-1, 1:-1, 1:-1] + B[1:-1, 1:-1, :-2])
            + B[1:-1, 1:-1, 1:-1]
        )


def main():
    sdfg = kernel.to_sdfg(simplify=True)

    change_and_propagate_fp_types(
        sdfg,
        {"A": dc.float16, "B": dc.float16},
        constant_type=dc.float16,
    )

    rng = np.random.default_rng(0)
    A = rng.uniform(0, 100, (GRID_N,) * 3)
    B = rng.uniform(0, 100, (GRID_N,) * 3)
    sdfg(TSTEPS=TSTEPS, A=A, B=B, N=GRID_N)


if __name__ == "__main__":
    main()
