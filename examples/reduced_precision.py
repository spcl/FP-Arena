# Copyright 2019-2026 ETH Zurich and the FP-Arena authors. All rights reserved.
"""Run the heat3d stencil in reduced precision."""

import dace as dc
import numpy as np

from corpus.heat3d import GRID_N, TSTEPS, heat3d_kernel
from fp_arena.transformations.change_and_propagate_fp_types import (
    change_and_propagate_fp_types,
)


def main():
    sdfg = heat3d_kernel.to_sdfg(simplify=True)

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
