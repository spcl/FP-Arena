# Copyright 2019-2026 ETH Zurich and the FP-Arena authors. All rights reserved.
"""Measure the heat3d stencil's sensitivity to input noise with the experiment framework."""

import dace as dc
from scipy import stats

from fp_arena.experiment import (
    ExperimentConfig,
    Noise,
    PerturbationAnalysisConfig,
    ResultStore,
    run_perturbation,
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
    experiment = ExperimentConfig(
        name="heat3d",
        program=kernel.to_sdfg(simplify=True),
        symbols={"N": GRID_N},
        scalar_args={"TSTEPS": TSTEPS},
        target="cpu",
        inputs={
            "A": stats.uniform(0, 100),
            "B": stats.uniform(0, 100),
        },
    )

    noise = Noise(relative=1e-3, relative_dist=stats.uniform(-1.0, 2.0))
    store = ResultStore("heat3d_results.db")
    results = run_perturbation(
        PerturbationAnalysisConfig(
            experiment,
            noise={"A": noise, "B": noise},
            precisions=[{}],  # the unmodified fp64 program
        ),
        store=store,
    )

    for r in results:
        for arr in sorted(r.errors):
            e = r.errors[arr]
            print(
                f"perturbed {r.perturbed} -> {arr}: rel_mean {e.rel_mean:.3e}  "
                f"rel_max {e.rel_max:.3e}"
            )


if __name__ == "__main__":
    main()
