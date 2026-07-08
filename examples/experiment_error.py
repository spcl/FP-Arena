# Copyright 2019-2026 ETH Zurich and the FP-Arena authors. All rights reserved.
"""Measure the heat3d stencil's error in reduced precision against an fp64 reference."""

import dace as dc
from scipy import stats

from fp_arena.experiment import (
    ErrorAnalysisConfig,
    ExperimentConfig,
    ResultStore,
    run_error,
)

GRID_N, TSTEPS = 40, 20
N_SAMPLES = 3

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

    store = ResultStore("heat3d_results.db")
    results = run_error(
        ErrorAnalysisConfig(
            experiment,
            precisions=[{"A": "fp32", "B": "fp32"}, {"A": "fp16", "B": "fp16"}],
            reference="fp64",
            n_samples=N_SAMPLES,
        ),
        store=store,
    )

    for r in results:
        label = " ".join(f"{k}={v}" for k, v in r.precision.items())
        for arr in sorted(r.errors):
            e = r.errors[arr]
            print(
                f"{label:>16} {arr}: rel_mean {e.rel_mean:.3e}  "
                f"rel_max {e.rel_max:.3e}  linf {e.linf:.3e}"
            )

    print(f"{len(store.query(experiment='heat3d', kind='error'))} rows in the database")
    store.close()


if __name__ == "__main__":
    main()
