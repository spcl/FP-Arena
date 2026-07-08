# Copyright 2019-2026 ETH Zurich and the FP-Arena authors. All rights reserved.
"""Time the heat3d stencil across precision points with the experiment framework."""

import statistics

import dace as dc

from scipy import stats

from fp_arena.experiment import (
    ExperimentConfig,
    PerformanceAnalysisConfig,
    ResultStore,
    run_performance,
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

    store = ResultStore("heat3d_results.db")
    results = run_performance(
        PerformanceAnalysisConfig(
            experiment,
            precisions=[
                {},  # the unmodified fp64 program
                {"A": "fp32", "B": "fp32"},
                {"A": "fp16", "B": "fp16"},
            ],
            n_warmup=1,
            n_reps=5,
        ),
        store=store,
    )

    for r in results:
        label = " ".join(f"{k}={v}" for k, v in r.precision.items()) or "fp64 baseline"
        print(
            f"{label:>16}: total {statistics.median(r.total_times):8.3f} ms  "
            f"kernel {statistics.median(r.kernel_times):8.3f} ms"
        )


if __name__ == "__main__":
    main()
