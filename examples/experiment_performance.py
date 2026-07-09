# Copyright 2019-2026 ETH Zurich and the FP-Arena authors. All rights reserved.
"""Time the heat3d stencil across precision points with the experiment framework."""

import statistics

from scipy import stats

from corpus.heat3d import GRID_N, TSTEPS, heat3d_kernel
from fp_arena.experiment import (
    ExperimentConfig,
    PerformanceAnalysisConfig,
    ResultStore,
    run_performance,
)


def main():
    experiment = ExperimentConfig(
        name="heat3d",
        program=heat3d_kernel.to_sdfg(simplify=True),
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
