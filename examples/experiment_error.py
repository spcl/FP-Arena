# Copyright 2019-2026 ETH Zurich and the FP-Arena authors. All rights reserved.
"""Measure the heat3d stencil's error in reduced precision against an fp64 reference."""

from scipy import stats

from corpus.heat3d import GRID_N, TSTEPS, heat3d_kernel
from fp_arena.experiment import (
    ErrorAnalysisConfig,
    ExperimentConfig,
    ResultStore,
    run_error,
)

N_SAMPLES = 3


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
