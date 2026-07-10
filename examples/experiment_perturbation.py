# Copyright 2019-2026 ETH Zurich and the FP-Arena authors. All rights reserved.
"""Measure the heat3d stencil's sensitivity to input noise with the experiment framework."""

from scipy import stats

from corpus.heat3d import GRID_N, TSTEPS, heat3d_kernel
from fp_arena.experiment import (
    ExperimentConfig,
    Noise,
    PerturbationAnalysisConfig,
    ResultStore,
    run_perturbation,
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
