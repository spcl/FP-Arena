# Copyright 2019-2026 ETH Zurich and the FP-Arena authors. All rights reserved.
"""
Pick the fastest per-array precision assignment for heat3d that stays within budget.
"""

import dace as dc
from dace.transformation.auto import auto_optimize as aopt
from dace.transformation.dataflow import MapFusion
from dace.transformation.interstate import LoopToMap
from scipy import stats

from corpus.heat3d import heat3d_kernel
from fp_arena.experiment import (
    CONSTANTS_KEY,
    ErrorBudget,
    ExperimentConfig,
    Noise,
    ResultStore,
    SelectionAnalysisConfig,
    precision_grid,
    run_selection,
)

DB_PATH = "heat3d_selection.db"

GRID_N = 128
TSTEPS = 20

SDFG = heat3d_kernel.to_sdfg(simplify=True)
SDFG = aopt.auto_optimize(SDFG, dc.DeviceType.CPU)
SDFG.apply_transformations_repeated(LoopToMap)
SDFG.apply_transformations_repeated(MapFusion)
SDFG.simplify()


def main() -> None:

    experiment = ExperimentConfig(
        name="heat3d",
        program=SDFG,
        symbols={"N": GRID_N},
        scalar_args={"TSTEPS": TSTEPS},
        inputs={
            "A": stats.uniform(0, 100),
            "B": stats.uniform(0, 100),
        },
    )

    error_budget = ErrorBudget(
        from_perturbation={"rel_max": 1.0, "snr": 1.0},
        limits={"rel_max": 1e-3, "snr": 60.0},
    )

    cfg = SelectionAnalysisConfig(
        experiment=experiment,
        precisions=precision_grid(
            {
                "A": ["fp16", "fp32", "fp64"],
                "B": ["fp16", "fp32", "fp64"],
                CONSTANTS_KEY: ["fp16", "fp32", "fp64"],
            }
        ),
        budget=error_budget,
        noise={
            "A": Noise(relative=1e-6, relative_dist=stats.uniform(-1.0, 2.0)),
            "B": Noise(relative=1e-6, relative_dist=stats.uniform(-1.0, 2.0)),
        },
        reference="fp64",
        speedup_baseline={"A": "fp64", "B": "fp64", CONSTANTS_KEY: "fp64"},
        n_samples=3,
        n_warmup=3,
        n_reps=10,
        time_all=True,
        objective="kernel",
    )

    store = ResultStore(DB_PATH)
    try:
        run_selection(cfg, store=store)
    finally:
        store.close()


if __name__ == "__main__":
    main()
