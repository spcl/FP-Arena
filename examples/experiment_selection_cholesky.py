# Copyright 2019-2026 ETH Zurich and the FP-Arena authors. All rights reserved.
"""
Pick the fastest per-array precision assignment for cholesky that stays within budget.
"""

import dace as dc
import numpy as np
from dace.transformation.auto import auto_optimize as aopt
from dace.transformation.dataflow import MapFusion
from dace.transformation.interstate import LoopToMap
from scipy import stats

from corpus.cholesky import cholesky_kernel
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

DB_PATH = "cholesky_selection.db"

N = 128

SDFG = cholesky_kernel.to_sdfg(simplify=True)
SDFG = aopt.auto_optimize(SDFG, dc.DeviceType.CPU)
SDFG.apply_transformations_repeated(LoopToMap)
SDFG.apply_transformations_repeated(MapFusion)
SDFG.simplify()


def spd(shape, rng):
    """A symmetric positive-definite matrix -- the only input Cholesky is defined on."""
    n = shape[0]
    b = rng.standard_normal(shape)
    return (b @ b.T) / n + np.eye(n)


def main() -> None:

    experiment = ExperimentConfig(
        name="cholesky",
        program=SDFG,
        symbols={"N": N},
        inputs={"A": spd},
    )

    error_budget = ErrorBudget(
        from_perturbation={"rel_max": 1.0}, limits={"rel_max": 1e-6, "snr": 60.0}
    )

    cfg = SelectionAnalysisConfig(
        experiment=experiment,
        precisions=precision_grid(
            {
                "A": ["fp16", "fp32", "fp64"],
                CONSTANTS_KEY: ["fp16", "fp32", "fp64"],
            }
        ),
        budget=error_budget,
        noise={
            "A": Noise(relative=1e-6, relative_dist=stats.uniform(-1.0, 2.0)),
        },
        reference="fp64",
        speedup_baseline={"A": "fp64", CONSTANTS_KEY: "fp64"},
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
