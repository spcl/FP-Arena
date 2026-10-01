# Copyright 2019-2026 ETH Zurich and the FP-Arena authors. All rights reserved.
"""
Search for the fastest per-array precision assignment for the horizontal
diffusion (hdiff) stencil that stays within budget, using the pruning selection
*search* (``run_search``).

``--knobs`` selects which knobs are in scope:
  * ``sources``   -- read inputs only (the default);
  * ``overrides`` -- also intermediates and outputs (many more knobs);
  * ``constants`` -- also the float literals in the kernel.
Results go to ``hdiff_search_<knobs>.db`` under experiment name ``hdiff_<knobs>``.
"""

import argparse

import dace as dc
from dace.transformation.auto import auto_optimize as aopt
from dace.transformation.dataflow import MapFusion, PruneConnectors
from dace.transformation.interstate import LoopToMap
from scipy import stats

from corpus.hdiff import hdiff
from fp_arena.experiment import (
    ErrorBudget,
    ExperimentConfig,
    KnobSelection,
    ResultStore,
    SelectionSearchConfig,
    run_search,
)

I, J, K = 1024, 1024, 256


def main() -> None:
    parser = argparse.ArgumentParser(description="hdiff precision selection search")
    parser.add_argument(
        "--knobs",
        nargs="+",
        choices=("sources", "overrides", "constants"),
        default=["sources"],
        metavar="SCOPE",
        help="knob scopes to search (space-separated); sources is always included. "
        "e.g. --knobs overrides constants",
    )
    parser.add_argument(
        "--max-configs",
        type=int,
        default=None,
        help="cap on distinct programs compiled (compute_budget); mainly for the "
        "'overrides' scope, which exposes many knobs. Default: run to exhaustion.",
    )
    args = parser.parse_args()
    scopes = set(args.knobs)
    tag = "_".join(s for s in ("overrides", "constants") if s in scopes) or "sources"

    sdfg = hdiff.to_sdfg(simplify=True)
    sdfg.specialize({"I": I, "J": J, "K": K})
    sdfg = aopt.auto_optimize(sdfg, dc.DeviceType.CPU)
    sdfg.apply_transformations_repeated(LoopToMap)
    sdfg.apply_transformations_repeated(MapFusion)
    sdfg.apply_transformations_repeated(PruneConnectors)
    sdfg.simplify()

    experiment = ExperimentConfig(
        name=f"hdiff_{tag}",
        program=sdfg,
        symbols={"I": I, "J": J, "K": K},
        inputs={
            "in_field": stats.uniform(0, 1),
            "coeff": stats.uniform(0, 1),
        },
        target="gpu",
        gpu_vectorize=True,
        gpu_block_size=(256, 1, 1),
    )

    budget = ErrorBudget(limits={"l2_norm": 1e-4})

    knob_selection = KnobSelection(
        sources=True,
        overrides="overrides" in scopes,
        constants="constants" in scopes,
    )

    cfg = SelectionSearchConfig(
        experiment=experiment,
        budget=budget,
        knobs=knob_selection,
        reference="fp64",
        n_samples=10,
        n_warmup=3,
        n_reps=10,
        objective="kernel",
        compute_budget=args.max_configs,
        devices=[0, 1, 2, 3],
    )

    store = ResultStore(f"hdiff_search_{tag}.db")
    try:
        run_search(cfg, store=store)
    finally:
        store.close()


if __name__ == "__main__":
    main()
