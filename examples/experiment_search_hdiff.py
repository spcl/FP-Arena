# Copyright 2019-2026 ETH Zurich and the FP-Arena authors. All rights reserved.
"""
Search for the fastest per-array precision assignment for the horizontal
diffusion (hdiff) stencil that stays within budget, using the pruning selection
*search* (``run_search``).

``--knobs`` selects which knobs are in scope:
  * ``sources``   -- read inputs only (the default);
  * ``overrides`` -- also intermediates and outputs (many more knobs);
  * ``constants`` -- also the float literals in the kernel.
``--snr-db`` sets the error budget: every output must stay at least this close
to the fp64 result, as a variance SNR in dB (default 60)
``--exhaustive`` evaluates and times every config instead of searching.
Results go to ``hdiff_<knobs>_<level>dB-<job id>.db`` (``..._exhaustive-<job id>.db`` with
``--exhaustive``), one database per run.
"""

import argparse
import os

from dace.transformation.dataflow import (
    MapCollapse,
    MapFusion,
    OTFMapFusion,
    PruneConnectors,
    RedundantArray,
    RedundantSecondArray,
    TrivialMapElimination,
)
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

I, J, K = 1024, 1024, 128


INPUTS = {
    "in_field": stats.norm(7.0, 1.0),
    "coeff": stats.uniform(0.02, 0.01),
}


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
    parser.add_argument(
        "--exhaustive",
        action="store_true",
        help="evaluate and time every config of the lattice instead of searching ",
    )
    parser.add_argument(
        "--snr-db",
        type=float,
        default=60.0,
    )
    args = parser.parse_args()
    scopes = set(args.knobs)
    tag = "_".join(s for s in ("overrides", "constants") if s in scopes) or "sources"

    sdfg = hdiff.to_sdfg(simplify=True)
    sdfg.simplify()
    sdfg.apply_transformations_repeated(LoopToMap)
    sdfg.simplify()
    sdfg.apply_transformations_repeated(MapCollapse)
    sdfg.apply_transformations_repeated(LoopToMap)
    sdfg.apply_transformations_repeated(TrivialMapElimination)
    sdfg.apply_transformations_repeated(OTFMapFusion)
    sdfg.apply_transformations_repeated(MapFusion)
    sdfg.apply_transformations_repeated(PruneConnectors)
    sdfg.simplify()
    sdfg.apply_transformations_repeated(RedundantArray)
    sdfg.apply_transformations_repeated(RedundantSecondArray)

    experiment = ExperimentConfig(
        name=f"hdiff_{tag}",
        program=sdfg,
        symbols={"I": I, "J": J, "K": K},
        inputs=INPUTS,
        target="gpu",
        gpu_vectorize=True,
        gpu_block_size=(256, 1, 1),
    )

    budget = ErrorBudget(limits={"snr_var": args.snr_db})

    knob_selection = KnobSelection(
        sources=True,
        overrides="overrides" in scopes,
        constants="constants" in scopes,
    )

    run_name = (
        f"{experiment.name}_exhaustive"
        if args.exhaustive
        else f"{experiment.name}_{args.snr_db:g}dB"
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
        exhaustive=args.exhaustive,
        name=run_name,
    )

    store = ResultStore(f"{run_name}-{os.environ.get('SLURM_JOB_ID', 'local')}.db")
    try:
        run_search(cfg, store=store)
    finally:
        store.close()


if __name__ == "__main__":
    main()
