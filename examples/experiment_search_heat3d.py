# Copyright 2019-2026 ETH Zurich and the FP-Arena authors. All rights reserved.
"""
Search for the fastest per-array precision assignment for heat3d that stays
within budget, using the pruning selection *search* (``run_search``).

``--knobs`` selects which knobs are in scope:
  * ``sources``   -- read inputs only (the default);
  * ``overrides`` -- also intermediates and outputs (many more knobs);
  * ``constants`` -- also the float literals in the kernel.
``--snr-db`` sets the error budget: every output must stay at least this close
to the fp64 result, as a variance SNR in dB (default 60)
``--exhaustive`` evaluates and times every config instead of searching.
Results go to ``heat3d_<knobs>_<level>dB-<job id>.db`` (``..._exhaustive-<job id>.db`` with
``--exhaustive``), one database per run.
"""

import argparse
import os
import zlib

import numpy as np
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

from corpus.heat3d import heat3d_kernel
from fp_arena.experiment import (
    ErrorBudget,
    ExperimentConfig,
    KnobSelection,
    ResultStore,
    SelectionSearchConfig,
    run_search,
)

GRID_N = 512
TSTEPS = 20


def _shared(rng: np.random.Generator, name: str) -> np.random.Generator:
    """
    A generator for values that several inputs of one sample share.
    """
    seed = rng.bit_generator.seed_seq
    key = (*seed.spawn_key, zlib.crc32(name.encode()))
    return np.random.default_rng(np.random.SeedSequence(seed.entropy, spawn_key=key))


def _field(shape, rng):
    return _shared(rng, "heat3d").normal(50.0, 10.0, shape)


INPUTS = {"A": _field, "B": _field}


def main() -> None:
    parser = argparse.ArgumentParser(description="heat3d precision selection search")
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

    sdfg = heat3d_kernel.to_sdfg(simplify=True)
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
        name=f"heat3d_{tag}",
        program=sdfg,
        symbols={"N": GRID_N},
        scalar_args={"TSTEPS": TSTEPS},
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
