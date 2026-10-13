# Copyright 2019-2026 ETH Zurich and the FP-Arena authors. All rights reserved.
"""
Search for the fastest per-array precision assignment for the vertical advection
(vadv) kernel that stays within budget, using the pruning selection *search*
(``run_search``).

``--knobs`` selects which knobs are in scope:
  * ``sources``   -- read inputs only (the default);
  * ``overrides`` -- also intermediates and outputs (many more knobs);
  * ``constants`` -- also the float literals in the kernel.
``--snr-db`` sets the error budget: every output must stay at least this close
to the fp64 result, as a variance SNR in dB (default 60)
``--exhaustive`` evaluates and times every config instead of searching.
``--prune-above-feasible`` also skips configs a feasible config lowers at least
as far (assumes more lowering is never slower).
Results go to ``vadv_<knobs>_<level>dB-<job id>.db`` (``..._exhaustive-<job id>.db`` with
``--exhaustive``), one database per run.
"""

import argparse
import os
import zlib

import dace as dc
import numpy as np
from dace.transformation.auto import auto_optimize as aopt
from dace.transformation.dataflow import (
    MapFusion,
    PruneConnectors,
    TrivialMapElimination,
)
from dace.transformation.interstate import LoopToMap, MoveLoopIntoMap
from dace.transformation.passes.insert_explicit_copies import InsertExplicitCopies
from scipy import stats

from corpus.vdav import dtr_stage, vadv
from fp_arena.experiment import (
    ErrorBudget,
    ExperimentConfig,
    KnobSelection,
    ResultStore,
    SelectionSearchConfig,
    run_search,
)

I, J, K = 1024, 1024, 512


def _shared(rng: np.random.Generator, name: str) -> np.random.Generator:
    """
    A generator for values that several inputs of one sample share.
    """
    seed = rng.bit_generator.seed_seq
    key = (*seed.spawn_key, zlib.crc32(name.encode()))
    return np.random.default_rng(np.random.SeedSequence(seed.entropy, spawn_key=key))


def _wind(shape, rng):
    profile = np.linspace(20.0, 5.0, shape[2])
    return profile + _shared(rng, "vadv wind").normal(0.0, 1.0, shape)


INPUTS = {
    "u_pos": _wind,
    "u_stage": _wind,
    "wcon": stats.norm(0.0, 0.05 * dtr_stage),
    "utens": stats.norm(0.0, 1e-4),
    "utens_stage": stats.norm(0.0, 1e-4),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="vadv precision selection search")
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
        "--prune-above-feasible",
        action="store_true",
        help="skip configs a feasible config lowers at least as far (assumes more "
        "lowering is never slower); the search then stops at the most-lowered "
        "feasible configs instead of running to exhaustion.",
    )
    parser.add_argument(
        "--snr-db",
        type=float,
        default=60.0,
    )
    args = parser.parse_args()
    scopes = set(args.knobs)
    tag = "_".join(s for s in ("overrides", "constants") if s in scopes) or "sources"

    sdfg = vadv.to_sdfg(simplify=True)
    for desc in sdfg.arrays.values():
        if isinstance(desc, dc.data.Array) and len(desc.shape) == 3:
            n0, n1, _ = desc.shape
            desc.strides = (n1, 1, n0 * n1)
    sdfg = aopt.auto_optimize(sdfg, dc.DeviceType.CPU)
    sdfg.apply_transformations_repeated(LoopToMap)
    sdfg.apply_transformations_repeated(TrivialMapElimination)
    sdfg.apply_transformations_repeated(MapFusion)
    sdfg.apply_transformations_repeated(PruneConnectors)
    sdfg.simplify()
    InsertExplicitCopies().apply_pass(sdfg, {})
    sdfg.apply_transformations_repeated(MoveLoopIntoMap, options={"cfg_body": True})
    sdfg.apply_transformations_repeated(MapFusion)
    sdfg.simplify()

    experiment = ExperimentConfig(
        name=f"vadv_{tag}",
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
        n_samples=7,
        n_warmup=3,
        n_reps=10,
        objective="kernel",
        compute_budget=args.max_configs,
        devices=[0, 1, 2, 3],
        exhaustive=args.exhaustive,
        prune_above_feasible=args.prune_above_feasible,
        name=run_name,
    )

    store = ResultStore(f"{run_name}-{os.environ.get('SLURM_JOB_ID', 'local')}.db")
    try:
        run_search(cfg, store=store)
    finally:
        store.close()


if __name__ == "__main__":
    main()
