# Copyright 2019-2026 ETH Zurich and the FP-Arena authors. All rights reserved.
"""
Search for the fastest per-array precision assignment for the vertical advection
(vadv) kernel that stays within budget, using the pruning selection *search*
(``run_search``).

``--knobs`` selects which knobs are in scope:
  * ``sources``   -- read inputs only (the default);
  * ``overrides`` -- also intermediates and outputs (many more knobs);
  * ``constants`` -- also the float literals in the kernel.
``--exhaustive`` evaluates and times every config instead of searching.
Results go to ``vadv_search_<knobs>.db`` under experiment name ``vadv_<knobs>``.
"""

import argparse
import zlib

import dace as dc
import numpy as np
from dace.transformation.auto import auto_optimize as aopt
from dace.transformation.dataflow import MapFusion, PruneConnectors
from dace.transformation.interstate import LoopToMap
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

I, J, K = 128, 128, 64


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
        help="evaluate and time every config of the lattice instead of searching ",
    )
    args = parser.parse_args()
    scopes = set(args.knobs)
    tag = "_".join(s for s in ("overrides", "constants") if s in scopes) or "sources"

    sdfg = vadv.to_sdfg(simplify=True)
    sdfg.specialize({"I": I, "J": J, "K": K})
    sdfg = aopt.auto_optimize(sdfg, dc.DeviceType.CPU)
    sdfg.apply_transformations_repeated(LoopToMap)
    sdfg.apply_transformations_repeated(MapFusion)
    sdfg.apply_transformations_repeated(PruneConnectors)
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
        exhaustive=args.exhaustive,
        name=f"{experiment.name}_exhaustive" if args.exhaustive else None,
    )

    store = ResultStore(f"vadv_search_{tag}.db")
    try:
        run_search(cfg, store=store)
    finally:
        store.close()


if __name__ == "__main__":
    main()
