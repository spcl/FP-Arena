# Copyright 2019-2026 ETH Zurich and the FP-Arena authors. All rights reserved.
"""
Search for the fastest per-array precision assignment for the ECMWF CloudSC
cloud-microphysics dwarf that stays within budget, using the pruning selection
*search* (``run_search``).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import dace as dc

from corpus.cloudsc.cloudsc_noise_floor import CLOUDSC_STATES, timestep_snr_floor
from corpus.cloudsc.cloudsc_reference_data import (
    REFERENCE_KLEV,
    reference_free_symbols,
    reference_input_distributions,
    reference_scalar_args,
)
from corpus.cloudsc.offload_cloudsc_to_gpu import offload_cloudsc_to_gpu
from fp_arena.search import (
    ErrorBudget,
    ExperimentConfig,
    KnobSelection,
    ResultStore,
    SelectionSearchConfig,
    run_search,
)

KLON = 262144
KLEV = REFERENCE_KLEV

# Physics step in seconds; None keeps input.h5's ptsphy (3600 s).
PTSPHY: float | None = None

# canon_cpu from the dace extended branch (tests/corpus/cloudsc/reproduce.py).
SDFG_PATH = Path(__file__).resolve().parent.parent / "corpus" / "cloudsc" / "CLOUDSC-CPU.sdfg"


def main() -> None:
    parser = argparse.ArgumentParser(description="cloudsc precision selection search")
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
        "--prune-above-feasible",
        action="store_true",
        help="skip configs a feasible config lowers at least as far (assumes more "
        "lowering is never slower); the search then stops at the most-lowered "
        "feasible configs instead of running to exhaustion.",
    )
    parser.add_argument(
        "--all-outputs",
        action="store_true",
        help="check every array the kernel writes (tendencies, fluxes, ...) against "
        "its own time-step noise floor, not only the end-of-step state.",
    )
    args = parser.parse_args()
    scopes = set(args.knobs)
    tag = "_".join(s for s in ("overrides", "constants") if s in scopes) or "sources"
    if args.all_outputs:
        tag += "_all_outputs"

    sdfg = dc.SDFG.from_file(SDFG_PATH)

    scalar_args = reference_scalar_args(sdfg)
    if PTSPHY is not None:
        scalar_args["ptsphy"] = PTSPHY

    experiment = ExperimentConfig(
        name=f"cloudsc_{tag}",
        program=sdfg,
        symbols=reference_free_symbols(sdfg, klon=KLON, klev=KLEV),
        scalar_args=scalar_args,
        inputs=reference_input_distributions(sdfg, klon=KLON),
        target="gpu",
        gpu_offload=offload_cloudsc_to_gpu,
        gpu_vectorize=False,
        gpu_block_size=(256, 1, 1),
        derived_outputs=CLOUDSC_STATES,
    )

    # A point is feasible when its end-of-step state (and with --all-outputs
    # every written array) stays at least as close to fp64 as one step and two
    # half steps are to each other (variance SNR).
    floor = timestep_snr_floor(sdfg, nsub=2, dt=PTSPHY, outputs=args.all_outputs)
    for state, db in floor.items():
        print(f"  noise floor {state:10s} {db:7.2f} dB")
    budget = ErrorBudget(limits={"snr_var": floor}, arrays=list(floor))

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
        n_samples=1,
        n_warmup=2,
        n_reps=4,
        objective="kernel",
        compute_budget=args.max_configs,
        prune_above_feasible=args.prune_above_feasible,
        devices=[0, 1, 2, 3],
        compile_workers=128,
    )

    store = ResultStore(
        f"cloudsc_search_{tag}-{os.environ.get('SLURM_JOB_ID', 'local')}.db"
    )
    try:
        run_search(cfg, store=store)
    finally:
        store.close()


if __name__ == "__main__":
    main()
