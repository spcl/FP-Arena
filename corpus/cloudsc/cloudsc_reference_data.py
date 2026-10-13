# Copyright 2019-2026 ETH Zurich and the FP-Arena authors. All rights reserved.
"""CloudSC inputs read from the ECMWF dwarf's own reference dataset"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import dace
import h5py
import numpy as np

CLOUDSC_SPECIES: dict[str, int] = {
    "nclv": 5,
    "ncldql": 1,  # liquid cloud water
    "ncldqi": 2,  # ice cloud water
    "ncldqr": 3,  # rain water
    "ncldqs": 4,  # snow
    "ncldqv": 5,  # vapour
}

REFERENCE_KLEV: int = 137
REFERENCE_KLON: int = 100

_DATA_DIR = Path(__file__).resolve().parent

#: YDTHF constant names in ``input.h5``; every other bare name is YDCST.
_YDTHF_NAMES: frozenset[str] = frozenset(
    {
        "r2es",
        "r3les",
        "r3ies",
        "r4les",
        "r4ies",
        "r5les",
        "r5ies",
        "r5alvcp",
        "r5alscp",
        "ralvdcp",
        "ralsdcp",
        "ralfdcp",
        "rtwat",
        "rtice",
        "rticecu",
        "rtwat_rtice_r",
        "rtwat_rticecu_r",
        "rkoop1",
        "rkoop2",
    }
)


@lru_cache(maxsize=2)
def _load(kind: str) -> dict[str, np.ndarray]:
    """``corpus/cloudsc/{kind}.h5`` as ``{lowercase name: array}``, booleans widened to int32."""
    state: dict[str, np.ndarray] = {}
    with h5py.File(_DATA_DIR / f"{kind}.h5", "r") as handle:
        for name, dataset in handle.items():
            values = np.asarray(dataset[...])
            if values.dtype == np.bool_:
                values = values.astype(np.int32)
            state[name.lower()] = values
    return state


@lru_cache(maxsize=1)
def _constants() -> dict[str, float | int]:
    """Every length-1 dataset of ``input.h5`` under its kernel argument name."""
    constants: dict[str, float | int] = {}
    for name, values in _load("input").items():
        if values.size != 1 or name in ("klev", "klon", "kfldx"):
            continue
        if not (name.startswith(("yrecldp_", "yrephli_")) or name == "ptsphy"):
            name = f"ydthf_{name}" if name in _YDTHF_NAMES else f"ydcst_{name}"
        value = values.reshape(())[()]
        constants[name] = (
            int(value) if np.issubdtype(values.dtype, np.integer) else float(value)
        )
    return constants


def reference_free_symbols(
    sdfg: Any, klon: int, klev: int = REFERENCE_KLEV
) -> dict[str, float | int]:
    """Values for exactly the free symbols ``sdfg`` declares."""
    if klev != REFERENCE_KLEV:
        raise ValueError(
            f"input.h5 only has {REFERENCE_KLEV}-level profiles, got klev={klev}"
        )
    known = {
        "klon": int(klon),
        "klev": int(klev),
        **CLOUDSC_SPECIES,
        "kidia": 1,
        "kfdia": int(klon),
        **_constants(),
    }
    free = {str(s) for s in sdfg.free_symbols}
    missing = free - known.keys()
    if missing:
        raise KeyError(
            f"No value for free symbol(s) {sorted(missing)} of the CloudSC SDFG"
        )
    return {name: known[name] for name in sorted(free)}


def reference_scalar_args(sdfg: Any) -> dict[str, float | int]:
    """The non-transient scalar arguments ``sdfg`` declares, filled from ``input.h5``."""
    constants = _constants()
    names = [
        name
        for name, desc in sdfg.arrays.items()
        if not desc.transient and not isinstance(desc, dace.data.Array)
    ]
    missing = [name for name in names if name not in constants]
    if missing:
        raise KeyError(
            f"input.h5 has no value for scalar argument(s) {sorted(missing)}"
        )
    return {name: constants[name] for name in names}


class _ZerosGenerator:
    """Zeros for the arrays ``input.h5`` lacks: outputs and flux / tendency accumulators."""

    def __call__(self, shape, rng):
        return np.zeros(tuple(int(d) for d in shape))


class _ReplicatedGenerator:
    """``source`` with its 100 columns replicated to ``klon``.

    A class, not a closure, so the config pickles to the search's spawned workers.
    """

    def __init__(self, source: np.ndarray, klon: int):
        self.source = source
        self.klon = klon

    def __call__(self, shape, rng):
        if int(shape[-1]) != self.klon:
            raise ValueError(
                f"array shape {shape} does not have klon={self.klon} columns"
            )
        columns = np.arange(self.klon) % self.source.shape[-1]
        return np.ascontiguousarray(self.source[..., columns])


def reference_input_distributions(sdfg: Any, klon: int) -> dict[str, Any]:
    """A ``(shape, rng) -> ndarray`` generator for every non-transient array of ``sdfg``."""
    state = _load("input")
    distributions: dict[str, Any] = {}
    for name, desc in sdfg.arrays.items():
        if desc.transient or not isinstance(desc, dace.data.Array):
            continue
        source = state.get(name.removeprefix("ydcst_").removeprefix("ydthf_"))
        distributions[name] = (
            _ZerosGenerator() if source is None else _ReplicatedGenerator(source, klon)
        )
    return distributions


def reference_outputs() -> dict[str, np.ndarray]:
    """The dwarf's expected outputs (``reference.h5``) at ``klon = 100``."""
    return dict(_load("reference"))
