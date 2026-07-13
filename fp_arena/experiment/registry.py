# Copyright 2019-2026 ETH Zurich and the FP-Arena authors. All rights reserved.
"""
Mapping between precision keys and DaCe typeclasses, and the promotion rules for them.
"""

import re
from typing import Dict

import dace

import fp_arena  # noqa: F401

_FIXED: Dict[str, dace.dtypes.typeclass] = {
    "fp16": dace.float16,
    "fp32": dace.float32,
    "fp64": dace.float64,
    "fp32sr": dace.float32sr,
    "fp64sr": dace.float64sr,
}

_MPFR_KEY = re.compile(r"^mpfr(\d+)$")
_FP_KEY = re.compile(r"^fp(\d+)_(\d+)$")


def to_typeclass(key: str) -> dace.dtypes.typeclass:
    """Resolve a precision key (e.g. ``"fp32"``, ``"mpfr128"``, ``"fp23_46"``) to its typeclass."""
    if key in _FIXED:
        return _FIXED[key]
    m = _MPFR_KEY.match(key)
    if m:
        return dace.mpfr(int(m.group(1)))
    m = _FP_KEY.match(key)
    if m:
        return dace.fp(int(m.group(1)), int(m.group(2)))
    raise ValueError(
        f"Unknown precision key {key!r}; known: {sorted(_FIXED)}, 'mpfr<bits>' or 'fp<exp>_<prec>'"
    )


def needs_mpfr_link(key: str) -> bool:
    """
    Whether ``key`` names a precision that requires linking libmpfr: the MPFR
    types always, the emulated fp types for their elemental functions.
    """
    return _MPFR_KEY.match(key) is not None or _FP_KEY.match(key) is not None


def key_of(tc: dace.dtypes.typeclass) -> str:
    """Inverse of :func:`to_typeclass`: the stable key for a typeclass."""
    if isinstance(tc, dace.mpfr):
        return f"mpfr{tc.precision}"
    if isinstance(tc, dace.fp):
        return f"fp{tc.exp_bits}_{tc.precision}"
    for key, fixed in _FIXED.items():
        if tc == fixed:
            return key
    raise ValueError(f"No precision key registered for typeclass {tc!r}")
