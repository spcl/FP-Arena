# Copyright 2019-2026 ETH Zurich and the FP-Arena authors. All rights reserved.
"""
Mapping between precision keys and DaCe typeclasses, and the promotion rules for them.
"""

import re

import dace

import fp_arena  # noqa: F401

_FIXED: dict[str, dace.dtypes.typeclass] = {
    "fp16": dace.float16,
    "fp32": dace.float32,
    "fp64": dace.float64,
    "fp32sr": dace.float32sr,
    "fp64sr": dace.float64sr,
}

_MPFR_KEY = re.compile(r"^mpfr(\d+)$")


def to_typeclass(key: str) -> dace.dtypes.typeclass:
    """Resolve a precision key (e.g. ``"fp32"``, ``"mpfr128"``) to its typeclass."""
    if key in _FIXED:
        return _FIXED[key]
    m = _MPFR_KEY.match(key)
    if m:
        return dace.mpfr(int(m.group(1)))
    raise ValueError(
        f"Unknown precision key {key!r}; known: {sorted(_FIXED)} or 'mpfr<bits>'"
    )


def is_mpfr(key: str) -> bool:
    """Whether ``key`` names an MPFR precision."""
    return _MPFR_KEY.match(key) is not None


def key_of(tc: dace.dtypes.typeclass) -> str:
    """Inverse of :func:`to_typeclass`: the stable key for a typeclass."""
    if isinstance(tc, dace.mpfr):
        return f"mpfr{tc.precision}"
    for key, fixed in _FIXED.items():
        if tc == fixed:
            return key
    raise ValueError(f"No precision key registered for typeclass {tc!r}")
