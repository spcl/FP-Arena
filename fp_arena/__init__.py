# Copyright 2024-2026 ETH Zurich and the FP-Arena authors. All rights reserved.
"""
FP-Arena: a rapid-prototyping environment for testing custom floating-point
types in DaCe.

Importing this package registers the FP-Arena types (e.g. ``float32sr``,
``float64sr``) into DaCe without modifying DaCe itself, so they can be used like
any built-in DaCe scalar type. Call :func:`enable_fp_arena_extensions` on an
SDFG before compiling it to pull in the matching C++ headers.
"""

from fp_arena.dtypes import (
    Float32sr,
    Float64sr,
    float32sr,
    float64sr,
    mpfr,
    register,
    FP_ARENA_TYPECLASSES,
)
from fp_arena.extensions import (
    enable_fp_arena_extensions,
    enable_auto_extensions,
    disable_auto_extensions,
    disable_fast_math,
    inject_headers,
    precise_math,
    uses_fp_arena_types,
    fp_arena_global_code,
    INCLUDE_DIR,
)
from fp_arena.transformations.change_fp_types import change_fptype
from fp_arena.transformations.change_and_propagate_fp_types import (
    DEFAULT_PROMOTION_RULES,
    change_and_propagate_fp_types,
)

# Register the types and the SDFG convenience method on import (idempotent).
register()

import dace as _dace

if not hasattr(_dace.SDFG, "enable_fp_arena_extensions"):
    _dace.SDFG.enable_fp_arena_extensions = lambda self: enable_fp_arena_extensions(self)

# Automatically enable FP-Arena for any SDFG that uses its types, so the
# explicit call above becomes optional. Disable with disable_auto_extensions().
enable_auto_extensions()

__all__ = [
    "Float32sr",
    "Float64sr",
    "float32sr",
    "float64sr",
    "mpfr",
    "register",
    "FP_ARENA_TYPECLASSES",
    "enable_fp_arena_extensions",
    "enable_auto_extensions",
    "disable_auto_extensions",
    "disable_fast_math",
    "inject_headers",
    "precise_math",
    "uses_fp_arena_types",
    "fp_arena_global_code",
    "INCLUDE_DIR",
    "change_fptype",
    "change_and_propagate_fp_types",
    "DEFAULT_PROMOTION_RULES",
]
