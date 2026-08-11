# Copyright 2024-2026 ETH Zurich and the FP-Arena authors. All rights reserved.
"""
FP-Arena: a rapid-prototyping environment for testing custom floating-point
types in DaCe.

Importing this package registers the FP-Arena types (e.g. ``float32sr``,
``float64sr``) into DaCe without modifying DaCe itself, so they can be used like
any built-in DaCe scalar type. Call :func:`enable_fp_arena_extensions` on an
SDFG before compiling it to attach the DaCe environments that pull in the
matching C++ headers and link flags.
"""

from fp_arena.dtypes import (
    FP_ARENA_TYPECLASSES,
    Float32sr,
    Float64sr,
    float32sr,
    float64sr,
    mpfr,
    register,
)
from fp_arena.environments import INCLUDE_DIR, MPFR, FPArenaSR
from fp_arena.extensions import (
    attach_environments,
    disable_auto_extensions,
    disable_fast_math,
    enable_auto_extensions,
    enable_fp_arena_extensions,
    patch_aligned_heap_allocation,
    patch_memcpy_copies,
    precise_math,
    required_environments,
    uses_fp_arena_types,
)
from fp_arena.transformations.change_and_propagate_fp_types import (
    DEFAULT_PROMOTION_RULES,
    change_and_propagate_fp_types,
)
from fp_arena.transformations.change_fp_types import change_fptype

# Register the types and the SDFG convenience method on import (idempotent).
register()

patch_aligned_heap_allocation()
patch_memcpy_copies()

import dace as _dace

if not hasattr(_dace.SDFG, "enable_fp_arena_extensions"):
    _dace.SDFG.enable_fp_arena_extensions = lambda self: enable_fp_arena_extensions(
        self
    )

# Automatically enable FP-Arena for any SDFG that uses its types, so the
# explicit call above becomes optional. Disable with disable_auto_extensions().
enable_auto_extensions()

__all__ = [
    "DEFAULT_PROMOTION_RULES",
    "FP_ARENA_TYPECLASSES",
    "INCLUDE_DIR",
    "MPFR",
    "FPArenaSR",
    "Float32sr",
    "Float64sr",
    "attach_environments",
    "change_and_propagate_fp_types",
    "change_fptype",
    "disable_auto_extensions",
    "disable_fast_math",
    "enable_auto_extensions",
    "enable_fp_arena_extensions",
    "float32sr",
    "float64sr",
    "mpfr",
    "patch_aligned_heap_allocation",
    "patch_memcpy_copies",
    "precise_math",
    "register",
    "required_environments",
    "uses_fp_arena_types",
]
