# Copyright 2019-2026 ETH Zurich and the FP-Arena authors. All rights reserved.
"""
DaCe library environments for the FP-Arena C++ runtime.
"""

from __future__ import annotations

import os
from typing import ClassVar

import dace.library

INCLUDE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "runtime", "include"
)


@dace.library.environment
class FPArenaSR:
    """
    The header-only stochastic-rounding value types (``fp_arena::float32sr`` and ``fp_arena::float64sr``).
    """

    cmake_minimum_version: ClassVar[str | None] = None
    cmake_packages: ClassVar[list] = []
    cmake_variables: ClassVar[dict] = {}
    cmake_includes: ClassVar[list] = [INCLUDE_DIR]
    cmake_libraries: ClassVar[list] = []
    cmake_compile_flags: ClassVar[list] = []
    cmake_link_flags: ClassVar[list] = []
    cmake_files: ClassVar[list] = []

    headers: ClassVar[dict] = {
        "frame": ["fp_arena/float32sr.h", "fp_arena/float64sr.h"],
        "cuda": ["fp_arena/float32sr.h", "fp_arena/float64sr.h"],
    }
    state_fields: ClassVar[list] = []
    init_code: ClassVar[str] = ""
    finalize_code: ClassVar[str] = ""
    dependencies: ClassVar[list] = []


@dace.library.environment
class MPFR:
    """
    The ``dace::mpfr<P>`` wrapper and the ``libmpfr`` it calls into.
    """

    cmake_minimum_version: ClassVar[str | None] = None
    cmake_packages: ClassVar[list] = []
    cmake_variables: ClassVar[dict] = {}
    cmake_includes: ClassVar[list] = [INCLUDE_DIR]
    cmake_libraries: ClassVar[list] = ["mpfr"]
    cmake_compile_flags: ClassVar[list] = []
    cmake_link_flags: ClassVar[list] = []
    cmake_files: ClassVar[list] = []

    headers: ClassVar[dict] = {"frame": ["fp_arena/mpfr.h"]}
    state_fields: ClassVar[list] = []
    init_code: ClassVar[str] = ""
    finalize_code: ClassVar[str] = ""
    dependencies: ClassVar[list] = []
