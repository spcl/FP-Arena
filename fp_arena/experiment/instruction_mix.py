# Copyright 2019-2026 ETH Zurich and the FP-Arena authors. All rights reserved.
"""
Static floating-point instruction mix of a GPU build, read from its PTX.
"""

from __future__ import annotations

import functools
import os
import shutil
import subprocess
import warnings
from collections import Counter

from dace.config import Config

PTX_ARCH = 80

_GENCODE = f"-gencode arch=compute_{PTX_ARCH},code=compute_{PTX_ARCH}"

#: PTX opcodes counted as floating-point arithmetic.
_ARITHMETIC = frozenset(
    {
        "add",
        "sub",
        "mul",
        "mad",
        "fma",
        "div",
        "rcp",
        "sqrt",
        "rsqrt",
        "neg",
        "abs",
        "min",
        "max",
        "ex2",
        "lg2",
        "sin",
        "cos",
        "tanh",
        "setp",
        "set",
        "copysign",
        "testp",
    }
)

#: PTX type suffix -> (precision key, operations per instruction).
_FP_TYPES = {
    "f64": ("fp64", 1),
    "f32": ("fp32", 1),
    "f16": ("fp16", 1),
    "f16x2": ("fp16", 2),
    "bf16": ("bf16", 1),
    "bf16x2": ("bf16", 2),
}

#: The columns of a mix, in print order (``bf16`` only when present, last so
#: the others stay aligned).
MIX_KEYS = ("fp64", "fp32", "fp16", "cvt", "bf16")


def embed_ptx() -> None:
    """Make every later CUDA build embed the PTX the mix is read from."""
    args = Config.get("compiler", "cuda", "args") or ""
    if _GENCODE not in args:
        Config.set("compiler", "cuda", "args", value=f"{args} {_GENCODE}".strip())


@functools.cache
def _cuobjdump() -> str | None:
    found = shutil.which("cuobjdump")
    if found is None:
        root = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
        candidate = os.path.join(root, "bin", "cuobjdump") if root else None
        found = candidate if candidate and os.path.exists(candidate) else None
    if found is None:
        warnings.warn("cuobjdump not found: no instruction mix", stacklevel=3)
    return found


def count_mix(ptx: str) -> dict[str, int]:
    """The floating-point instruction mix of ``ptx``"""
    mix: Counter[str] = Counter()
    for line in ptx.splitlines():
        line = line.strip().lstrip("{").strip()
        if line.startswith("@"):  # predicated: drop the guard
            line = line.partition(" ")[2].strip()
        if not line or line[0] in "./}$":  # directive, comment, brace or label
            continue
        op, *mods = line.split(None, 1)[0].rstrip(";").split(".")
        if op == "cvt":
            if len(mods) >= 2 and mods[-1] in _FP_TYPES and mods[-2] in _FP_TYPES:
                mix["cvt"] += 1
        elif op in _ARITHMETIC and mods and mods[-1] in _FP_TYPES:
            key, n = _FP_TYPES[mods[-1]]
            mix[key] += n
    return {key: mix.get(key, 0) for key in MIX_KEYS}


def instruction_mix(build_folder: str) -> dict[str, int] | None:
    """
    The mix of the GPU build in ``build_folder``, or ``None`` when it has no embedded PTX.
    """
    objects = [
        os.path.join(root, name)
        for root, _, names in os.walk(os.path.join(build_folder, "build", "CMakeFiles"))
        for name in names
        if name.endswith(".cu.o")
    ]
    tool = _cuobjdump()
    if not objects or tool is None:
        return None
    out = subprocess.run(
        [tool, "-ptx", *sorted(objects)], capture_output=True, text=True, check=False
    )
    if out.returncode != 0 or ".target" not in out.stdout:
        return None
    return count_mix(out.stdout)


def format_mix(mix: dict[str, int] | None, width: int = 0) -> str:
    if mix is None:
        return ""
    shown = [k for k in MIX_KEYS if k != "bf16" or mix.get(k)]
    return ", ".join(f"{k} {mix.get(k, 0):>{width}}" for k in shown)
