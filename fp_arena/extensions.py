# Copyright 2024-2026 ETH Zurich and the FP-Arena authors. All rights reserved.
"""
Enablement of the FP-Arena runtime for SDFG code generation.

DaCe only needs ``dtype.ctype`` strings to generate code that uses the FP-Arena
value types, but the generated C++ must ``#include`` the matching headers and be
compiled without ``-ffast-math`` (stochastic rounding needs exact IEEE
rounding).

Two ways to apply this:

* Automatic (default, installed by ``import fp_arena``): the codegen entry point
  is wrapped to inject the headers (a code-generation concern) and
  ``SDFG.compile`` is wrapped to strip fast-math (a build concern), both only for
  SDFGs that actually use an FP-Arena dtype. SDFGs that do not are untouched.
  Toggle with :func:`enable_auto_extensions` / :func:`disable_auto_extensions`.
* Explicit: call :func:`enable_fp_arena_extensions` on an SDFG (it also strips
  fast-math from DaCe's config as a persistent default).
"""

from collections.abc import Iterator
from contextlib import contextmanager

import dace
from dace.config import Config

from fp_arena.dtypes import mpfr
from fp_arena.environments import MPFR, FPArenaSR

#: Fast-math flags removed when FP-Arena is enabled (incompatible with the
#: exact IEEE rounding that stochastic rounding depends on). ``Config.get``
#: returns the platform-resolved args, so this covers both the Linux/GCC
#: defaults (``-ffast-math``) and the Windows/MSVC defaults (``/fp:fast``).
_FAST_MATH_FLAGS = ("-ffast-math", "-ffinite-math-only", "--use_fast_math", "/fp:fast")

#: DaCe compiler-argument config paths that may carry fast-math flags.
_COMPILER_ARG_PATHS = (("compiler", "cpu", "args"), ("compiler", "cuda", "args"))

#: The FP-Arena environments, each with the C-type substrings that require it
#: (``fp_arena::`` for the SR types, ``dace::mpfr`` for MPFR).
_ENVIRONMENTS = (
    (FPArenaSR, ("fp_arena::",)),
    (MPFR, ("dace::mpfr",)),
)


def _type_strings(sdfg: dace.SDFG) -> Iterator[str]:
    """
    :param sdfg: the SDFG to inspect.
    :returns: an iterator over every string in ``sdfg`` and its nested SDFGs that
        may name a C type: data descriptor C types and tasklet bodies.
    """
    from dace.sdfg import nodes as _dnodes

    for nested in sdfg.all_sdfgs_recursive():
        for desc in nested.arrays.values():
            yield getattr(desc.dtype, "ctype", "") or ""
        for state in nested.states():
            for node in state.nodes():
                if isinstance(node, _dnodes.Tasklet):
                    try:
                        yield node.code.as_string
                    except AttributeError:
                        yield str(node.code)


def required_environments(sdfg: dace.SDFG) -> set[str]:
    """
    :param sdfg: the SDFG to inspect.
    :returns: the full class paths (DaCe's environment identifiers) of the
        FP-Arena environments the types used by ``sdfg`` need -- empty if it uses
        none of them.
    """
    strings = list(_type_strings(sdfg))
    return {
        env.full_class_path()
        for env, markers in _ENVIRONMENTS
        if any(marker in s for s in strings for marker in markers)
    }


def attach_environments(sdfg: dace.SDFG) -> dace.SDFG:
    """
    Attach the FP-Arena environments that ``sdfg`` needs to its code nodes.

    :param sdfg: the SDFG to attach to.
    :returns: the same SDFG, for chaining.
    """
    from dace.sdfg import nodes as _dnodes

    envs = required_environments(sdfg)
    if not envs:
        return sdfg
    for nested in sdfg.all_sdfgs_recursive():
        for state in nested.states():
            for node in state.nodes():
                if isinstance(node, _dnodes.CodeNode):
                    node.environments = frozenset(node.environments) | envs
    return sdfg


#: Whether the aligned-allocation patch is installed.
_aligned_patch_installed = False


def patch_aligned_heap_allocation() -> bool:
    """Route mpfr heap arrays through plain ``new[]`` / ``delete[]``, which run destructors."""

    global _aligned_patch_installed
    if _aligned_patch_installed:
        return False

    from dace.codegen.targets import cpu, experimental_cpu

    original = cpu.use_aligned_operator_new

    def _use_aligned_operator_new(desc) -> bool:
        if isinstance(desc.dtype, mpfr):
            return False
        return original(desc)

    cpu.use_aligned_operator_new = _use_aligned_operator_new
    experimental_cpu.use_aligned_operator_new = _use_aligned_operator_new

    _aligned_patch_installed = True
    return True


#: Copy implementations that emit a raw ``memcpy``, which shallow-copies mpfr's limb pointer.
_MEMCPY_IMPLEMENTATIONS = frozenset(
    {"MemcpyCPU", "MemcpyCUDA1D", "MemcpyCUDA2D", "MemcpyCUDANDStrided"}
)

#: Whether the element-wise copy patch is installed.
_copy_patch_installed = False


def patch_memcpy_copies() -> bool:
    """Route mpfr array copies through element-wise assignment, which deep-copies."""

    global _copy_patch_installed
    if _copy_patch_installed:
        return False

    from dace.libraries.standard.nodes import copy as copy_lib
    from dace.libraries.standard.nodes.copy import select as copy_select
    from dace.libraries.standard.nodes.copy.expansions import auto as copy_auto

    original = copy_select.select_copy_implementation

    def _select_copy_implementation(node, parent_state) -> str:
        impl = original(node, parent_state)
        if impl not in _MEMCPY_IMPLEMENTATIONS:
            return impl
        _, inp, in_subset, _, _, out_subset = node.validate(
            parent_state.sdfg, parent_state, allow_cross_storage=True
        )
        if not isinstance(inp.dtype, mpfr):
            return impl
        single = (
            in_subset.num_elements_exact() == 1 and out_subset.num_elements_exact() == 1
        )
        return "Tasklet" if single else "MappedTasklet"

    copy_select.select_copy_implementation = _select_copy_implementation
    copy_lib.select_copy_implementation = _select_copy_implementation
    copy_auto.select_copy_implementation = _select_copy_implementation

    _copy_patch_installed = True
    return True


def _strip_fast_math(args: str) -> str:
    """:returns: ``args`` with any fast-math flag removed."""
    return " ".join(tok for tok in args.split() if tok not in _FAST_MATH_FLAGS)


def disable_fast_math():
    """
    Remove fast-math flags from DaCe's compiler arguments (persistent default).

    Stochastic rounding needs the exact IEEE-754 rounding that ``-ffast-math``
    (and ``--use_fast_math``) break, so this is required for the FP-Arena types
    to behave correctly. It reads DaCe's compiler-arg config entries and writes
    back the same flags minus any fast-math option. Idempotent.

    :returns: ``True`` if any flag was removed, ``False`` otherwise.
    """
    changed = False
    for path in _COMPILER_ARG_PATHS:
        args = Config.get(*path)
        if not args:
            continue
        new_args = _strip_fast_math(args)
        if new_args != args:
            Config.set(*path, value=new_args)
            changed = True
    return changed


@contextmanager
def precise_math():
    """
    Context manager that removes fast-math flags from DaCe's compiler args for
    its duration only, restoring the previous values on exit.
    """
    saved = {path: Config.get(*path) for path in _COMPILER_ARG_PATHS}
    try:
        for path, args in saved.items():
            if args:
                Config.set(*path, value=_strip_fast_math(args))
        yield
    finally:
        for path, args in saved.items():
            Config.set(*path, value=args)


def enable_fp_arena_extensions(sdfg: dace.SDFG) -> dace.SDFG:
    """
    Make ``sdfg`` compile with the FP-Arena types by attaching the FP-Arena
    environments and remove ``-ffast-math`` from DaCe's compiler flags (persistently;
    see :func:`disable_fast_math`).

    Idempotent. With automatic enablement active (the default), calling this
    explicitly is optional.

    :param sdfg: the SDFG to enable FP-Arena types for.
    :returns: the same SDFG, for chaining.
    """
    disable_fast_math()
    attach_environments(sdfg)
    return sdfg


def uses_fp_arena_types(sdfg: dace.SDFG) -> bool:
    """
    :param sdfg: the SDFG to inspect.
    :returns: ``True`` if any data descriptor or tasklet body in ``sdfg`` or its
        nested SDFGs references an FP-Arena C type, ``False`` otherwise.
    """
    return bool(required_environments(sdfg))


#: Whether the automatic wrappers are currently installed.
_auto_installed = False

#: Originals saved before wrapping.
_original_compile = None
_original_generate_code = None


def enable_auto_extensions():
    """
    Automatically enable any SDFG that uses an FP-Arena dtype. Idempotent;
    installed automatically by ``import fp_arena``.

    Two wrappers are installed, each at the layer its concern belongs to:

    * Attaching the environments (a code-generation concern) wraps the single
      codegen entry point ``dace.codegen.codegen.generate_code``, so they are
      attached on every path -- ``compile``, ``SDFG.generate_code``, or a direct
      codegen call. ``compile`` code-generates a deep copy, so the caller's SDFG
      object is left untouched.
    * Removing ``-ffast-math`` (a build concern) wraps ``SDFG.compile``, scoped
      to the build of FP-Arena SDFGs only.

    SDFGs that do not use FP-Arena types are generated and compiled unchanged.
    """
    global _auto_installed, _original_compile, _original_generate_code
    if _auto_installed:
        return
    from dace.codegen import codegen

    _original_generate_code = codegen.generate_code

    def _generate_code(sdfg, *args, **kwargs):
        attach_environments(sdfg)
        return _original_generate_code(sdfg, *args, **kwargs)

    codegen.generate_code = _generate_code

    _original_compile = dace.SDFG.compile

    def _compile(self, *args, **kwargs):
        if uses_fp_arena_types(self):
            with precise_math():
                return _original_compile(self, *args, **kwargs)
        return _original_compile(self, *args, **kwargs)

    dace.SDFG.compile = _compile
    _auto_installed = True


def disable_auto_extensions():
    """Uninstall the automatic codegen/compile wrappers (see :func:`enable_auto_extensions`)."""
    global _auto_installed, _original_compile, _original_generate_code
    if not _auto_installed:
        return
    from dace.codegen import codegen

    codegen.generate_code = _original_generate_code
    dace.SDFG.compile = _original_compile
    _original_generate_code = None
    _original_compile = None
    _auto_installed = False
