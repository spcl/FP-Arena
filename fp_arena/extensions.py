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

The headers are referenced by absolute path, so no ``-I`` flag or config change
is required -- this keeps the integration fully non-invasive.
"""

import os
from contextlib import contextmanager

import dace
from dace.config import Config

#: Absolute path to the bundled C++ include root (``.../runtime/include``).
INCLUDE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runtime", "include")

#: Fast-math flags removed when FP-Arena is enabled (incompatible with the
#: exact IEEE rounding that stochastic rounding depends on). ``Config.get``
#: returns the platform-resolved args, so this covers both the Linux/GCC
#: defaults (``-ffast-math``) and the Windows/MSVC defaults (``/fp:fast``).
_FAST_MATH_FLAGS = ("-ffast-math", "-ffinite-math-only", "--use_fast_math", "/fp:fast")

#: DaCe compiler-argument config paths that may carry fast-math flags.
_COMPILER_ARG_PATHS = (("compiler", "cpu", "args"), ("compiler", "cuda", "args"))

#: Headers injected into generated code.
_HEADERS = (
    os.path.join(INCLUDE_DIR, "fp_arena", "float32sr.h"),
    os.path.join(INCLUDE_DIR, "fp_arena", "float64sr.h"),
)

#: Backends whose global-code section receives the includes (CPU frame + CUDA).
_BACKENDS = ("frame", "cuda")

#: Marker so the includes are only injected once per SDFG.
_GUARD = "// fp_arena extensions enabled"

#: Substring identifying an FP-Arena C type (used to detect SR usage in an SDFG).
_CTYPE_MARKER = "fp_arena::"


def fp_arena_global_code() -> str:
    """
    :returns: the C++ ``#include`` block (absolute paths) for the SR headers.
    """
    includes = "\n".join('#include "%s"' % h for h in _HEADERS)
    return "%s\n%s\n" % (_GUARD, includes)


def inject_headers(sdfg: dace.SDFG) -> dace.SDFG:
    """
    Inject the SR header ``#include``s into ``sdfg``'s global code (CPU and CUDA).
    Idempotent.

    :param sdfg: the SDFG to inject into.
    :returns: the same SDFG, for chaining.
    """
    code = fp_arena_global_code()
    for backend in _BACKENDS:
        existing = sdfg.global_code.get(backend)
        if existing is not None and _GUARD in existing.code:
            continue
        sdfg.append_global_code(code, backend)
    return sdfg


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
    Make ``sdfg`` compile with the FP-Arena types by injecting the SR header
    includes into its generated global code, and remove ``-ffast-math`` from
    DaCe's compiler flags (persistently; see :func:`disable_fast_math`).

    Idempotent. Applies to the CPU and CUDA backends. With automatic enablement
    active (the default), calling this explicitly is optional.

    :param sdfg: the SDFG to enable FP-Arena types for.
    :returns: the same SDFG, for chaining.
    """
    disable_fast_math()
    inject_headers(sdfg)
    return sdfg


def uses_fp_arena_types(sdfg: dace.SDFG) -> bool:
    """
    :param sdfg: the SDFG to inspect.
    :returns: ``True`` if any data descriptor in ``sdfg`` or its nested SDFGs has
        an FP-Arena C type (e.g. ``float32sr``), ``False`` otherwise.
    """
    for nested in sdfg.all_sdfgs_recursive():
        for desc in nested.arrays.values():
            if _CTYPE_MARKER in (getattr(desc.dtype, "ctype", "") or ""):
                return True
    return False


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

    * Header injection (a code-generation concern) wraps the single codegen
      entry point ``dace.codegen.codegen.generate_code``, so headers are added
      on every path -- ``compile``, ``SDFG.generate_code``, or a direct codegen
      call. ``compile`` code-generates a deep copy, so the caller's SDFG object
      is left untouched.
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
        if uses_fp_arena_types(sdfg):
            inject_headers(sdfg)
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
