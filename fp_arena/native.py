# Copyright 2019-2026 ETH Zurich and the FP-Arena authors. All rights reserved.
"""
Loader for the FP-Arena native module (``_fpcore``): the nanobind bindings
with the Python-side ``fp<Exp, Prec>`` value classes
(``fp_arena.native.fp23_46``).

The library lives in ``runtime/lib``. Its file name embeds a hash of the
sources, so source edits rebuild automatically; each build writes to a temp
file that is atomically renamed, so concurrent builds cannot collide. Built
lazily on first attribute access, not at ``import fp_arena``. Compiler:
``$CXX`` (default ``c++``), honoring ``$CPPFLAGS``/``$CXXFLAGS``/``$LDFLAGS``.
"""

import hashlib
import importlib.machinery
import importlib.util
import os
import shlex
import subprocess
import sysconfig

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))

#: Bindings sources and the fixed library directory inside the repo.
BINDINGS_DIR = os.path.join(_PKG_DIR, "runtime", "bindings")
LIB_DIR = os.path.join(_PKG_DIR, "runtime", "lib")

#: Everything the built library depends on; hashed into its file name.
_SOURCES = (
    os.path.join(BINDINGS_DIR, "fpcore.cpp"),
    os.path.join(BINDINGS_DIR, "instantiations.h"),
    os.path.join(_PKG_DIR, "runtime", "include", "fp_arena", "fp.h"),
)


def _source_hash() -> str:
    h = hashlib.sha256()
    for src in _SOURCES:
        with open(src, "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:12]


def lib_path() -> str:
    """:returns: the source- and Python-version-specific library path."""
    suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
    return os.path.join(LIB_DIR, f"_fpcore.{_source_hash()}{suffix}")


def build() -> str:
    """
    Compile the native module (nanobind's non-CMake build: the module source
    plus ``nb_combined.cpp``, linked against MPFR) into :func:`lib_path`.

    :returns: the path of the built library.
    :raises RuntimeError: if the compiler fails.
    """
    import nanobind

    nb_dir = os.path.dirname(os.path.abspath(nanobind.__file__))
    out = lib_path()
    os.makedirs(LIB_DIR, exist_ok=True)
    tmp = f"{out}.tmp{os.getpid()}"
    cmd = [
        os.environ.get("CXX", "c++"),
        "-std=c++20",
        "-O2",
        "-shared",
        "-fPIC",
        "-fvisibility=hidden",
        "-DNDEBUG",
        "-DNB_COMPACT_ASSERTIONS",
        *shlex.split(os.environ.get("CPPFLAGS", "")),
        *shlex.split(os.environ.get("CXXFLAGS", "")),
        "-I",
        os.path.join(nb_dir, "include"),
        "-I",
        os.path.join(nb_dir, "ext", "robin_map", "include"),
        "-I",
        sysconfig.get_paths()["include"],
        "-I",
        os.path.join(_PKG_DIR, "runtime", "include"),
        os.path.join(BINDINGS_DIR, "fpcore.cpp"),
        os.path.join(nb_dir, "src", "nb_combined.cpp"),
        *shlex.split(os.environ.get("LDFLAGS", "")),
        "-lmpfr",
        "-o",
        tmp,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                "Failed to build the FP-Arena native module.\n"
                f"Command: {' '.join(cmd)}\n{proc.stderr}"
            )
        os.replace(tmp, out)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    # Best-effort cleanup of libraries built from older sources.
    prefix, current = "_fpcore.", os.path.basename(out)
    for name in os.listdir(LIB_DIR):
        if name.startswith(prefix) and name != current:
            try:
                os.remove(os.path.join(LIB_DIR, name))
            except OSError:
                pass
    return out


#: Loaded extension module / first build failure (both memoized).
_core = None
_build_error = None


def ensure_native():
    """
    Load the native module, compiling it first if the library for the current
    sources is missing. Memoizes both success and failure.

    :returns: the loaded ``_fpcore`` extension module.
    """
    global _core, _build_error
    if _core is not None:
        return _core
    if _build_error is not None:
        raise RuntimeError(
            "The FP-Arena native module failed to build earlier in this "
            "process; fix the toolchain and restart."
        ) from _build_error
    try:
        path = lib_path()
        if not os.path.exists(path):
            build()
        spec = importlib.util.spec_from_file_location("fp_arena._fpcore", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as exc:
        _build_error = exc
        raise
    _core = module
    return module


def __getattr__(name):
    # PEP 562: forward to the lazily built extension. Dunder probes (import
    # machinery, introspection) must not trigger a build.
    if name.startswith("__") and name.endswith("__"):
        raise AttributeError(name)
    return getattr(ensure_native(), name)
