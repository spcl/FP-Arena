"""Tests for the FP-Arena native module loader (``fp_arena.native``)."""

import importlib.machinery
import os

from fp_arena import native

SUFFIX = importlib.machinery.EXTENSION_SUFFIXES[0]


def test_stale_lib_selection():
    """Only built libraries from older sources are swept, never temp files."""
    current = f"_fpcore.abc123def456{SUFFIX}"
    stale = f"_fpcore.0123456789ab{SUFFIX}"

    assert native._is_stale_lib(stale, current, SUFFIX)
    assert not native._is_stale_lib(current, current, SUFFIX)

    # A concurrent build writes "<lib>.tmp<pid>" and renames it into place.
    # Sweeping that would break the build that is still writing it, so the
    # extension suffix -- not just the "_fpcore." prefix -- decides.
    for pid in (1, 99999):
        assert not native._is_stale_lib(f"{stale}.tmp{pid}", current, SUFFIX)
        assert not native._is_stale_lib(f"{current}.tmp{pid}", current, SUFFIX)

    # Unrelated files in the directory are left alone.
    assert not native._is_stale_lib("README.md", current, SUFFIX)
    assert not native._is_stale_lib("_other.so", current, SUFFIX)


def test_tmp_name_shape_matches_the_filter():
    """The temp name build() actually uses must survive the stale filter."""
    out = native.lib_path()
    tmp = f"{out}.tmp{os.getpid()}"  # exactly what build() constructs
    current = os.path.basename(out)
    assert not native._is_stale_lib(os.path.basename(tmp), current, SUFFIX)


def test_lib_path_is_source_specific():
    """The library name embeds a hash of the sources, under LIB_DIR."""
    path = native.lib_path()
    assert os.path.dirname(path) == native.LIB_DIR
    assert os.path.basename(path).startswith("_fpcore.")
    assert path.endswith(SUFFIX)
    assert native.lib_path() == path  # deterministic for unchanged sources


if __name__ == "__main__":
    test_stale_lib_selection()
    test_tmp_name_shape_matches_the_filter()
    test_lib_path_is_source_specific()
    print("All native loader tests passed.")
