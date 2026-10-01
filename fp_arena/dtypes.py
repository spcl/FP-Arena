# Copyright 2024-2026 ETH Zurich and the FP-Arena authors. All rights reserved.
"""
FP-Arena floating-point data types and their registration into DaCe.

The types here subclass :class:`dace.dtypes.typeclass` so they behave like any
built-in DaCe scalar type (``dace.float32`` etc.), but they generate code that
uses the FP-Arena C++ value types (see ``runtime/include/fp_arena``). They are
distinguished from the built-in types by their C type string, so existing
``float32`` / ``float64`` arrays are never silently affected.

Registration follows the same pattern as ``dace-fpga``: importing this module
mutates DaCe's global registries at runtime, so no DaCe source file is changed.
"""

import ctypes

import dace
import numpy
from dace import dtypes as _ddtypes
from dace import typeclass

#: C++ namespace-qualified type names emitted into generated code.
_FLOAT32SR_CTYPE = "fp_arena::float32sr"
_FLOAT64SR_CTYPE = "fp_arena::float64sr"


class Float32sr(_ddtypes.typeclass):
    """
    32-bit stochastically-rounded float. Storage and ABI match ``float`` /
    ``numpy.float32``; arithmetic in generated code rounds stochastically.
    """

    def __init__(self):
        super().__init__(numpy.float32, typename="float32sr")
        self.ctype = _FLOAT32SR_CTYPE
        self.ctype_unaligned = _FLOAT32SR_CTYPE

    def to_json(self):
        return "float32sr"

    @staticmethod
    def from_json(json_obj, context=None) -> "Float32sr":
        return float32sr

    def __repr__(self) -> str:
        return "fp_arena.float32sr"


class Float64sr(_ddtypes.typeclass):
    """
    64-bit stochastically-rounded float. Storage and ABI match ``double`` /
    ``numpy.float64``; arithmetic in generated code rounds stochastically.
    """

    def __init__(self):
        super().__init__(numpy.float64, typename="float64sr")
        self.ctype = _FLOAT64SR_CTYPE
        self.ctype_unaligned = _FLOAT64SR_CTYPE

    def to_json(self):
        return "float64sr"

    @staticmethod
    def from_json(json_obj, context=None) -> "Float64sr":
        return float64sr

    def __repr__(self) -> str:
        return "fp_arena.float64sr"


#: Singleton instances used everywhere (mirrors ``dace.float32`` being an instance).
float32sr = Float32sr()
float64sr = Float64sr()

#: All FP-Arena typeclasses, keyed by their serialization string.
FP_ARENA_TYPECLASSES = {
    "float32sr": float32sr,
    "float64sr": float64sr,
}


class _mpfr_t(ctypes.Structure):
    _fields_ = [
        ("_mpfr_prec", ctypes.c_long),
        ("_mpfr_sign", ctypes.c_int),
        ("_mpfr_exp", ctypes.c_long),
        ("_mpfr_d", ctypes.c_void_p),
    ]


class mpfr(typeclass):
    """
    A data type for custom Multiple Precision Floating-Point (MPFR) types.

    Example use: `dace.mpfr(128)` for 128-bit precision.
    """

    def __init__(self, precision: int):
        self.precision = precision
        self.type = numpy.object_
        self.bytes = ctypes.sizeof(_mpfr_t)
        self.dtype = self
        self.typename = f"mpfr{precision}"
        # Expose this concrete precision as ``dace.<typename>``
        setattr(_ddtypes, self.typename, self)
        setattr(dace, self.typename, self)

    def to_string(self):
        return self.typename

    def to_json(self):
        return {"type": "mpfr", "precision": self.precision}

    @staticmethod
    def from_json(json_obj, context=None):
        if json_obj["type"] != "mpfr":
            raise TypeError("Invalid type for mpfr")
        return mpfr(json_obj["precision"])

    @property
    def ctype(self):
        return f"dace::mpfr<{self.precision}>"

    @property
    def ctype_unaligned(self):
        return self.ctype

    def as_ctypes(self):
        return ctypes.c_void_p

    def as_numpy_dtype(self):
        return numpy.dtype(numpy.object_)

    @property
    def base_type(self):
        return self


def register():
    """
    Register the FP-Arena types into DaCe's global registries.

    Idempotent and non-invasive: it only mutates the in-memory registries that
    DaCe exposes for extension (the same ones ``@dace.serialize.serializable``
    writes to), so it can be called repeatedly and changes no DaCe source.

    :returns: the mapping of registered typeclasses.
    """
    import dace.serialize as _serialize

    for name, tc in FP_ARENA_TYPECLASSES.items():
        # Round-trip path: json_to_typeclass(name) -> get_serializer(name) -> tc.
        _serialize._DACE_SERIALIZE_TYPES.setdefault(name, tc)
        # Expose as dace.<name> / dace.dtypes.<name> so they extend dace.dtypes.
        setattr(_ddtypes, name, tc)
        setattr(dace, name, tc)
        # Keep the string/lookup tables consistent with the built-in types.
        if name not in _ddtypes.TYPECLASS_STRINGS:
            _ddtypes.TYPECLASS_STRINGS.append(name)
        _ddtypes.TYPECLASS_TO_STRING.setdefault(tc, tc.ctype)

    # Also expose the parametric mpfr class so `dace.mpfr(128)` works.
    _ddtypes.mpfr = mpfr
    dace.mpfr = mpfr

    _teach_dace_mpfr()

    return FP_ARENA_TYPECLASSES


def _teach_dace_mpfr() -> None:
    """
    Extend the two DaCe helpers that assume every float has a numpy scalar type.

    ``mpfr`` maps to ``numpy.object_``, whose instances are plain Python ints:
    DaCe's type inference (``result_type_of``, used for interstate edge
    expressions) and its ``Fill`` literals (``python_literal``) both call numpy
    scalar methods on them and fail. Both are wrapped so an ``mpfr`` operand is
    handled first and everything else keeps DaCe's own behaviour. Idempotent.
    """
    if getattr(_ddtypes.result_type_of, "_fp_arena_mpfr", False):
        return

    original_result_type_of = _ddtypes.result_type_of

    def result_type_of(lhs, *rhs):
        if len(rhs) > 1:
            return result_type_of(result_type_of(lhs, rhs[0]), *rhs[1:])
        other = rhs[0] if rhs else None
        if isinstance(lhs, mpfr) or isinstance(other, mpfr):
            # Same rule as the precision pass: mpfr outranks any other float,
            # the higher precision wins.
            if isinstance(lhs, mpfr) and isinstance(other, mpfr):
                return lhs if lhs.precision >= other.precision else other
            return lhs if isinstance(lhs, mpfr) else other
        return original_result_type_of(lhs, *rhs)

    result_type_of._fp_arena_mpfr = True
    _ddtypes.result_type_of = result_type_of

    from dace.libraries.standard.nodes.fill import common as fill_common

    original_numpy_scalar = fill_common.numpy_scalar

    def numpy_scalar(value, dtype):
        if isinstance(dtype, mpfr):
            # The fill value as a double: every Fill literal renders from this,
            # and dace::mpfr assigns from double. Its 8 bytes never match
            # sizeof(mpfr_t), so a fill of an mpfr array is never a memset
            # (which would clobber the limb pointer inside mpfr_t).
            return numpy.float64(value)
        return original_numpy_scalar(value, dtype)

    fill_common.numpy_scalar = numpy_scalar
