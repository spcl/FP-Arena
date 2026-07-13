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

import numpy
import ctypes

import dace
from dace import dtypes as _ddtypes, typeclass

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


class fp(typeclass):
    """
    An emulated floating-point type with a custom exponent width and precision
    (significand bits including the hidden bit, as in MPFR) ``fp_arena::fp<Exp, Prec>``.

    Example use: ``dace.fp(23, 46)`` for 23 exponent bits and 46 bits of
    precision (typename ``fp23_46``, 9 bytes per element).
    """

    _interned: dict = {}

    def __new__(cls, exp_bits: int, precision: int):
        inst = cls._interned.get((exp_bits, precision))
        if inst is None:
            inst = super().__new__(cls)
        return inst

    # Immutable and interned: copies are the instance itself, pickling goes
    # through the constructor.
    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    def __reduce__(self):
        return (fp, (self.exp_bits, self.precision))

    def __init__(self, exp_bits: int, precision: int):
        if self._interned.get((exp_bits, precision)) is self:
            return  # already initialized (interned instance)
        if not 2 <= exp_bits <= 30:
            raise ValueError(f"exp_bits must be in [2, 30], got {exp_bits}")
        if not 2 <= precision <= 64:
            raise ValueError(
                f"precision must be in [2, 64], got {precision}; use dace.mpfr for more"
            )
        self._interned[(exp_bits, precision)] = self
        self.exp_bits = exp_bits
        self.precision = precision
        self.type = numpy.void
        self.bytes = (exp_bits + precision + 7) // 8
        self.dtype = self
        self.typename = f"fp{exp_bits}_{precision}"
        # Expose this concrete format as ``dace.<typename>``
        setattr(_ddtypes, self.typename, self)
        setattr(dace, self.typename, self)

    def to_string(self):
        return self.typename

    def to_json(self):
        return {"type": "fp", "exp_bits": self.exp_bits, "precision": self.precision}

    @staticmethod
    def from_json(json_obj, context=None):
        if json_obj["type"] != "fp":
            raise TypeError("Invalid type for fp")
        return fp(json_obj["exp_bits"], json_obj["precision"])

    @property
    def ctype(self):
        return f"fp_arena::fp<{self.exp_bits}, {self.precision}>"

    @property
    def ctype_unaligned(self):
        return self.ctype

    def as_ctypes(self):
        return ctypes.c_byte * self.bytes

    def as_numpy_dtype(self):
        return numpy.dtype((numpy.void, self.bytes))

    @property
    def base_type(self):
        return self

    def __repr__(self) -> str:
        return f"fp_arena.fp({self.exp_bits}, {self.precision})"


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

    # Expose the parametric classes as dace.mpfr / dace.fp and register their
    # serializers so `{"type": "mpfr"/"fp", ...}` json deserializes.
    setattr(_ddtypes, "mpfr", mpfr)
    setattr(dace, "mpfr", mpfr)
    setattr(_ddtypes, "fp", fp)
    setattr(dace, "fp", fp)
    _serialize._DACE_SERIALIZE_TYPES.setdefault("mpfr", mpfr)
    _serialize._DACE_SERIALIZE_TYPES.setdefault("fp", fp)

    return FP_ARENA_TYPECLASSES
