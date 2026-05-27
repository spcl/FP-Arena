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

import dace
from dace import dtypes as _ddtypes

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

    return FP_ARENA_TYPECLASSES
