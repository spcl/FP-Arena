# Copyright 2019-2026 ETH Zurich and the FP-Arena authors. All rights reserved.
"""
Build the CloudSC graph the CloudSC experiments run on, from ``corpus/cloudsc/cloudsc.py``.
"""

from __future__ import annotations

import dace
from dace.transformation.passes.canonicalize import canonicalize

from corpus.cloudsc.cloudsc import cloudsc_py
from corpus.cloudsc.cloudsc_reference_data import CLOUDSC_SPECIES, _constants

#: The run-time flags baked in at the reference values, beside the species.
FLAGS = ("yrecldp_nssopt", "yrecldp_laericesed")


def build_cloudsc_sdfg() -> dace.SDFG:
    """The canonicalized CloudSC SDFG, with ``klon`` and ``klev`` left symbolic."""
    constants = _constants()
    sdfg = cloudsc_py.to_sdfg(simplify=False)
    canonicalize(
        sdfg,
        validate=True,
        target="gpu",
        specialize_constants={
            **CLOUDSC_SPECIES,
            **{name: int(constants[name]) for name in FLAGS},
        },
    )
    sdfg.validate()
    return sdfg
