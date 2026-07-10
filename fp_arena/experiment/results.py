# Copyright 2019-2026 ETH Zurich and the FP-Arena authors. All rights reserved.
"""
Result records returned by the analysis drivers, one per precision point. Each
serialises to a plain dict (``to_dict``) for the results database.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

from fp_arena.experiment.config import PrecisionMap


@dataclass
class ErrorStats:
    """
    Error metrics of an array versus a reference, reduced over elements and samples.

    Calculations use the error vector `e` (array - reference) and the reference vector `r`.

    Metrics
    -------
    abs_mean  : Mean(|e|)

    rel_mean  : Mean(|e| / |r|)
    rel_max   : Max(|e| / |r|)

    l1        : sum(|e|)
    l2        : sqrt(sum(e**2))
    linf      : max(|e|)

    l1_norm   : sum(|e|) / sum(|r|)
    l2_norm   : sqrt(sum(e**2)) / sqrt(sum(r**2))
    linf_norm : max(|e|) / max(|r|)
    snr       : 10 * log10(sum(r**2) / sum(e**2))
    """

    abs_mean: float
    rel_mean: float
    rel_max: float
    l1: float
    l2: float
    linf: float
    l1_norm: float
    l2_norm: float
    linf_norm: float
    snr: float


@dataclass
class PerfResult:
    """
    Per-repetition phase timings (milliseconds) for one precision point, with
    ``total = h2d + cast_in + kernel + cast_out + d2h``. Only raw lists are kept;
    derive medians/etc. from them.
    """

    precision: PrecisionMap
    total_times: List[float] = field(default_factory=list)
    h2d_times: List[float] = field(
        default_factory=list
    )  # host->device transfer (GPU only)
    d2h_times: List[float] = field(
        default_factory=list
    )  # device->host transfer (GPU only)
    cast_in_times: List[float] = field(default_factory=list)  # input precision cast
    cast_out_times: List[float] = field(default_factory=list)  # output precision cast
    kernel_times: List[float] = field(default_factory=list)  # compute
    seed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ErrorResult:
    """Per-array error for one precision point."""

    precision: PrecisionMap
    errors: Dict[str, ErrorStats]
    n_samples: int
    seed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PerturbationResult:
    """
    Per-array output deviation induced by perturbing the single input
    ``perturbed``, versus the clean run at the same precision point.
    """

    precision: PrecisionMap
    perturbed: str
    errors: Dict[str, ErrorStats]
    n_samples: int
    seed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
