# Copyright 2019-2026 ETH Zurich and the FP-Arena authors. All rights reserved.
"""
Result records returned by the analysis drivers, one per precision point. Each
serialises to a plain dict (``to_dict``) for the results database.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List

from fp_arena.experiment.config import PrecisionMap


@dataclass
class ErrorStats:
    """Error of one array vs the reference, reduced over elements and samples."""

    abs_mean: float
    abs_max: float
    rel_mean: float
    rel_max: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class PerfResult:
    """Timing for one precision point (seconds)."""

    precision: PrecisionMap
    times: List[float]
    time_median: float
    time_std: float
    time_min: float
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
        return {
            "precision": self.precision,
            "errors": {k: v.to_dict() for k, v in self.errors.items()},
            "n_samples": self.n_samples,
            "seed": self.seed,
        }


# TODO(select): Add SelectResult
