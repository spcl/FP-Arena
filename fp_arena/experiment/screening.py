# Copyright 2019-2026 ETH Zurich and the FP-Arena authors. All rights reserved.
"""
Screening: For each source (input) knob, perturb *that input only* on the baseline SDFG
with relative noise at fp16 and fp32 magnitude and measure the induced output
error.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy import stats

from fp_arena.experiment.config import (
    ErrorBudget,
    ExperimentConfig,
    PerturbationAnalysisConfig,
)
from fp_arena.experiment.inputs import Noise
from fp_arena.experiment.knobs import Knob
from fp_arena.experiment.results import METRICS, ErrorStats
from fp_arena.experiment.runner import _output_arrays, run_perturbation
from fp_arena.experiment.selection import check, noise_floor, resolve_limits
from fp_arena.experiment.store import ResultStore

#: Relative rounding magnitude of an fp16 / fp32 mantissa (its ulp).
FP16_ULP = 2.0**-11
FP32_ULP = 2.0**-24

#: Zero-mean unit noise: uniform on [-1, 1]. Scaled by the ulp per probe.
_UNIT_NOISE = stats.uniform(-1.0, 2.0)


@dataclass
class Screening:
    """
    :param limits: the resolved budget thresholds, ``{metric: {array: limit}}``.
    :param safe_format: ``{input: lowest format whose probe stays within budget}``;
        inputs with no safe precision are absent.
    :param sensitivity: ``{input: worst fraction of any limit}``;
        ``>= 1`` means the probe is already over budget.
    """

    limits: dict[str, dict[str, float]]
    safe_format: dict[str, str]
    sensitivity: dict[str, float]


def _severity(
    errors: dict[str, ErrorStats], limits: dict[str, dict[str, float]]
) -> float:
    """The worst fraction of any (lower-is-better, finite) limit the probe consumes."""
    worst = 0.0
    for c in check(errors, limits):
        if METRICS[c.metric].higher_is_better:
            continue
        if c.limit > 0.0 and math.isfinite(c.limit) and math.isfinite(c.value):
            worst = max(worst, c.value / c.limit)
    return worst


def screen(
    experiment: ExperimentConfig,
    budget: ErrorBudget,
    noise: dict[str, Noise],
    source_knobs: list[Knob],
    n_samples: int,
    store: ResultStore | None = None,
) -> Screening:
    """
    Resolve the budget thresholds and probe every source knob at fp16 and fp32.
    """
    metrics: list[str] = sorted({*budget.limits, *(budget.from_perturbation or {})})
    floor: dict[str, dict[str, float]] = {}
    if budget.from_perturbation:
        base = run_perturbation(
            PerturbationAnalysisConfig(
                experiment, noise=noise, precisions=[{}], n_samples=n_samples
            ),
            store=store,
        )
        floor = noise_floor(base, metrics)
    arrays = budget.arrays or _output_arrays(experiment.program)
    limits = resolve_limits(budget, floor, arrays)

    probes: dict[str, dict[str, dict[str, ErrorStats]]] = {}
    for fmt, ulp in (("fp16", FP16_ULP), ("fp32", FP32_ULP)):
        wanted = [k.name for k in source_knobs if fmt in k.domain[:-1]]
        if not wanted:
            probes[fmt] = {}
            continue
        noise_map = {
            name: Noise(relative=ulp, relative_dist=_UNIT_NOISE) for name in wanted
        }
        results = run_perturbation(
            PerturbationAnalysisConfig(
                experiment, noise=noise_map, precisions=[{}], n_samples=n_samples
            ),
            store=store,
        )
        probes[fmt] = {r.perturbed: r.errors for r in results}

    safe_format: dict[str, str] = {}
    sensitivity: dict[str, float] = {}
    for knob in source_knobs:
        name = knob.name
        for fmt in knob.domain[:-1]:  # low precision to high, skipping the original
            errors = probes.get(fmt, {}).get(name)
            if errors is not None and all(c.ok for c in check(errors, limits)):
                safe_format[name] = fmt
                break
        errors16 = probes.get("fp16", {}).get(name)
        sensitivity[name] = _severity(errors16, limits) if errors16 is not None else 1.0

    return Screening(limits=limits, safe_format=safe_format, sensitivity=sensitivity)
