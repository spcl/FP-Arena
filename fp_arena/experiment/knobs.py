# Copyright 2019-2026 ETH Zurich and the FP-Arena authors. All rights reserved.
"""
Knobs: the decision variables of the selection search.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

import dace

from fp_arena.experiment import registry
from fp_arena.experiment.config import CONSTANTS_KEY, ExperimentConfig, PrecisionMap
from fp_arena.experiment.retarget import apply_precision, fresh_sdfg
from fp_arena.transformations.change_and_propagate_fp_types import (
    _class_types,
    _collect_sdfgs,
    _is_fp,
    _unify_nested_boundaries,
)

#: Bit width of each precisio, low precision to high; the cap and ordering.
_BITS = {"fp16": 16, "fp32": 32, "fp64": 64}

#: The default precision ladder, low precision to high.
DEFAULT_LADDER = ("fp16", "fp32", "fp64")


@dataclass(frozen=True)
class Knob:
    """
    One decision variable of the search.

    :param name: the top-level pin name (an array name, or :data:`CONSTANTS_KEY`).
    :param kind: ``"source"`` (a read input -- what screening probes; may also be
        written, as in an in-place kernel), ``"override"`` (an output or
        intermediate, whose type otherwise follows propagation), or
        ``"constants"``.
    :param original: precision key of the class's original dtype.
    :param domain: the precision this knob may take, low precision to high, capped at ``original``
    """

    name: str
    kind: str
    original: str
    domain: tuple[str, ...]

    @property
    def highest(self) -> str:
        """The highest precision this knob may take"""
        return self.domain[-1]


@dataclass
class KnobSelection:
    """
    Which knobs are in scope for the search.
    """

    sources: bool = True
    constants: bool = False
    overrides: bool = False
    include: frozenset[str] = field(default_factory=frozenset)
    exclude: frozenset[str] = field(default_factory=frozenset)

    def wants(self, knob: Knob) -> bool:
        if knob.name in self.exclude:
            return False
        if knob.name in self.include:
            return True
        toggle = {
            "source": self.sources,
            "override": self.overrides,
            "constants": self.constants,
        }
        return toggle[knob.kind]


def _has_float_literals(sdfgs: list[dace.SDFG]) -> bool:
    """Whether any Python tasklet body contains a float literal (the constants knob)."""
    for sd in sdfgs:
        for state in sd.all_states():
            for node in state.nodes():
                if not (
                    isinstance(node, dace.nodes.Tasklet)
                    and node.language == dace.Language.Python
                ):
                    continue
                try:
                    tree = ast.parse(node.code.as_string)
                except (SyntaxError, ValueError):
                    continue
                for sub in ast.walk(tree):
                    if isinstance(sub, ast.Constant) and isinstance(sub.value, float):
                        return True
    return False


def _domain(original_key: str, ladder: tuple[str, ...]) -> tuple[str, ...]:
    """The precision at or below ``original_key``; empty if it is off-ladder."""
    if original_key not in _BITS:
        return (original_key,)
    return tuple(f for f in ladder if _BITS[f] <= _BITS[original_key])


def find_knobs(
    sdfg: dace.SDFG,
    selection: KnobSelection | None = None,
    ladder: tuple[str, ...] = DEFAULT_LADDER,
) -> list[Knob]:
    """
    Enumerate the search's knobs from ``sdfg``'s propagation structures.

    :param sdfg: the program under test.
    :param selection: restrict to a subset; ``None`` keeps every knob.
    :param ladder: the precision ladder, low precision to high.
    :returns: the knobs, sorted by name.
    """
    sdfgs = _collect_sdfgs(sdfg)
    uf = _unify_nested_boundaries(sdfgs)
    members, original = _class_types(sdfgs, uf)
    reads, _ = sdfg.read_and_write_sets()

    knobs: list[Knob] = []
    for rep, mem in members.items():
        if not _is_fp(original[rep]):
            continue
        top = sorted(name for sd, name in mem if sd is sdfg)
        if not top:  # nested-only class: not pinnable
            continue
        name = top[0]
        try:
            orig_key = registry.key_of(original[rep])
        except ValueError:
            continue
        domain = _domain(orig_key, ladder)
        if len(domain) < 2:  # no lower precision available: decides nothing
            continue
        is_input = name in reads and not sdfg.arrays[name].transient
        knobs.append(
            Knob(
                name=name,
                kind="source" if is_input else "override",
                original=orig_key,
                domain=domain,
            )
        )

    if _has_float_literals(sdfgs):
        knobs.append(
            Knob(
                name=CONSTANTS_KEY,
                kind="constants",
                original="fp64",
                domain=tuple(ladder),
            )
        )

    knobs.sort(key=lambda k: k.name)
    if selection is not None:
        knobs = [k for k in knobs if selection.wants(k)]
    return knobs


# A candidate's identity: its inferred per-array typing plus the constants precision.
# Two pin-maps with the same key compile to the same program.
CanonicalKey = tuple[tuple[tuple[str, str], ...], str | None]


def canonical_typing(
    experiment: ExperimentConfig, pin_map: PrecisionMap
) -> CanonicalKey:
    """
    The canonical key of ``pin_map``: the inferred typing it produces.
    """
    sdfg = fresh_sdfg(experiment)
    apply_precision(sdfg, pin_map, experiment.promotion_rules)
    typing: list[tuple[str, str]] = []
    for name, desc in sdfg.arrays.items():
        if not isinstance(desc, dace.data.Array):
            continue
        try:
            typing.append((name, registry.key_of(desc.dtype)))
        except ValueError:
            continue  # non-fp array
    typing.sort()
    return (tuple(typing), pin_map.get(CONSTANTS_KEY))
