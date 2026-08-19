# Copyright 2019-2026 ETH Zurich and the FP-Arena authors. All rights reserved.
"""
Selection search

Flow:

0. Enumerate and select the knobs; compile the reference once.
1. Perturb each input at fp16/fp32 magnitude on the baseline SDFG.
2. Seed a per-input candidate for each safe input (based on perturbation) plus one combined candidate.
3. Prioritise by benefit (big, low-sensitivity arrays lowered first) defined by the scoring function.
4. Evaluate: error-check each candidate; infeasible prunes its down-set,
   feasible is timed and expands its one-step-lower neighbours.
5. Return the fastest measured feasible config.
"""

from __future__ import annotations

import concurrent.futures as cf
import heapq
import itertools
import multiprocessing as mp
import os
from dataclasses import dataclass, field

import dace
import numpy as np

from fp_arena.experiment.config import (
    OBJECTIVES,
    ErrorBudget,
    ExperimentConfig,
    PrecisionMap,
)
from fp_arena.experiment.inputs import Noise, make_call_args, materialize_shape
from fp_arena.experiment.knobs import (
    _BITS,
    DEFAULT_LADDER,
    CanonicalKey,
    KnobSelection,
    canonical_typing,
    find_knobs,
)
from fp_arena.experiment.results import (
    ErrorStats,
    PerfResult,
    SearchCandidate,
    SearchResult,
)
from fp_arena.experiment.retarget import resolve_vectorize_config
from fp_arena.experiment.runner import (
    _accumulate,
    _copy_args,
    _finalize,
    _new_acc,
    _output_arrays,
    _sample_rngs,
    build_candidate_sdfg,
    compile_reference,
    measure,
)
from fp_arena.experiment.screening import screen
from fp_arena.experiment.selection import _objective_ms, check
from fp_arena.experiment.store import ResultStore


@dataclass
class SelectionSearchConfig:
    """
    Search for the fastest per-knob precision assignment meeting ``budget``.

    :param experiment: the program under test.
    :param budget: the accuracy a config must deliver to be feasible.
    :param noise: inputs to perturb for the budget's noise floor (only needed
        when ``budget.from_perturbation`` is set).
    :param knobs: which knobs are in scope (default sources + constants).
    :param reference: the error analysis' high-precision reference.
    :param ladder: the precision ladder, low precision to high.
    :param n_samples: input realisations for error and screening.
    :param n_warmup: untimed warmup invocations per timed config.
    :param n_reps: timed repetitions per timed config.
    :param objective: the timing phase to minimise, one of :data:`OBJECTIVES`.
    :param compute_budget: max distinct programs to compile; ``None`` runs to
        exhaustion.
    :param n_workers: number of unpinned worker processes when ``devices`` is
        ``None`` (CPU / testing); ignored when ``devices`` is given.
    :param devices: GPU ids to pin workers to via ``CUDA_VISIBLE_DEVICES`` -- one
        worker per id, each owning its device exclusively.
    :param name: database/report name; defaults to the experiment's.

    """

    experiment: ExperimentConfig
    budget: ErrorBudget
    noise: dict[str, Noise] = field(default_factory=dict)
    knobs: KnobSelection = field(default_factory=KnobSelection)
    reference: str | dict[str, str] = "fp64"
    ladder: tuple[str, ...] = DEFAULT_LADDER
    n_samples: int = 1
    n_warmup: int = 1
    n_reps: int = 10
    objective: str = "total"
    compute_budget: int | None = None
    n_workers: int = 1
    devices: list[int] | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        if self.objective not in OBJECTIVES:
            raise ValueError(
                f"Unknown objective {self.objective!r}; expected one of {list(OBJECTIVES)}"
            )


#: A config: every in-scope knob mapped to a ladder rung.
Config = dict[str, str]


class _Slot:
    """
    One evaluation slot, owning a single device (or a CPU slot).
    """

    def __init__(
        self,
        experiment: ExperimentConfig,
        reference: str | dict[str, str],
        n_samples: int,
        limits: dict[str, dict[str, float]],
        n_warmup: int,
        n_reps: int,
    ) -> None:
        self.experiment = experiment
        self.outputs = _output_arrays(experiment.program)
        self.limits = limits
        self.n_warmup = n_warmup
        self.n_reps = n_reps
        reads, _ = experiment.program.read_and_write_sets()
        ref = compile_reference(experiment, reference)
        self._samples: list[tuple[dict, dict[str, np.ndarray]]] = []
        for rng in _sample_rngs(experiment.seed, n_samples):
            args = make_call_args(experiment.program, experiment, rng, reads=reads)
            ref_args = _copy_args(args)
            ref(**ref_args)
            self._samples.append((args, {n: ref_args[n] for n in self.outputs}))
        ref.finalize()

    def evaluate(
        self, pin_map: PrecisionMap
    ) -> tuple[dict[str, ErrorStats], PerfResult | None]:
        """
        Grade ``pin_map`` against the reference and, when it meets the numeric limits, time it.
        """
        sdfg = build_candidate_sdfg(self.experiment, pin_map, instrument=True)
        csdfg = sdfg.compile()
        accs = {name: _new_acc() for name in self.outputs}
        for args, ref_out in self._samples:
            cand_args = _copy_args(args)
            csdfg(**cand_args)
            for name in self.outputs:
                _accumulate(accs[name], ref_out[name], cand_args[name])
        errors = {name: _finalize(a) for name, a in accs.items()}
        perf: PerfResult | None = None
        if all(c.ok for c in check(errors, self.limits)):
            rng = _sample_rngs(self.experiment.seed, 1)[0]
            # The error samples already ran on this compiled object; the timing
            # report accumulates them, so measure() drops all but the last reps.
            perf = PerfResult(
                precision=dict(pin_map),
                seed=self.experiment.seed,
                **measure(
                    sdfg,
                    csdfg,
                    self.experiment,
                    self.n_warmup,
                    self.n_reps,
                    rng,
                    prior_invocations=len(self._samples),
                ),
            )
        else:
            csdfg.finalize()
        return errors, perf


_WORKER: dict = {}


def _init_slot(
    experiment: ExperimentConfig,
    reference: str | dict[str, str],
    n_samples: int,
    limits: dict[str, dict[str, float]],
    n_warmup: int,
    n_reps: int,
    device_queue,
) -> None:
    # Pin the device *before* anything creates a CUDA context.
    # Each worker pops one id, so devices are distinct.
    device = device_queue.get()
    if device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    dace.Config.set("instrumentation", "report_each_invocation", value=False)
    if experiment.target == "gpu":
        dace.Config.set("compiler", "cuda", "max_concurrent_streams", value=-1)
    _WORKER["slot"] = _Slot(experiment, reference, n_samples, limits, n_warmup, n_reps)


def _eval_task(
    pin_map: PrecisionMap,
) -> tuple[dict[str, ErrorStats], PerfResult | None]:
    return _WORKER["slot"].evaluate(pin_map)


def _array_sizes(experiment: ExperimentConfig, names: list[str]) -> dict[str, int]:
    """Element count of each knob's array (1 for constants / unknown)."""
    sizes: dict[str, int] = {}
    for name in names:
        desc = experiment.program.arrays.get(name)
        if not isinstance(desc, dace.data.Array):  # constants / scalars: no shape
            sizes[name] = 1
            continue
        try:
            shape = materialize_shape(desc.shape, experiment.symbols)
            sizes[name] = max(1, int(np.prod(shape)))
        except (ValueError, TypeError):
            sizes[name] = 1
    return sizes


def _feasible(
    errors: dict[str, ErrorStats],
    limits: dict[str, dict[str, float]],
    budget: ErrorBudget,
) -> bool:
    """Whether ``errors`` meets every numeric constraint and the predicate."""
    if not all(c.ok for c in check(errors, limits)):
        return False
    if budget.predicate is not None:
        return bool(budget.predicate(errors))
    return True


def run_search(
    cfg: SelectionSearchConfig, store: ResultStore | None = None
) -> SearchResult:
    """
    Run the selection search and return its :class:`SearchResult`.

    Prints :func:`format_search` of the outcome.
    """
    exp = cfg.experiment
    knobs = find_knobs(exp.program, cfg.knobs, cfg.ladder)
    if not knobs:
        raise ValueError(
            "No knobs in scope: widen KnobSelection or check the program has "
            "lowerable floating-point arrays."
        )
    names = [k.name for k in knobs]
    knob_of = {k.name: k for k in knobs}
    rung = {k.name: {fmt: i for i, fmt in enumerate(k.domain)} for k in knobs}
    highest = {k.name: k.highest for k in knobs}
    sizes = _array_sizes(exp, names)

    source_knobs = [k for k in knobs if k.kind == "source"]
    screening = screen(exp, cfg.budget, cfg.noise, source_knobs, cfg.n_samples, store)
    limits = screening.limits

    root: Config = {name: highest[name] for name in names}

    def pin_map(config: Config) -> PrecisionMap:
        return {n: f for n, f in config.items() if f != highest[n]}

    def vec(config: Config) -> tuple[int, ...]:
        return tuple(rung[n][config[n]] for n in names)

    def score(config: Config) -> float:
        total = 0.0
        for name in names:
            saved = _BITS.get(knob_of[name].original, 64) - _BITS.get(config[name], 64)
            weight = 1.0 / (1.0 + screening.sensitivity.get(name, 0.0))
            total += sizes[name] * max(saved, 0) * weight
        return total

    infeasible: list[tuple[int, ...]] = []

    def dominated(config: Config) -> bool:
        v = vec(config)
        return any(all(a <= b for a, b in zip(v, iv, strict=True)) for iv in infeasible)

    def neighbors(config: Config) -> list[Config]:
        out: list[Config] = []
        for name in names:
            i = rung[name][config[name]]
            if i > 0:
                nb = dict(config)
                nb[name] = knob_of[name].domain[i - 1]
                out.append(nb)
        return out

    heap: list[tuple[float, int, Config]] = []
    counter = itertools.count()
    visited: set[tuple[int, ...]] = set()  # traversal dedup, over the config lattice
    stats = {"evaluated": 0, "pruned": 0, "timed": 0}

    def push(config: Config) -> None:
        v = vec(config)
        if v in visited:
            return
        if dominated(config):
            stats["pruned"] += 1
            return
        visited.add(v)
        heapq.heappush(heap, (-score(config), next(counter), config))

    def expand(config: Config) -> None:
        for nb in neighbors(config):
            push(nb)

    ctx = mp.get_context("spawn")
    # One worker per device (each pinned + exclusive); with no devices, n_workers
    # unpinned CPU slots. Compile, error grade, and timing of a config all run on
    # the one slot that owns it.
    devices = cfg.devices if cfg.devices else [None] * max(1, cfg.n_workers)
    n_slots = len(devices)

    errors_cache: dict[
        CanonicalKey, dict[str, ErrorStats]
    ] = {}  # dedup across pin-maps
    perf_cache: dict[CanonicalKey, PerfResult] = {}
    eval_futs: dict[cf.Future, tuple[Config, CanonicalKey]] = {}
    best: list[Config] = [root]
    best_ms: list[float | None] = [None]
    candidates: list[SearchCandidate] = []

    def key_of(config: Config) -> CanonicalKey:
        return canonical_typing(exp, pin_map(config))

    def update_best(config: Config, perf: PerfResult) -> None:
        ms = _objective_ms(perf, cfg.objective)
        candidates.append(
            SearchCandidate(
                pin_map(config), ms, (root_ms / ms) if root_ms and ms else 0.0
            )
        )
        if best_ms[0] is None or ms < best_ms[0]:
            best_ms[0], best[0] = ms, config

    def integrate(
        config: Config,
        key: CanonicalKey,
        errors: dict[str, ErrorStats],
        perf: PerfResult | None,
    ) -> None:
        """Fold one config's result in. ``perf`` is this config's own timing, or
        ``None`` when it was over the numeric budget or this is a dedup hit (its
        timing, if any, already sits in ``perf_cache``)."""
        if _feasible(errors, limits, cfg.budget):
            if key in perf_cache:
                update_best(config, perf_cache[key])  # dedup: reuse the timing
            elif perf is not None:
                perf_cache[key] = perf
                stats["timed"] += 1
                update_best(config, perf)
            expand(config)
        else:
            infeasible.append(vec(config))

    # Each worker pops one device id from the queue at init and pins to it.
    device_queue = ctx.Queue()
    for device in devices:
        device_queue.put(device)
    pool = cf.ProcessPoolExecutor(
        n_slots,
        mp_context=ctx,
        initializer=_init_slot,
        initargs=(
            exp,
            cfg.reference,
            cfg.n_samples,
            limits,
            cfg.n_warmup,
            cfg.n_reps,
            device_queue,
        ),
    )
    root_ms = None
    try:
        # Root first: the feasibility floor and the speedup denominator.
        visited.add(vec(root))
        root_key = key_of(root)
        stats["evaluated"] += 1
        root_errors, root_perf = pool.submit(_eval_task, pin_map(root)).result()
        errors_cache[root_key] = root_errors
        if not _feasible(root_errors, limits, cfg.budget):
            return _finish(
                SearchResult(
                    best=None,
                    best_ms=None,
                    speedup=None,
                    objective=cfg.objective,
                    knobs=knobs,
                    limits=limits,
                    n_evaluated=stats["evaluated"],
                    n_pruned=0,
                    n_timed=0,
                    unsatisfiable=True,
                ),
                cfg,
                exp,
                store,
            )
        # Feasible root cleared the numeric limits, so the worker timed it.
        assert root_perf is not None
        perf_cache[root_key] = root_perf
        stats["timed"] += 1
        root_ms = _objective_ms(root_perf, cfg.objective)
        best_ms[0] = root_ms
        candidates.append(SearchCandidate(pin_map(root), root_ms, 1.0))

        # Seeds: per-input safe candidates and the combined candidate, then neighbours.
        for name, fmt in screening.safe_format.items():
            if name in root:
                cand = dict(root)
                cand[name] = fmt
                push(cand)
        if screening.safe_format:
            combined = dict(root)
            for name, fmt in screening.safe_format.items():
                if name in combined:
                    combined[name] = fmt
            push(combined)
        expand(root)

        budget_hit = False
        while (heap and not budget_hit) or eval_futs:
            # Keep up to n_slots compile+error+timing tasks in flight.
            while heap and not budget_hit and len(eval_futs) < n_slots:
                if (
                    cfg.compute_budget is not None
                    and stats["evaluated"] >= cfg.compute_budget
                ):
                    budget_hit = True
                    break
                _, _, config = heapq.heappop(heap)
                if dominated(config):
                    stats["pruned"] += 1
                    continue
                key = key_of(config)
                if key in errors_cache:
                    integrate(config, key, errors_cache[key], None)
                    continue
                stats["evaluated"] += 1
                eval_futs[pool.submit(_eval_task, pin_map(config))] = (config, key)

            if not eval_futs:
                break
            done, _ = cf.wait(list(eval_futs), return_when=cf.FIRST_COMPLETED)
            for fut in done:
                config, key = eval_futs.pop(fut)
                errors, perf = fut.result()
                errors_cache[key] = errors
                integrate(config, key, errors, perf)
    finally:
        pool.shutdown(wait=True)

    result = SearchResult(
        best=best[0],
        best_ms=best_ms[0],
        speedup=(root_ms / best_ms[0]) if root_ms and best_ms[0] else None,
        objective=cfg.objective,
        knobs=knobs,
        limits=limits,
        n_evaluated=stats["evaluated"],
        n_pruned=stats["pruned"],
        n_timed=stats["timed"],
        candidates=sorted(candidates, key=lambda c: c.objective_ms),
    )
    return _finish(result, cfg, exp, store)


def _finish(
    result: SearchResult,
    cfg: SelectionSearchConfig,
    exp: ExperimentConfig,
    store: ResultStore | None,
) -> SearchResult:
    """Persist (if a store is given), print, and return the result."""
    if store is not None:
        store.add(
            cfg.name or exp.name,
            result,
            symbols=exp.symbols,
            scalars=exp.scalar_args,
            vectorization=resolve_vectorize_config(
                exp.target, exp.gpu_vectorize, exp.gpu_vectorize_config
            ),
        )
    print(format_search(result, name=cfg.name or exp.name))
    return result


def format_search(
    result: SearchResult, name: str | None = None, max_rows: int | None = 10
) -> str:
    """A short human-readable summary of a search."""
    head = f"=== search: {name} ===" if name else "=== search ==="
    out = [head]
    out.append(
        f"knobs={len(result.knobs)}  evaluated={result.n_evaluated}  "
        f"pruned={result.n_pruned}  timed={result.n_timed}  "
        f"objective={result.objective}"
    )
    if result.unsatisfiable:
        out.append(
            "Budget unsatisfiable: even the all-highest configuration is over "
            "budget. Loosen the budget or raise the reference."
        )
        return "\n".join(out)

    highest = {k.name: k.highest for k in result.knobs}
    lowered = {n: f for n, f in (result.best or {}).items() if f != highest.get(n)}
    assign = (
        " ".join(f"{n}={f}" for n, f in sorted(lowered.items())) or "(nothing lowered)"
    )
    out.append(f"Fastest feasible found: {assign}")
    speed = f", {result.speedup:.2f}x over root" if result.speedup else ""
    out.append(f"  {result.best_ms:.4g} ms {result.objective}{speed}")

    if result.candidates:
        shown = result.candidates if max_rows is None else result.candidates[:max_rows]
        out.append("")
        out.append("--- measured feasible (fastest first) ---")
        for c in shown:
            pins = (
                " ".join(f"{k}={v}" for k, v in sorted(c.precision.items())) or "(root)"
            )
            out.append(f"  {c.objective_ms:>10.4g} ms  {c.speedup:>5.2f}x  {pins}")
        if len(shown) < len(result.candidates):
            out.append(f"  ... {len(result.candidates) - len(shown)} more")
    return "\n".join(out)
