# Copyright 2019-2026 ETH Zurich and the FP-Arena authors. All rights reserved.
"""
Selection search

Flow:

0. Enumerate and select the knobs; compile the reference once.
1. Perturb each input at fp16/fp32 magnitude on the baseline SDFG.
2. Seed a per-input candidate for each safe input (based on perturbation), and
   start a chain for each wide seed (all safe inputs, all fp32, all fp16).
3. Prioritise by benefit (big, low-sensitivity arrays lowered first) defined by the scoring function.
4. Evaluate: error-check each candidate; infeasible prunes its down-set,
   feasible is timed and expands its one-step-lower neighbours. A candidate that
   fails to build is recorded in ``SearchResult.failures`` and skipped.
5. Return the fastest measured feasible config.

Chains reach deep configs in big steps instead of one knob per round. A chain
keeps a base (its last feasible config) and the knobs of its goal still open. It
tries base + all open knobs; a pass becomes the new base, a fail is split into
halves tried on the base, and a single knob that fails is retried one rung
higher, or kept at its original precision. A chain's configs do not expand
their one-step neighbours. Only its final base does.
"""

from __future__ import annotations

import concurrent.futures as cf
import heapq
import itertools
import multiprocessing as mp
import os
import shutil
import subprocess
import warnings
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import dace
import numpy as np
from dace.codegen.compiler import load_precompiled_sdfg
from dace.codegen.exceptions import CodegenError, CompilationError
from dace.sdfg.validation import InvalidSDFGError

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
    SearchFailure,
    SearchResult,
)
from fp_arena.experiment.retarget import resolve_vectorize_config
from fp_arena.experiment.runner import (
    _accumulate,
    _checked_outputs,
    _collect_outputs,
    _copy_args,
    _finalize,
    _new_acc,
    _sample_rngs,
    build_candidate_sdfg,
    compile_reference,
    measure,
)
from fp_arena.experiment.screening import screen
from fp_arena.experiment.selection import _objective_ms, check
from fp_arena.experiment.store import ResultStore
from fp_arena.experiment.timers import insert_timers, timer_categories


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
    :param measure_workers: number of unpinned *measurement* worker processes when
        ``devices`` is ``None`` (CPU / testing); ignored when ``devices`` is given.
    :param devices: GPU ids to pin measurement workers to via
        ``CUDA_VISIBLE_DEVICES`` -- one worker per id, each owning its device
        exclusively.
    :param compile_workers: size of the compile pool that builds and
        compiles candidates ahead of the measurement workers. ``None`` uses
        ``os.cpu_count() - <measurement workers>``.
    :param compile_ahead: per measurement slot, how many configs may be keyed,
        compiling, or compiled and waiting for a slot. Bounds how far
        compilation runs ahead of measurement, so configs are picked from the
        queue with recent results (pruning, best-first order) in hand.
    :param grade_cpus_per_device: with ``devices``, each measurement worker is
        pinned to this many CPUs
    :param keep_builds: keep each candidate's build folder after it is graded.
    :param prune_above_feasible: also skip configs a feasible config lowers at
        least as far (covered: known to pass, assumed no faster). Finds the
        most-lowered feasible configs and stops, instead of the fastest overall.
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
    measure_workers: int = 1
    devices: list[int] | None = None
    compile_workers: int | None = None
    compile_ahead: int = 8
    grade_cpus_per_device: int = 8
    keep_builds: bool = False
    prune_above_feasible: bool = False
    name: str | None = None

    def __post_init__(self) -> None:
        if self.objective not in OBJECTIVES:
            raise ValueError(
                f"Unknown objective {self.objective!r}; expected one of {list(OBJECTIVES)}"
            )


#: A config: every in-scope knob mapped to a ladder rung.
Config = dict[str, str]


@dataclass(eq=False)
class _Chain:
    """
    One wide seed, lowered in big steps.

    :param label: the seed's name, for the log.
    :param base: the chain's last feasible config.
    :param open: the goal's knobs not lowered in ``base`` yet, with their target rung.
    :param done: nothing is open any more; ``base`` is the chain's result.
    """

    label: str
    base: Config
    open: dict[str, str]
    done: bool = False


class _PriorityQueue:
    """
    Items by rank, lowest first, ties in insertion order. ``put`` inserts an
    item or moves it up to a lower rank; its old entry is skipped when it comes up.
    """

    def __init__(self) -> None:
        self._heap: list = []
        self._rank: dict = {}
        self._counter = itertools.count()

    def put(self, ident, rank, item) -> None:
        if ident in self._rank and self._rank[ident] <= rank:
            return
        self._rank[ident] = rank
        heapq.heappush(self._heap, (rank, next(self._counter), ident, item))

    def pop(self):
        while True:
            rank, _, ident, item = heapq.heappop(self._heap)
            if self._rank.get(ident) == rank:
                del self._rank[ident]
                return item

    def __contains__(self, ident) -> bool:
        return ident in self._rank

    def __len__(self) -> int:
        return len(self._rank)


@dataclass(eq=False)
class _Job:
    """
    One program (canonical key) in flight and the configs waiting for its
    result: the first is graded, the others take its result.

    :param state: ``"compiling"``, ``"ready"`` (built, waiting for a slot) or
        ``"grading"``.
    """

    key: CanonicalKey
    configs: list[Config] = field(default_factory=list)
    state: str = "compiling"
    build_folder: str | None = None


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
        self.outputs = _checked_outputs(experiment)
        self.limits = limits
        self.n_warmup = n_warmup
        self.n_reps = n_reps
        reads, writes = experiment.program.read_and_write_sets()
        self._written = set(writes)
        ref = compile_reference(experiment, reference)
        self._samples: list[tuple[dict, dict[str, np.ndarray]]] = []
        for rng in _sample_rngs(experiment.seed, n_samples):
            args = make_call_args(experiment.program, experiment, rng, reads=reads)
            ref_args = _copy_args(args)
            ref(**ref_args)
            self._samples.append(
                (args, _collect_outputs(experiment, ref_args, self.outputs))
            )
        ref.finalize()

    def grade_and_time(
        self, build_folder: str, pin_map: PrecisionMap
    ) -> tuple[dict[str, ErrorStats], PerfResult | None]:
        """
        Grade the candidate compiled into ``build_folder`` against the reference
        and, when it meets the numeric limits, time it.
        """
        csdfg = load_precompiled_sdfg(build_folder)
        sdfg = csdfg.sdfg
        accs = {name: _new_acc() for name in self.outputs}
        for args, ref_out in self._samples:
            cand_args = _copy_args(args)
            csdfg(**cand_args)
            cand_out = _collect_outputs(self.experiment, cand_args, self.outputs)
            for name in self.outputs:
                _accumulate(accs[name], ref_out[name], cand_out[name])
        errors = {name: _finalize(a) for name, a in accs.items()}
        perf: PerfResult | None = None
        if all(c.ok for c in check(errors, self.limits)):
            rng = _sample_rngs(self.experiment.seed, 1)[0]
            # The error samples already ran on this compiled object; measure()
            # resets the timer buffer up front, so only its own reps are timed.
            perf = PerfResult(
                precision=dict(pin_map),
                seed=self.experiment.seed,
                **measure(
                    sdfg,
                    csdfg,
                    timer_categories(sdfg),
                    self.experiment,
                    self.n_warmup,
                    self.n_reps,
                    rng,
                    initial_args=self._samples[0][0],
                    written=self._written,
                ),
            )
        else:
            csdfg.finalize()
        csdfg._lib.unload()
        return errors, perf


_WORKER: dict = {}


def _worker_dace_config(experiment: ExperimentConfig) -> None:
    """Config shared by both pools. ``use_cache`` lets a measurement worker
    reload a compile worker's ``.so`` instead of rebuilding it; the CUDA stream
    setting affects codegen, so it must match on the pool that generates code."""
    dace.Config.set("compiler", "use_cache", value=True)
    if experiment.target == "gpu":
        dace.Config.set("compiler", "cuda", "max_concurrent_streams", value=-1)


def _init_slot(
    experiment: ExperimentConfig,
    reference: str | dict[str, str],
    n_samples: int,
    limits: dict[str, dict[str, float]],
    n_warmup: int,
    n_reps: int,
    device_queue,
) -> None:
    # Pin the device and the CPU cores
    device, cpus = device_queue.get()
    if device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    if cpus:
        os.sched_setaffinity(0, cpus)
    _worker_dace_config(experiment)
    _WORKER["slot"] = _Slot(experiment, reference, n_samples, limits, n_warmup, n_reps)


def _init_compiler(experiment: ExperimentConfig, cpus: set[int] | None) -> None:
    if cpus:
        os.sched_setaffinity(0, cpus)
    _worker_dace_config(experiment)
    _WORKER["experiment"] = experiment


def _parse_cpulist(text: str) -> list[int]:
    """Parse a kernel CPU list such as ``"0-71,144-150"``."""
    cpus: list[int] = []
    for part in text.strip().split(","):
        if part:
            lo, _, hi = part.partition("-")
            cpus.extend(range(int(lo), int(hi or lo) + 1))
    return cpus


def _gpu_local_cpus(device: int) -> list[int]:
    """The CPU cores attached to GPU ``device`` (a physical index), ascending."""
    bus_id = subprocess.run(
        [
            "nvidia-smi",
            f"--id={device}",
            "--query-gpu=pci.bus_id",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    # nvidia-smi pads the PCI domain to 8 hex digits, sysfs to 4.
    domain, _, rest = bus_id.partition(":")
    path = Path("/sys/bus/pci/devices") / f"{int(domain, 16):04x}:{rest.lower()}"
    return _parse_cpulist((path / "local_cpulist").read_text())


def _partition_cpus(
    devices: list[int | None], per_device: int
) -> tuple[list[set[int] | None], set[int] | None]:
    """Cores for each measurement worker and for the compile pool."""
    if per_device <= 0 or any(d is None for d in devices):
        return [None] * len(devices), None
    allowed = os.sched_getaffinity(0)
    grade: list[set[int] | None] = []
    for device in devices:
        local = [c for c in _gpu_local_cpus(device) if c in allowed]
        grade.append(set(local[-per_device:]) or None)
    reserved = set().union(*(g for g in grade if g))
    rest = allowed - reserved
    if not rest:
        warnings.warn(
            "No cores left for the compile pool; not pinning it.", stacklevel=2
        )
    return grade, rest or None


def _key_task(pin_map: PrecisionMap) -> CanonicalKey:
    """The dedup key of one candidate."""
    return canonical_typing(_WORKER["experiment"], pin_map)


def _compile_task(pin_map: PrecisionMap) -> str:
    """Build and compile one candidate SDFG, timers inserted."""
    exp = _WORKER["experiment"]
    sdfg = build_candidate_sdfg(exp, pin_map)
    insert_timers(sdfg, exp.target)
    sdfg.build_folder = os.path.abspath(sdfg.build_folder)
    sdfg.compile()
    return sdfg.build_folder


def _grade_task(
    build_folder: str,
    pin_map: PrecisionMap,
) -> tuple[dict[str, ErrorStats], PerfResult | None]:
    return _WORKER["slot"].grade_and_time(build_folder, pin_map)


#: Exceptions that condemn one config, not the search. The config is recorded and skipped.
_CONFIG_FAILURES = (CompilationError, CodegenError, InvalidSDFGError)


def _failure_text(exc: BaseException) -> str:
    """The exception's type and its message"""
    return f"{type(exc).__name__}: {exc}"


def _format_pins(precision: PrecisionMap) -> str:
    """One config's lowered knobs, sorted; ``(root)`` when nothing is lowered."""
    return " ".join(f"{k}={v}" for k, v in sorted(precision.items())) or "(root)"


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
    feasible: list[tuple[int, ...]] = []  # only kept with prune_above_feasible

    def dominated(config: Config) -> bool:
        v = vec(config)
        return any(all(a <= b for a, b in zip(v, iv, strict=True)) for iv in infeasible)

    def covered(config: Config) -> bool:
        v = vec(config)
        return any(all(a >= b for a, b in zip(v, fv, strict=True)) for fv in feasible)

    def neighbors(config: Config) -> list[Config]:
        out: list[Config] = []
        for name in names:
            i = rung[name][config[name]]
            if i > 0:
                nb = dict(config)
                nb[name] = knob_of[name].domain[i - 1]
                out.append(nb)
        return out

    queue = _PriorityQueue()  # configs to key and build, by vec
    visited: set[tuple[int, ...]] = set()  # traversal dedup, over the config lattice
    # evaluated: builds started (the compute budget); compiled/graded: finished;
    # cached: dedup hits, resolved by an earlier config's result; failed: configs
    # whose program does not build (their own build or a known failing one);
    # covered: skipped as known to pass (prune_above_feasible); pruned_built:
    # builds thrown away because all their configs were pruned or covered after it.
    stats = {
        "evaluated": 0,
        "compiled": 0,
        "graded": 0,
        "cached": 0,
        "pruned": 0,
        "pruned_built": 0,
        "covered": 0,
        "failed": 0,
        "timed": 0,
    }
    # Every resolved config's outcome (pruned and failed builds count as
    # infeasible, covered ones as feasible) and the chains waiting on a queued one.
    outcome: dict[tuple[int, ...], bool] = {}
    waiters: dict[tuple[int, ...], list[tuple[_Chain, frozenset[str]]]] = {}

    def rank(config: Config) -> tuple[bool, float]:
        """Queue order: chain configs first, then highest score."""
        return (vec(config) not in waiters, -score(config))

    def skip(config: Config) -> bool:
        """Settle ``config`` without building it when its outcome is known: pruned
        (fails like an infeasible config it lowers at least as far) or covered
        (passes like a feasible config that lowers it at least as far). A covered
        config is not expanded: everything above a feasible config is covered."""
        if dominated(config):
            stats["pruned"] += 1
            resolve(config, False)
            return True
        if covered(config):
            stats["covered"] += 1
            resolve(config, True)
            return True
        return False

    def push(config: Config) -> None:
        v = vec(config)
        if v in visited:
            return
        visited.add(v)
        if skip(config):
            return
        queue.put(v, rank(config), config)

    def expand(config: Config) -> None:
        for nb in neighbors(config):
            push(nb)

    def resolve(config: Config, ok: bool) -> None:
        """Record a config's outcome and hand it to the chains waiting on it."""
        v = vec(config)
        outcome[v] = ok
        for chain, piece in waiters.pop(v, []):
            chain_result(chain, config, piece, ok)

    def chain_try(chain: _Chain, piece: Iterable[str]) -> None:
        """Queue the chain's base with ``piece``'s open knobs at their target."""
        add = {n: chain.open[n] for n in piece if n in chain.open}
        if chain.done or not add:
            return
        config = {**chain.base, **add}
        v = vec(config)
        if v in outcome:
            chain_result(chain, config, frozenset(add), outcome[v])
            return
        if any(c is chain for c, _ in waiters.get(v, [])):
            return  # already waiting on it
        waiters.setdefault(v, []).append((chain, frozenset(add)))
        if v in visited:
            promote(config)  # already queued or in flight: move it up
        else:
            push(config)

    def waiting(chain: _Chain) -> bool:
        """Whether ``chain`` waits on any config's result."""
        return any(c is chain for ws in waiters.values() for c, _ in ws)

    def chain_result(
        chain: _Chain, config: Config, piece: frozenset[str], ok: bool
    ) -> None:
        """Advance ``chain`` by the outcome of ``config`` (base + ``piece``)."""
        if chain.done:
            return
        # Only the knobs still open at the rung this config tried count.
        tried = [n for n in piece if chain.open.get(n) == config[n]]
        if ok:
            # Adopt it unless it was built on an older, less lowered base.
            if all(rung[n][config[n]] <= rung[n][chain.base[n]] for n in names):
                chain.base = config
                chain.open = {n: f for n, f in chain.open.items() if config[n] != f}
            chain_try(chain, chain.open)
        elif len(tried) < len(piece):
            pass  # stale: some of its knobs changed since, so it tells nothing now
        elif len(tried) > 1:
            # Sensitive knobs land in the same half, so the other half can pass.
            order = sorted(tried, key=lambda n: (screening.sensitivity.get(n, 0.0), n))
            half = len(order) // 2
            chain_try(chain, order[:half])
            chain_try(chain, order[half:])
        else:
            (name,) = tried
            failed = chain.open[name]
            up = knob_of[name].domain[rung[name][failed] + 1]
            if up == chain.base[name]:
                del chain.open[name]
                print(f"[chain {chain.label}] {name}={failed} fails, keeping {up}")
            else:
                chain.open[name] = up
                print(f"[chain {chain.label}] {name}={failed} fails, trying {up}")
            chain_try(chain, chain.open)
        if not chain.open and not chain.done:
            chain.done = True
            print(
                f"[chain {chain.label}] done: {_format_pins(pin_map(chain.base))}",
                flush=True,
            )
            expand(chain.base)

    ctx = mp.get_context("spawn")
    # Measurement slots: one per device (each pinned + exclusive);
    # Compilation is split off onto a separate, larger pool of unpinned workers
    devices = cfg.devices if cfg.devices else [None] * max(1, cfg.measure_workers)
    n_slots = len(devices)
    n_compilers = (
        max(1, cfg.compile_workers)
        if cfg.compile_workers is not None
        else max(1, (os.cpu_count() or 1) - n_slots)
    )
    # Configs allowed between the queue and a measurement slot at once.
    lookahead = max(1, cfg.compile_ahead) * n_slots

    errors_cache: dict[
        CanonicalKey, dict[str, ErrorStats]
    ] = {}  # dedup across pin-maps
    perf_cache: dict[CanonicalKey, PerfResult] = {}
    failed_keys: set[CanonicalKey] = set()  # never recompile a config that failed
    failures: list[SearchFailure] = []
    # Pipeline: a config is keyed, then subscribes to the job of its program (one
    # build and grading per canonical key). The job compiles on the compile pool,
    # waits in ``ready`` for a measurement slot, and its result settles every
    # subscriber. A job ranks as its best subscriber, so a chain waiting on any of
    # them moves it up.
    jobs: dict[CanonicalKey, _Job] = {}
    job_of: dict[tuple[int, ...], _Job] = {}  # subscribed config -> its job
    key_futs: dict[cf.Future, Config] = {}
    compile_futs: dict[cf.Future, _Job] = {}
    ready = _PriorityQueue()  # built jobs, by key
    grade_futs: dict[cf.Future, _Job] = {}

    def job_rank(job: _Job) -> tuple[bool, float]:
        return min(rank(c) for c in job.configs)

    def promote(config: Config) -> None:
        """Move an already visited config up after a chain started waiting on it."""
        v = vec(config)
        if v in queue:
            queue.put(v, rank(config), config)
        elif v in job_of and job_of[v].state == "ready":
            job = job_of[v]
            ready.put(job.key, job_rank(job), job)

    best: list[Config] = [root]
    best_ms: list[float | None] = [None]
    candidates: list[SearchCandidate] = []

    def key_of(config: Config) -> CanonicalKey:
        return canonical_typing(exp, pin_map(config))

    def progress(outcome: str) -> str:
        """A result line's head"""
        return (
            f"[{outcome:<10} | feasible {len(candidates)}, "
            f"infeasible {len(infeasible)}, compiled {stats['compiled']}, "
            f"graded {stats['graded']}, cached {stats['cached']}, "
            f"pruned {stats['pruned']}, covered {stats['covered']} "
            f"({stats['pruned_built']} skipped after build), failed {stats['failed']}]"
        )

    def update_best(config: Config, perf: PerfResult, cached: bool) -> None:
        ms = _objective_ms(perf, cfg.objective)
        pins = pin_map(config)
        speedup = (root_ms / ms) if root_ms and ms else 0.0
        candidates.append(SearchCandidate(pins, ms, speedup))
        improved = best_ms[0] is None or ms < best_ms[0]
        if improved:
            best_ms[0], best[0] = ms, config
        mark = "<- new best" if improved else "(cached)" if cached else ""
        print(
            f"{progress('feasible')} {ms:>10.4g} ms "
            f"{speedup:>5.2f}x  {mark:<11}  {_format_pins(pins)}",
            flush=True,
        )

    def integrate(
        config: Config,
        key: CanonicalKey,
        errors: dict[str, ErrorStats],
        perf: PerfResult | None,
        cached: bool = False,
    ) -> None:
        """Fold one config's result in. ``perf`` is this config's own timing, or
        ``None`` when it was over the numeric budget or this is a dedup hit (its
        timing, if any, already sits in ``perf_cache``). ``cached``: a dedup hit,
        never built itself."""
        ok = _feasible(errors, limits, cfg.budget)
        if ok:
            if cfg.prune_above_feasible:
                feasible.append(vec(config))
            if key in perf_cache:
                # dedup: reuse the timing
                update_best(config, perf_cache[key], cached)
            elif perf is not None:
                perf_cache[key] = perf
                stats["timed"] += 1
                update_best(config, perf, cached)
            # A chain's configs do not expand; its final base does when it ends.
            if vec(config) not in waiters:
                expand(config)
        else:
            infeasible.append(vec(config))
            over = [c for c in check(errors, limits) if not c.ok]
            worst = max(
                over,
                key=lambda c: abs(c.value / c.limit) if c.limit else float("inf"),
                default=None,
            )
            detail = (
                f"{len(over)} over, worst {worst.metric} {worst.array}={worst.value:.3g} "
                f"(limit {worst.limit:.3g})"
                if worst is not None
                else "predicate failed"
            )
            print(
                f"{progress('infeasible')} "
                f"{'(cached) ' if cached else ''}{detail}  "
                f"{_format_pins(pin_map(config))}",
                flush=True,
            )
        resolve(config, ok)

    # Each measurement worker pops one (device, cores) pair from the queue at
    # init and pins to both; the compile pool runs on the remaining cores.
    grade_cpus, compile_cpus = _partition_cpus(devices, cfg.grade_cpus_per_device)
    device_queue = ctx.Queue()
    for device, cpus in zip(devices, grade_cpus, strict=True):
        device_queue.put((device, cpus))
    compile_pool = cf.ProcessPoolExecutor(
        n_compilers,
        mp_context=ctx,
        initializer=_init_compiler,
        initargs=(exp, compile_cpus),
    )
    grade_pool = cf.ProcessPoolExecutor(
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
    # Delete the build folders of configs that were compiled and graded
    cleaner = cf.ThreadPoolExecutor(1, thread_name_prefix="rm-build")

    def discard(folder: str) -> None:
        if not cfg.keep_builds:
            cleaner.submit(shutil.rmtree, folder, ignore_errors=True)

    root_ms = None
    try:
        # Root first: the feasibility floor and the speedup denominator.
        visited.add(vec(root))
        root_key = key_of(root)
        stats["evaluated"] += 1
        root_folder = compile_pool.submit(_compile_task, pin_map(root)).result()
        stats["compiled"] += 1
        root_errors, root_perf = grade_pool.submit(
            _grade_task, root_folder, pin_map(root)
        ).result()
        stats["graded"] += 1
        discard(root_folder)
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
        update_best(root, root_perf, cached=False)
        outcome[vec(root)] = True
        if cfg.prune_above_feasible:
            feasible.append(vec(root))

        # Seeds: per-input safe candidates, a chain per wide seed, then neighbours.
        for name, fmt in screening.safe_format.items():
            if name in root:
                cand = dict(root)
                cand[name] = fmt
                push(cand)
        goals = {"safe": {n: f for n, f in screening.safe_format.items() if n in root}}
        for fmt in ("fp32", "fp16"):
            goals[fmt] = {n: fmt for n in names if fmt in rung[n] and fmt != highest[n]}
        chains = [_Chain(label, base=root, open=goal) for label, goal in goals.items()]
        for chain in chains:
            chain_try(chain, chain.open)
        expand(root)

        def subscribe(config: Config, key: CanonicalKey) -> None:
            """Settle a keyed config: skipped, dedup hit, known build failure, or
            subscribed to its program's job (started if none is in flight)."""
            if skip(config):
                return
            if key in errors_cache:
                stats["cached"] += 1
                integrate(config, key, errors_cache[key], None, cached=True)
                return
            if key in failed_keys:  # same program, same build failure
                stats["failed"] += 1
                resolve(config, False)
                return
            job = jobs.get(key)
            if job is None:
                job = jobs[key] = _Job(key)
                stats["evaluated"] += 1
                compile_futs[compile_pool.submit(_compile_task, pin_map(config))] = job
            job.configs.append(config)
            job_of[vec(config)] = job
            if job.state == "ready":  # the new subscriber may rank it higher
                ready.put(key, job_rank(job), job)

        def finish(job: _Job) -> None:
            """Retire ``job``: nothing waits on it any more."""
            del jobs[job.key]
            for v in [v for v, j in job_of.items() if j is job]:
                del job_of[v]

        budget_hit = False
        while True:
            # A chain whose last results all came back stale waits on nothing:
            # it goes on from its current state.
            for chain in chains:
                if not chain.done and not waiting(chain):
                    chain_try(chain, chain.open)
            if not (
                (queue and not budget_hit)
                or key_futs
                or compile_futs
                or ready
                or grade_futs
            ):
                break
            # Hand built jobs to free measurement slots first, re-checking their
            # configs: results that landed while it was building may have pruned
            # or covered them, sparing its measurement.
            while ready and len(grade_futs) < n_slots:
                job = ready.pop()
                job.configs = [c for c in job.configs if not skip(c)]
                if not job.configs:  # all pruned or covered while it was built
                    finish(job)
                    stats["pruned_built"] += 1
                    discard(job.build_folder)
                    continue
                job.state = "grading"
                graded = pin_map(job.configs[0])
                grade_futs[grade_pool.submit(_grade_task, job.build_folder, graded)] = (
                    job
                )

            # Feed the compile pool from the queue; each config is keyed first.
            # Stop when the compile pool is full, the lookahead is reached, or the budget is hit.
            while (
                queue
                and not budget_hit
                and len(key_futs) + len(compile_futs) < n_compilers
                and len(key_futs) + len(compile_futs) + len(ready) < lookahead
            ):
                if cfg.compute_budget is not None:
                    if stats["evaluated"] >= cfg.compute_budget:
                        budget_hit = True
                        break
                    # Each config still being keyed may use one more unit of the
                    # budget (or turn out a dedup hit); wait for those first.
                    if stats["evaluated"] + len(key_futs) >= cfg.compute_budget:
                        break
                config = queue.pop()
                if skip(config):
                    continue
                key_futs[compile_pool.submit(_key_task, pin_map(config))] = config

            if not key_futs and not compile_futs and not grade_futs:
                continue  # nothing in flight: back to the top, which may end the search
            done, _ = cf.wait(
                list(key_futs) + list(compile_futs) + list(grade_futs),
                return_when=cf.FIRST_COMPLETED,
            )
            for fut in done:
                if fut in key_futs:
                    subscribe(key_futs.pop(fut), fut.result())
                    continue
                if fut in compile_futs:
                    job = compile_futs.pop(fut)
                    try:
                        job.build_folder = fut.result()
                    except _CONFIG_FAILURES as exc:
                        # Never became a program: neither feasible nor infeasible.
                        text = _failure_text(exc)
                        failed_keys.add(job.key)
                        built = pin_map(job.configs[0])
                        failures.append(SearchFailure(built, text))
                        warnings.warn(
                            f"Skipping config {built}: it failed to build.\n{text}",
                            stacklevel=2,
                        )
                        finish(job)
                        for config in job.configs:
                            stats["failed"] += 1
                            resolve(config, False)
                        continue
                    stats["compiled"] += 1
                    job.state = "ready"
                    ready.put(job.key, job_rank(job), job)
                    continue
                job = grade_futs.pop(fut)
                errors, perf = fut.result()
                stats["graded"] += 1
                discard(job.build_folder)
                errors_cache[job.key] = errors
                finish(job)
                first, *rest = job.configs
                integrate(first, job.key, errors, perf)
                for config in rest:  # same program: a dedup hit on its result
                    subscribe(config, job.key)
        if queue:
            print(
                f"[budget] compute budget reached: {len(queue)} queued configs "
                "left unexplored",
                flush=True,
            )
    finally:
        compile_pool.shutdown(wait=True)
        grade_pool.shutdown(wait=True)
        for job in jobs.values():
            if job.build_folder is not None:
                discard(job.build_folder)
        cleaner.shutdown(wait=True)

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
        failures=failures,
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


def format_search(result: SearchResult, name: str | None = None) -> str:
    """A short human-readable summary of a search."""
    head = f"=== search: {name} ===" if name else "=== search ==="
    out = [head]
    out.append(
        f"knobs={len(result.knobs)}  evaluated={result.n_evaluated}  "
        f"pruned={result.n_pruned}  timed={result.n_timed}  "
        f"failed={len(result.failures)}  objective={result.objective}"
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
        out.append("")
        out.append("--- measured feasible (fastest first) ---")
        for c in result.candidates:
            pins = _format_pins(c.precision)
            out.append(f"  {c.objective_ms:>10.4g} ms  {c.speedup:>5.2f}x  {pins}")

    if result.failures:
        out.append("")
        out.append(f"--- failed to build ({len(result.failures)}), skipped ---")
        for f in result.failures:
            out.append(f"  {_format_pins(f.precision)}")
            out.extend(f"    {line}" for line in f.error.splitlines())
    return "\n".join(out)
