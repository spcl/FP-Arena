# Copyright 2019-2026 ETH Zurich and the FP-Arena authors. All rights reserved.
"""Tests for the selection search (knobs, canonical typing, run_search)."""

import concurrent.futures as cf

import dace
import pytest
from dace.codegen.exceptions import CompilationError

import fp_arena  # noqa: F401
from fp_arena.experiment import (
    CONSTANTS_KEY,
    ErrorBudget,
    ExperimentConfig,
    KnobSelection,
    ResultStore,
    SelectionSearchConfig,
    find_knobs,
    run_search,
    search,
)
from fp_arena.experiment.knobs import canonical_typing
from fp_arena.experiment.search import format_search

N = dace.symbol("N")


@dace.program
def _axpy(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    for i in dace.map[0:N]:
        c[i] = a[i] * b[i] + c[i]


@dace.program
def _scaled(a: dace.float64[N], c: dace.float64[N]):
    for i in dace.map[0:N]:
        c[i] = a[i] * 2.5


_AXPY = _axpy.to_sdfg(simplify=True)
_SCALED = _scaled.to_sdfg(simplify=True)


def _experiment(sdfg, name="prog", n=64, **inputs):
    return ExperimentConfig(name=name, program=sdfg, symbols={"N": n}, inputs=inputs)


@pytest.fixture(autouse=True)
def _small_compile_pool(monkeypatch):
    """Keep real-process tests from spawning one compile worker per CPU core."""
    monkeypatch.setattr(search.os, "cpu_count", lambda: 2)


# --------------------------------------------------------------------------- knobs


def test_find_knobs_classifies_inputs_as_sources():
    knobs = {k.name: k for k in find_knobs(_AXPY)}
    # a, b, c are all read inputs (c is read-and-written): sources.
    for name in ("a", "b", "c"):
        assert knobs[name].kind == "source"
        assert knobs[name].original == "fp64"
        assert knobs[name].domain == ("fp16", "fp32", "fp64")
    # No float literals -> no constants knob.
    assert CONSTANTS_KEY not in knobs


def test_find_knobs_detects_constants():
    names = {k.name: k for k in find_knobs(_SCALED)}
    assert CONSTANTS_KEY in names
    assert names[CONSTANTS_KEY].kind == "constants"


def test_knob_selection_toggles_and_overrides():
    default = {k.name for k in find_knobs(_SCALED, KnobSelection())}
    assert CONSTANTS_KEY not in default  # constants off by default
    with_consts = {k.name for k in find_knobs(_SCALED, KnobSelection(constants=True))}
    assert CONSTANTS_KEY in with_consts
    excluded = {
        k.name for k in find_knobs(_SCALED, KnobSelection(exclude=frozenset({"a"})))
    }
    assert "a" not in excluded


def test_domain_capped_at_original():
    @dace.program
    def half(a: dace.float32[N], c: dace.float32[N]):
        for i in dace.map[0:N]:
            c[i] = a[i]

    knobs = {k.name: k for k in find_knobs(half.to_sdfg(simplify=True))}
    assert knobs["a"].domain == ("fp16", "fp32")  # no fp64 rung above fp32


# ------------------------------------------------------------------- canonical key


def test_canonical_typing_dedups_redundant_pins():
    exp = _experiment(_AXPY)
    root = canonical_typing(exp, {})
    # Pinning to the type propagation already gives is redundant -> same key.
    assert canonical_typing(exp, {"a": "fp64"}) == root
    # A genuine lowering is a different key.
    assert canonical_typing(exp, {"a": "fp16"}) != root
    # Constants are part of the key.
    assert canonical_typing(exp, {CONSTANTS_KEY: "fp32"}) != root


# --------------------------------------------------------------------- run_search


def test_search_tight_budget_keeps_root():
    """A budget only fp64 can meet -> the search returns the all-highest root."""
    exp = _experiment(_AXPY)
    # rel_max well below fp32's rounding but above fp64-vs-fp64 (which is exact).
    budget = ErrorBudget(limits={"rel_max": 1e-10})
    result = run_search(
        SelectionSearchConfig(
            experiment=exp,
            budget=budget,
            reference="fp64",
            n_samples=2,
            n_warmup=1,
            n_reps=3,
            objective="total",
        )
    )
    assert not result.unsatisfiable
    # Nothing lowered: every knob at its highest rung.
    highest = {k.name: k.highest for k in result.knobs}
    assert result.best == highest


def test_search_reports_unsatisfiable_when_root_fails():
    """A predicate no config can satisfy -> even the root fails -> unsatisfiable."""
    exp = _experiment(_AXPY)
    budget = ErrorBudget(limits={"rel_max": 1.0}, predicate=lambda errs: False)
    result = run_search(
        SelectionSearchConfig(
            experiment=exp,
            budget=budget,
            reference="fp64",
            n_samples=1,
            n_warmup=1,
            n_reps=2,
        )
    )
    assert result.unsatisfiable
    assert result.best is None


def test_search_worker_count_does_not_change_result():
    """The result is independent of measure_workers (1 pool worker vs several)."""
    exp = _experiment(_AXPY)
    budget = ErrorBudget(limits={"rel_max": 1e-10})  # only fp64 passes
    kw = {
        "experiment": exp,
        "budget": budget,
        "reference": "fp64",
        "n_samples": 1,
        "n_warmup": 1,
        "n_reps": 2,
        "objective": "total",
    }
    one = run_search(SelectionSearchConfig(measure_workers=1, **kw))
    many = run_search(SelectionSearchConfig(measure_workers=4, **kw))
    assert not many.unsatisfiable
    assert many.best == one.best  # nothing lowered, both
    assert many.n_pruned == one.n_pruned


def test_search_records_candidates_and_persists(tmp_path):
    """Feasible configs are recorded (root included) and the result is stored."""
    exp = _experiment(_AXPY)
    budget = ErrorBudget(limits={"rel_max": 1e-10})  # only the root passes
    store = ResultStore(str(tmp_path / "r.db"))
    try:
        result = run_search(
            SelectionSearchConfig(
                experiment=exp,
                budget=budget,
                reference="fp64",
                n_samples=1,
                n_warmup=1,
                n_reps=2,
            ),
            store=store,
        )
        # root is feasible -> at least one candidate, the root (nothing lowered, 1x).
        assert result.candidates
        assert result.candidates[0].precision == {}
        assert result.candidates[0].speedup == 1.0
        rows = store.query(kind="search")
        assert len(rows) == 1
        assert rows[0].payload["objective"] == "total"
        assert len(rows[0].payload["candidates"]) == len(result.candidates)
    finally:
        store.close()


class _InlinePool:
    """A drop-in ``ProcessPoolExecutor`` that runs every task in this process."""

    def __init__(self, max_workers, mp_context=None, initializer=None, initargs=()):
        if initializer is not None:
            initializer(*initargs)

    def submit(self, fn, *args, **kwargs):
        fut = cf.Future()
        try:
            fut.set_result(fn(*args, **kwargs))
        except BaseException as exc:  # noqa: BLE001
            fut.set_exception(exc)
        return fut

    def shutdown(self, wait=True):
        pass


def test_search_skips_configs_that_fail_to_build(monkeypatch):
    """A config DaCe cannot compile is recorded and skipped, not fatal."""
    log = 'Compiler failure:\nerror: no instance of function template "ITE"'
    original = search._compile_task

    def _compile(pin_map):
        if pin_map.get("a") == "fp32":
            raise CompilationError(log)
        return original(pin_map)

    monkeypatch.setattr(search, "_compile_task", _compile)
    monkeypatch.setattr(search.cf, "ProcessPoolExecutor", _InlinePool)

    result = run_search(
        SelectionSearchConfig(
            experiment=_experiment(_AXPY),
            budget=ErrorBudget(limits={"rel_max": 1e-10}),  # only the root passes
            reference="fp64",
            n_samples=1,
            n_warmup=1,
            n_reps=2,
        )
    )

    # The search survived and still returned the root as the best config.
    highest = {k.name: k.highest for k in result.knobs}
    assert result.best == highest
    # The failing config is recorded, with the whole compiler log kept verbatim.
    assert [f.precision for f in result.failures] == [{"a": "fp32"}]
    assert result.failures[0].error == f"CompilationError: {log}"
    printed = format_search(result)  # indented, but every line is there
    assert all(line in printed for line in log.splitlines())
    assert "failed=1" in printed


def test_search_build_failure_of_the_root_is_fatal(monkeypatch):
    """Without the root there is no baseline, so its build failure propagates."""

    def _compile(pin_map):
        raise CompilationError("boom")

    monkeypatch.setattr(search, "_compile_task", _compile)
    monkeypatch.setattr(search.cf, "ProcessPoolExecutor", _InlinePool)

    with pytest.raises(CompilationError, match="boom"):
        run_search(
            SelectionSearchConfig(
                experiment=_experiment(_AXPY),
                budget=ErrorBudget(limits={"rel_max": 1e-10}),
                n_samples=1,
                n_warmup=1,
                n_reps=2,
            )
        )


def test_search_no_knobs_raises():
    exp = _experiment(_AXPY)
    with pytest.raises(ValueError, match="No knobs in scope"):
        run_search(
            SelectionSearchConfig(
                experiment=exp,
                budget=ErrorBudget(limits={"rel_max": 1.0}),
                knobs=KnobSelection(sources=False, constants=False, overrides=False),
            )
        )
