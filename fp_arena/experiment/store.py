# Copyright 2019-2026 ETH Zurich and the FP-Arena authors. All rights reserved.
"""
Append-only results database.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from dace.transformation.passes.vectorization.config import VectorizeConfig

from fp_arena.experiment.results import ErrorResult, PerfResult, PerturbationResult

DEFAULT_DB_PATH = ".fp_arena_results.db"

#: The ``kind`` column value for each result type.
_KIND_OF = {
    PerfResult: "performance",
    ErrorResult: "error",
    PerturbationResult: "perturbation",
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS results (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    kind          TEXT NOT NULL,        -- 'performance' | 'error' | 'perturbation'
    experiment    TEXT NOT NULL,        -- ExperimentConfig.name
    created_at    TEXT NOT NULL,        -- ISO-8601 UTC
    precision     TEXT NOT NULL,        -- JSON: the precision map
    symbols       TEXT NOT NULL,        -- JSON: the SDFG free-symbol values
    scalars       TEXT NOT NULL,        -- JSON: the non-array scalar arguments
    vectorization TEXT NOT NULL,        -- JSON: the applied VectorizeConfig, null if unvectorized
    payload       TEXT NOT NULL         -- JSON: the full result record
);
CREATE INDEX IF NOT EXISTS idx_results_lookup ON results (experiment, kind);
"""


def _jsonable(value: Any) -> Any:
    """An enum knob as its string form (its value for the string enums, else its name)."""
    if isinstance(value, Enum):
        return value.value if isinstance(value.value, str) else value.name
    return value


def _vectorization_json(config: VectorizeConfig | None) -> str:
    """Serialise the applied vectorization config; ``None`` (no vectorization) is JSON ``null``."""
    if config is None:
        return "null"
    return json.dumps({k: _jsonable(v) for k, v in asdict(config).items()})


@dataclass
class StoredResult:
    """One row from the database."""

    id: int
    kind: str
    experiment: str
    created_at: str
    precision: dict[str, Any]
    symbols: dict[str, Any]
    scalars: dict[str, Any]
    payload: dict[str, Any]
    vectorization: dict[str, Any] | None = None


class ResultStore:
    """A thin SQLite wrapper for appending and querying experiment results."""

    def __init__(self, path: str = DEFAULT_DB_PATH):
        self.path = path
        self._con = sqlite3.connect(path, check_same_thread=False)
        self._con.row_factory = sqlite3.Row
        self._con.execute("PRAGMA journal_mode=WAL")
        self._con.execute("PRAGMA busy_timeout=30000")
        self._con.executescript(_SCHEMA)
        self._con.commit()

    def close(self) -> None:
        self._con.close()

    def add(
        self,
        experiment: str,
        result: PerfResult | ErrorResult | PerturbationResult,
        symbols: dict[str, Any] | None = None,
        scalars: dict[str, Any] | None = None,
        vectorization: VectorizeConfig | None = None,
    ) -> int:
        """Append one result record. :returns: the new row id."""
        kind = _KIND_OF.get(type(result))
        if kind is None:
            raise TypeError(f"Cannot store a {type(result).__name__}")
        with self._con:
            cur = self._con.execute(
                "INSERT INTO results (kind, experiment, created_at, precision, symbols, scalars, vectorization, payload) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    kind,
                    experiment,
                    datetime.now(timezone.utc).isoformat(),
                    json.dumps(result.precision),
                    json.dumps(symbols or {}, default=str),
                    json.dumps(scalars or {}, default=str),
                    _vectorization_json(vectorization),
                    json.dumps(result.to_dict()),
                ),
            )
            return int(cur.lastrowid)

    def query(
        self, experiment: str | None = None, kind: str | None = None
    ) -> list[StoredResult]:
        """Fetch stored results, newest first, optionally filtered by experiment/kind."""
        clauses, params = [], []
        if experiment is not None:
            clauses.append("experiment = ?")
            params.append(experiment)
        if kind is not None:
            clauses.append("kind = ?")
            params.append(kind)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = self._con.execute(
            f"SELECT * FROM results{where} ORDER BY id DESC", params
        ).fetchall()
        return [
            StoredResult(
                id=r["id"],
                kind=r["kind"],
                experiment=r["experiment"],
                created_at=r["created_at"],
                precision=json.loads(r["precision"]),
                symbols=json.loads(r["symbols"]),
                scalars=json.loads(r["scalars"]),
                payload=json.loads(r["payload"]),
                vectorization=json.loads(r["vectorization"]),
            )
            for r in rows
        ]
