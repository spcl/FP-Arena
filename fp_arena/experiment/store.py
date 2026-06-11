# Copyright 2019-2026 ETH Zurich and the FP-Arena authors. All rights reserved.
"""
Append-only results database.
"""

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fp_arena.experiment.results import ErrorResult, PerfResult

DEFAULT_DB_PATH = ".fp_arena_results.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS results (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    kind        TEXT NOT NULL,          -- 'performance' | 'error'
    experiment  TEXT NOT NULL,          -- ExperimentConfig.name
    created_at  TEXT NOT NULL,          -- ISO-8601 UTC
    precision   TEXT NOT NULL,          -- JSON: the precision map
    payload     TEXT NOT NULL           -- JSON: the full result record
);
CREATE INDEX IF NOT EXISTS idx_results_lookup ON results (experiment, kind);
"""


@dataclass
class StoredResult:
    """One row from the database."""

    id: int
    kind: str
    experiment: str
    created_at: str
    precision: Dict[str, Any]
    payload: Dict[str, Any]


class ResultStore:
    """A thin SQLite wrapper for appending and querying experiment results."""

    def __init__(self, path: str = DEFAULT_DB_PATH):
        self.path = path
        self._con = sqlite3.connect(path, check_same_thread=False)
        self._con.row_factory = sqlite3.Row
        self._con.executescript(_SCHEMA)
        self._con.commit()

    def close(self) -> None:
        self._con.close()

    def _add(
        self,
        kind: str,
        experiment: str,
        precision: Dict[str, Any],
        payload: Dict[str, Any],
    ) -> int:
        with self._con:
            cur = self._con.execute(
                "INSERT INTO results (kind, experiment, created_at, precision, payload) VALUES (?, ?, ?, ?, ?)",
                (
                    kind,
                    experiment,
                    datetime.now(timezone.utc).isoformat(),
                    json.dumps(precision),
                    json.dumps(payload),
                ),
            )
            return int(cur.lastrowid)

    def add_perf(self, experiment: str, result: PerfResult) -> int:
        """Append one :class:`PerfResult`. :returns: the new row id."""
        return self._add("performance", experiment, result.precision, result.to_dict())

    def add_error(self, experiment: str, result: ErrorResult) -> int:
        """Append one :class:`ErrorResult`. :returns: the new row id."""
        return self._add("error", experiment, result.precision, result.to_dict())

    # TODO: ``add_select`` and ``add_perturbation``.

    def query(
        self, experiment: Optional[str] = None, kind: Optional[str] = None
    ) -> List[StoredResult]:
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
                payload=json.loads(r["payload"]),
            )
            for r in rows
        ]
