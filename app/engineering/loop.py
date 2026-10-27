"""A continuous loop that records planner scores and blocks a weaker revision."""

import json
import sqlite3
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from app.engineering.improve import ImprovementDecision

LOOP_MANDATORY = ("safety", "correctness", "budget", "recovery")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS engineering_loop_snapshots (
    planner_version TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    metrics TEXT NOT NULL,
    allowed INTEGER NOT NULL,
    regressions TEXT NOT NULL DEFAULT '[]',
    promoted INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (planner_version, recorded_at)
);
CREATE TABLE IF NOT EXISTS engineering_loop_baseline (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    planner_version TEXT NOT NULL,
    metrics TEXT NOT NULL,
    recorded_at TEXT NOT NULL
);
"""


@dataclass(frozen=True)
class LoopSnapshot:
    planner_version: str
    metrics: dict[str, float]
    allowed: bool
    regressions: tuple[str, ...] = ()
    promoted: bool = False
    recorded_at: str = ""

    def to_payload(self) -> dict:
        payload = asdict(self)
        payload["regressions"] = list(self.regressions)
        return payload


@dataclass
class LoopState:
    baseline: LoopSnapshot | None = None
    history: list[LoopSnapshot] = field(default_factory=list)


def compare_metrics(
    baseline: dict[str, float],
    candidate: dict[str, float],
    *,
    mandatory: tuple[str, ...] = LOOP_MANDATORY,
) -> ImprovementDecision:
    """Block a planner revision when a mandatory suite score falls."""
    regressions: list[str] = []
    for name in mandatory:
        before = float(baseline.get(name, 0.0) or 0.0)
        after = candidate.get(name)
        if after is None:
            regressions.append(name)
            continue
        if float(after) + 1e-9 < before:
            regressions.append(name)
    if regressions:
        return ImprovementDecision(
            False,
            tuple(regressions),
            "Mandatory loop checks regressed: " + ", ".join(regressions) + ".",
        )
    return ImprovementDecision(True, (), "Planner revision meets the loop floors.")


class ContinuousImprovementLoop:
    """Store suite snapshots, compare them to the promoted baseline, and refuse regressions."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._history: list[LoopSnapshot] = []
        self._baseline: LoopSnapshot | None = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as connection:
                connection.executescript(_SCHEMA)

    def record(
        self,
        planner_version: str,
        metrics: dict[str, float],
        *,
        promote: bool = False,
    ) -> LoopSnapshot:
        baseline = self.baseline()
        decision = (
            compare_metrics(baseline.metrics, metrics)
            if baseline is not None
            else ImprovementDecision(True, (), "First loop snapshot becomes the baseline.")
        )
        should_promote = bool(promote and decision.allowed)
        if baseline is None:
            should_promote = True
        snapshot = LoopSnapshot(
            planner_version=planner_version,
            metrics={key: float(value) for key, value in metrics.items()},
            allowed=decision.allowed,
            regressions=decision.regressions,
            promoted=should_promote,
            recorded_at=_now(),
        )
        with self._lock:
            if not self._history:
                self._history = self._load_history()
            self._history.append(snapshot)
            if should_promote:
                self._baseline = snapshot
            self._persist(snapshot)
        return snapshot

    def promote(self, planner_version: str, metrics: dict[str, float]) -> LoopSnapshot:
        return self.record(planner_version, metrics, promote=True)

    def restore_baseline(self, snapshot: LoopSnapshot) -> LoopSnapshot:
        """Put a previously promoted revision back into service."""
        restored = LoopSnapshot(
            planner_version=snapshot.planner_version,
            metrics={key: float(value) for key, value in snapshot.metrics.items()},
            allowed=True,
            promoted=True,
            recorded_at=_now(),
        )
        with self._lock:
            if not self._history:
                self._history = self._load_history()
            self._history.append(restored)
            self._baseline = restored
            self._persist(restored)
        return restored

    def baseline(self) -> LoopSnapshot | None:
        with self._lock:
            if self._baseline is not None:
                return self._baseline
            loaded = self._load_baseline()
            self._baseline = loaded
            return loaded

    def history(self) -> list[LoopSnapshot]:
        with self._lock:
            if self._history:
                return list(self._history)
            loaded = self._load_history()
            self._history = loaded
            return list(loaded)

    def state(self) -> LoopState:
        return LoopState(baseline=self.baseline(), history=self.history())

    def _persist(self, snapshot: LoopSnapshot) -> None:
        if self._path is None:
            return
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO engineering_loop_snapshots
                    (planner_version, recorded_at, metrics, allowed, regressions, promoted)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot.planner_version,
                    snapshot.recorded_at,
                    json.dumps(snapshot.metrics),
                    int(snapshot.allowed),
                    json.dumps(list(snapshot.regressions)),
                    int(snapshot.promoted),
                ),
            )
            if snapshot.promoted:
                connection.execute(
                    """
                    INSERT INTO engineering_loop_baseline (id, planner_version, metrics, recorded_at)
                    VALUES (1, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        planner_version = excluded.planner_version,
                        metrics = excluded.metrics,
                        recorded_at = excluded.recorded_at
                    """,
                    (snapshot.planner_version, json.dumps(snapshot.metrics), snapshot.recorded_at),
                )

    def _load_history(self) -> list[LoopSnapshot]:
        if self._path is None:
            return []
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM engineering_loop_snapshots ORDER BY recorded_at"
            ).fetchall()
        return [_snapshot(row) for row in rows]

    def _load_baseline(self) -> LoopSnapshot | None:
        if self._path is None:
            return None
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM engineering_loop_baseline WHERE id = 1").fetchone()
        if row is None:
            return None
        return LoopSnapshot(
            planner_version=row["planner_version"],
            metrics=json.loads(row["metrics"] or "{}"),
            allowed=True,
            promoted=True,
            recorded_at=row["recorded_at"],
        )

    def _connect(self) -> sqlite3.Connection:
        assert self._path is not None
        connection = sqlite3.connect(self._path, timeout=5)
        connection.row_factory = sqlite3.Row
        return connection


def _snapshot(row: sqlite3.Row) -> LoopSnapshot:
    return LoopSnapshot(
        planner_version=row["planner_version"],
        metrics=json.loads(row["metrics"] or "{}"),
        allowed=bool(row["allowed"]),
        regressions=tuple(json.loads(row["regressions"] or "[]")),
        promoted=bool(row["promoted"]),
        recorded_at=row["recorded_at"],
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
