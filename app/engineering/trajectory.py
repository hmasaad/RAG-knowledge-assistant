"""Persist ordered execution events for one agent run."""

import json
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RUN_STARTED = "run_started"
TOOL_OUTCOME = "tool_outcome"
ERROR = "error"
VERIFICATION = "verification"
POLICY = "policy"
INTERRUPTED = "interrupted"
RUN_FINISHED = "run_finished"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS engineering_runs (
    id TEXT PRIMARY KEY,
    agent_version TEXT NOT NULL DEFAULT '',
    session_id TEXT NOT NULL DEFAULT '',
    trace_id TEXT NOT NULL DEFAULT '',
    question TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'running',
    started_at TEXT NOT NULL,
    finished_at TEXT,
    duration_ms REAL NOT NULL DEFAULT 0,
    token_usage INTEGER NOT NULL DEFAULT 0,
    estimated_cost REAL NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS engineering_events (
    run_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    kind TEXT NOT NULL,
    action TEXT NOT NULL DEFAULT '',
    ok INTEGER,
    error TEXT NOT NULL DEFAULT '',
    payload TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    PRIMARY KEY (run_id, seq)
);
CREATE INDEX IF NOT EXISTS idx_engineering_events_run ON engineering_events(run_id, seq);
CREATE TABLE IF NOT EXISTS engineering_evaluations (
    run_id TEXT PRIMARY KEY,
    report TEXT NOT NULL
);
"""


@dataclass(frozen=True)
class TrajectoryEvent:
    seq: int
    kind: str
    action: str = ""
    ok: bool | None = None
    error: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""


@dataclass
class Trajectory:
    id: str
    agent_version: str = ""
    session_id: str = ""
    trace_id: str = ""
    question: str = ""
    status: str = "running"
    started_at: str = ""
    finished_at: str = ""
    duration_ms: float = 0.0
    token_usage: int = 0
    estimated_cost: float = 0.0
    events: list[TrajectoryEvent] = field(default_factory=list)


class TrajectoryRecorder:
    """Append-only event log shared by the API process and a worker."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._memory: dict[str, Trajectory] = {}
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as connection:
                connection.executescript(_SCHEMA)

    def start(
        self,
        *,
        run_id: str | None = None,
        agent_version: str = "",
        session_id: str = "",
        trace_id: str = "",
        question: str = "",
    ) -> str:
        rid = (run_id or "").strip() or uuid.uuid4().hex[:16]
        now = _now()
        record = Trajectory(
            id=rid,
            agent_version=agent_version,
            session_id=session_id,
            trace_id=trace_id,
            question=question,
            status="running",
            started_at=now,
        )
        created = False
        with self._lock:
            existing = self._lookup(rid)
            if existing is None:
                self._memory[rid] = record
                self._persist_run(record)
                created = True
        if created:
            self.record(rid, RUN_STARTED, action="start", ok=True, question=question)
        return rid

    def record(
        self,
        run_id: str,
        kind: str,
        *,
        action: str = "",
        ok: bool | None = None,
        error: str = "",
        **payload: Any,
    ) -> TrajectoryEvent | None:
        with self._lock:
            run = self._lookup(run_id)
            if run is None:
                return None
            event = TrajectoryEvent(
                seq=len(run.events),
                kind=kind,
                action=action,
                ok=ok,
                error=error,
                payload={key: value for key, value in payload.items() if value is not None},
                created_at=_now(),
            )
            run.events.append(event)
            self._persist_event(run_id, event)
        return event

    def finish(
        self,
        run_id: str,
        *,
        status: str = "completed",
        duration_ms: float = 0.0,
        token_usage: int = 0,
        estimated_cost: float = 0.0,
        error: str = "",
    ) -> Trajectory | None:
        kind = INTERRUPTED if status in {"halted", "cancelled", "interrupted"} else RUN_FINISHED
        self.record(run_id, kind, action="finish", ok=status == "completed", error=error, status=status)
        with self._lock:
            run = self._lookup(run_id)
            if run is None:
                return None
            run.status = status
            run.finished_at = _now()
            if duration_ms <= 0 and run.started_at:
                duration_ms = _elapsed_ms(run.started_at, run.finished_at)
            run.duration_ms = duration_ms
            run.token_usage = token_usage
            run.estimated_cost = estimated_cost
            self._persist_run(run)
            return self._copy(run)

    def get(self, run_id: str) -> Trajectory | None:
        with self._lock:
            run = self._lookup(run_id)
            return self._copy(run) if run is not None else None

    def find_by_session(self, session_id: str) -> Trajectory | None:
        with self._lock:
            for run in self._memory.values():
                if run.session_id == session_id:
                    return self._copy(run)
            if self._path is None:
                return None
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT id FROM engineering_runs WHERE session_id = ? ORDER BY started_at DESC LIMIT 1",
                    (session_id,),
                ).fetchone()
            if row is None:
                return None
            run = self._lookup(row["id"])
            return self._copy(run) if run is not None else None

    def save_report(self, run_id: str, payload: dict[str, Any]) -> None:
        with self._lock:
            if self._path is None:
                return
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO engineering_evaluations (run_id, report)
                    VALUES (?, ?)
                    ON CONFLICT(run_id) DO UPDATE SET report = excluded.report
                    """,
                    (run_id, json.dumps(payload, default=str)),
                )

    def load_reports(self) -> dict[str, dict[str, Any]]:
        if self._path is None:
            return {}
        with self._connect() as connection:
            rows = connection.execute("SELECT run_id, report FROM engineering_evaluations").fetchall()
        return {row["run_id"]: json.loads(row["report"] or "{}") for row in rows}

    def list_runs(self) -> list[Trajectory]:
        with self._lock:
            if self._path is None:
                return [self._copy(run) for run in self._memory.values()]
            with self._connect() as connection:
                rows = connection.execute(
                    "SELECT id FROM engineering_runs ORDER BY started_at DESC"
                ).fetchall()
            return [self._copy(run) for run_id in (row["id"] for row in rows) if (run := self._lookup(run_id))]

    def _lookup(self, run_id: str) -> Trajectory | None:
        stored = self._memory.get(run_id)
        if stored is not None:
            return stored
        loaded = self._load(run_id)
        if loaded is not None:
            self._memory[run_id] = loaded
        return loaded

    def _persist_run(self, run: Trajectory) -> None:
        if self._path is None:
            return
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO engineering_runs (
                    id, agent_version, session_id, trace_id, question, status,
                    started_at, finished_at, duration_ms, token_usage, estimated_cost
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    status = excluded.status,
                    finished_at = excluded.finished_at,
                    duration_ms = excluded.duration_ms,
                    token_usage = excluded.token_usage,
                    estimated_cost = excluded.estimated_cost
                """,
                (
                    run.id,
                    run.agent_version,
                    run.session_id,
                    run.trace_id,
                    run.question,
                    run.status,
                    run.started_at,
                    run.finished_at,
                    run.duration_ms,
                    run.token_usage,
                    run.estimated_cost,
                ),
            )

    def _persist_event(self, run_id: str, event: TrajectoryEvent) -> None:
        if self._path is None:
            return
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO engineering_events
                    (run_id, seq, kind, action, ok, error, payload, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    event.seq,
                    event.kind,
                    event.action,
                    None if event.ok is None else int(event.ok),
                    event.error,
                    json.dumps(event.payload, default=str),
                    event.created_at,
                ),
            )

    def _load(self, run_id: str) -> Trajectory | None:
        if self._path is None:
            return None
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM engineering_runs WHERE id = ?",
                (run_id,),
            ).fetchone()
            if row is None:
                return None
            events = [
                TrajectoryEvent(
                    seq=int(item["seq"]),
                    kind=item["kind"],
                    action=item["action"] or "",
                    ok=None if item["ok"] is None else bool(item["ok"]),
                    error=item["error"] or "",
                    payload=json.loads(item["payload"] or "{}"),
                    created_at=item["created_at"],
                )
                for item in connection.execute(
                    "SELECT * FROM engineering_events WHERE run_id = ? ORDER BY seq",
                    (run_id,),
                ).fetchall()
            ]
        return Trajectory(
            id=row["id"],
            agent_version=row["agent_version"] or "",
            session_id=row["session_id"] or "",
            trace_id=row["trace_id"] or "",
            question=row["question"] or "",
            status=row["status"],
            started_at=row["started_at"],
            finished_at=row["finished_at"] or "",
            duration_ms=float(row["duration_ms"] or 0),
            token_usage=int(row["token_usage"] or 0),
            estimated_cost=float(row["estimated_cost"] or 0),
            events=events,
        )

    def _connect(self) -> sqlite3.Connection:
        assert self._path is not None
        connection = sqlite3.connect(self._path, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def _copy(self, run: Trajectory) -> Trajectory:
        return Trajectory(
            id=run.id,
            agent_version=run.agent_version,
            session_id=run.session_id,
            trace_id=run.trace_id,
            question=run.question,
            status=run.status,
            started_at=run.started_at,
            finished_at=run.finished_at,
            duration_ms=run.duration_ms,
            token_usage=run.token_usage,
            estimated_cost=run.estimated_cost,
            events=list(run.events),
        )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _elapsed_ms(started: str, finished: str) -> float:
    try:
        begin = datetime.fromisoformat(started)
        end = datetime.fromisoformat(finished)
    except ValueError:
        return 0.0
    if begin.tzinfo is None:
        begin = begin.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    return max(0.0, (end - begin).total_seconds() * 1000.0)
