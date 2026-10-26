"""Persist research cycle state so a session can resume."""

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.research.contracts import ActionOutcome, ActionRequest, ResearchState

_SCHEMA = """
CREATE TABLE IF NOT EXISTS research_sessions (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    question TEXT NOT NULL,
    top_k INTEGER NOT NULL,
    status TEXT NOT NULL,
    action TEXT NOT NULL DEFAULT '',
    report TEXT NOT NULL DEFAULT '',
    sufficient_evidence INTEGER NOT NULL DEFAULT 0,
    steps TEXT NOT NULL DEFAULT '[]',
    gaps TEXT NOT NULL DEFAULT '[]',
    contradictions TEXT NOT NULL DEFAULT '[]',
    cost INTEGER NOT NULL DEFAULT 0,
    state_json TEXT NOT NULL DEFAULT '{}',
    result_json TEXT NOT NULL DEFAULT '{}',
    error TEXT,
    attempts INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    trace_id TEXT NOT NULL DEFAULT '',
    lease_until TEXT,
    lease_owner TEXT NOT NULL DEFAULT '',
    priority INTEGER NOT NULL DEFAULT 0,
    cancel_requested INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_research_sessions_user
    ON research_sessions(user_id, updated_at);
CREATE INDEX IF NOT EXISTS idx_research_sessions_status
    ON research_sessions(status, created_at);
CREATE TABLE IF NOT EXISTS research_action_log (
    session_id TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    action TEXT NOT NULL,
    query TEXT NOT NULL,
    claim TEXT NOT NULL,
    outcome_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (session_id, fingerprint)
);
CREATE TABLE IF NOT EXISTS research_side_effects (
    session_id TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    action TEXT NOT NULL,
    runs INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (session_id, fingerprint)
);
"""


@dataclass(frozen=True)
class ResearchSession:
    id: str
    user_id: str
    question: str
    top_k: int
    status: str
    action: str = ""
    report: str = ""
    sufficient_evidence: bool = False
    steps: tuple[str, ...] = ()
    gaps: tuple[str, ...] = ()
    contradictions: tuple[str, ...] = ()
    cost: int = 0
    state: ResearchState | None = None
    result: dict | None = None
    error: str | None = None
    attempts: int = 0
    created_at: str = ""
    updated_at: str = ""
    started_at: str | None = None
    finished_at: str | None = None
    trace_id: str = ""
    lease_until: str | None = None
    lease_owner: str = ""
    priority: int = 0
    cancel_requested: bool = False


class ResearchSessionStore:
    """SQLite records for research cycles and their resume state."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(_SCHEMA)
            _ensure_session_columns(connection)

    def create(
        self,
        *,
        user_id: str,
        question: str,
        top_k: int,
        trace_id: str = "",
        status: str = "queued",
        priority: int = 0,
    ) -> ResearchSession:
        session_id = uuid.uuid4().hex
        now = _now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO research_sessions (
                    id, user_id, question, top_k, status, trace_id,
                    created_at, updated_at, priority
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (session_id, user_id, question, top_k, status, trace_id, now, now, max(0, priority)),
            )
        stored = self.get(session_id)
        if stored is None:
            raise RuntimeError("Research session was not stored.")
        return stored

    def get(self, session_id: str) -> ResearchSession | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM research_sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
        return _session(row) if row else None

    def list_for_user(self, user_id: str) -> list[ResearchSession]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM research_sessions
                WHERE user_id = ?
                ORDER BY updated_at DESC, id DESC
                """,
                (user_id,),
            ).fetchall()
        return [_session(row) for row in rows]

    def save_progress(self, session_id: str, state: ResearchState) -> None:
        self.persist_step(session_id, state)

    def persist_step(
        self,
        session_id: str,
        state: ResearchState,
        request: ActionRequest | None = None,
        outcome: ActionOutcome | None = None,
    ) -> None:
        """Write state and the action log in one transaction."""
        now = _now()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE research_sessions
                SET state_json = ?, cost = ?, steps = ?, gaps = ?,
                    contradictions = ?, report = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    json.dumps(state.to_payload()),
                    state.cost,
                    json.dumps(list(state.steps)),
                    json.dumps(list(state.gaps)),
                    json.dumps(list(state.contradictions)),
                    state.report,
                    now,
                    session_id,
                ),
            )
            if request is None or outcome is None:
                return
            self._bump_effect(connection, session_id, request)
            connection.execute(
                """
                INSERT OR IGNORE INTO research_action_log (
                    session_id, fingerprint, action, query, claim, outcome_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    fingerprint_key(request),
                    request.action,
                    request.query,
                    request.claim,
                    json.dumps(_outcome_payload(outcome)),
                    now,
                ),
            )

    def has_action(self, session_id: str, request: ActionRequest) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT 1 AS found FROM research_action_log
                WHERE session_id = ? AND fingerprint = ?
                """,
                (session_id, fingerprint_key(request)),
            ).fetchone()
        return row is not None

    def heartbeat(self, session_id: str, *, owner: str = "", seconds: float = 900) -> None:
        until = _now_plus(seconds)
        with self._connect() as connection:
            if owner:
                connection.execute(
                    """
                    UPDATE research_sessions
                    SET lease_until = ?, updated_at = ?
                    WHERE id = ? AND (lease_owner = ? OR lease_owner = '')
                    """,
                    (until, _now(), session_id, owner),
                )
                return
            connection.execute(
                """
                UPDATE research_sessions
                SET lease_until = ?, updated_at = ?
                WHERE id = ?
                """,
                (until, _now(), session_id),
            )

    def halt(self, session_id: str, result: dict, state: ResearchState) -> None:
        now = _now()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE research_sessions
                SET status = 'halted', action = ?, report = ?,
                    sufficient_evidence = ?, steps = ?, gaps = ?,
                    contradictions = ?, cost = ?, state_json = ?,
                    result_json = ?, error = 'Halted after a committed checkpoint.',
                    finished_at = NULL, updated_at = ?, lease_until = NULL,
                    lease_owner = ''
                WHERE id = ?
                """,
                (
                    str(result.get("action", "halt")),
                    str(result.get("report", state.report)),
                    1 if result.get("sufficient_evidence") else 0,
                    json.dumps(list(result.get("steps", state.steps))),
                    json.dumps(list(result.get("gaps", state.gaps))),
                    json.dumps(list(result.get("contradictions", state.contradictions))),
                    int(result.get("cost", state.cost) or 0),
                    json.dumps(state.to_payload()),
                    json.dumps(result),
                    now,
                    session_id,
                ),
            )

    def list_effects(self, session_id: str) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT action, fingerprint, runs FROM research_side_effects
                WHERE session_id = ?
                ORDER BY action, fingerprint
                """,
                (session_id,),
            ).fetchall()
        return [
            {"action": row["action"], "fingerprint": row["fingerprint"], "runs": int(row["runs"])}
            for row in rows
        ]

    def complete(self, session_id: str, result: dict, state: ResearchState) -> None:
        now = _now()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE research_sessions
                SET status = 'completed', action = ?, report = ?,
                    sufficient_evidence = ?, steps = ?, gaps = ?,
                    contradictions = ?, cost = ?, state_json = ?,
                    result_json = ?, error = NULL, finished_at = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    str(result.get("action", "")),
                    str(result.get("report", state.report)),
                    1 if result.get("sufficient_evidence") else 0,
                    json.dumps(list(result.get("steps", state.steps))),
                    json.dumps(list(result.get("gaps", state.gaps))),
                    json.dumps(list(result.get("contradictions", state.contradictions))),
                    int(result.get("cost", state.cost) or 0),
                    json.dumps(state.to_payload()),
                    json.dumps(result),
                    now,
                    now,
                    session_id,
                ),
            )

    def fail(self, session_id: str, error: str, state: ResearchState | None = None) -> None:
        now = _now()
        payload = json.dumps(state.to_payload()) if state is not None else None
        with self._connect() as connection:
            if payload is None:
                connection.execute(
                    """
                    UPDATE research_sessions
                    SET status = 'failed', error = ?, finished_at = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (error[:1000], now, now, session_id),
                )
                return
            connection.execute(
                """
                UPDATE research_sessions
                SET status = 'failed', error = ?, state_json = ?, cost = ?,
                    steps = ?, gaps = ?, contradictions = ?, report = ?,
                    finished_at = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    error[:1000],
                    payload,
                    state.cost,
                    json.dumps(list(state.steps)),
                    json.dumps(list(state.gaps)),
                    json.dumps(list(state.contradictions)),
                    state.report,
                    now,
                    now,
                    session_id,
                ),
            )

    def mark_running(self, session_id: str, *, owner: str = "worker", lease_seconds: float = 900) -> None:
        now = _now()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE research_sessions
                SET status = 'running', started_at = ?, attempts = attempts + 1,
                    updated_at = ?, lease_until = ?, lease_owner = ?
                WHERE id = ?
                """,
                (now, now, _now_plus(lease_seconds), owner, session_id),
            )

    def requeue(self, session_id: str) -> ResearchSession | None:
        now = _now()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE research_sessions
                SET status = 'queued', error = NULL, finished_at = NULL, updated_at = ?,
                    lease_until = NULL, lease_owner = ''
                WHERE id = ? AND status IN ('failed', 'queued', 'running', 'halted', 'cancelled')
                """,
                (now, session_id),
            )
        return self.get(session_id)

    def requeue_expired(self, stale_after_seconds: float) -> int:
        """Return running sessions whose lease expired to the queue."""
        stale_before = _now_minus(stale_after_seconds)
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE research_sessions
                SET status = 'queued', error = 'Requeued after the worker stopped.',
                    updated_at = ?, lease_until = NULL, lease_owner = ''
                WHERE status = 'running'
                  AND (
                        (lease_until IS NOT NULL AND lease_until < ?)
                        OR (
                            (lease_until IS NULL OR lease_until = '')
                            AND started_at IS NOT NULL
                            AND started_at < ?
                        )
                  )
                """,
                (_now(), stale_before, stale_before),
            )
            return int(cursor.rowcount or 0)

    def cancel(self, session_id: str) -> ResearchSession | None:
        """Stop a queued job now, or ask a running job to stop at the next step."""
        now = _now()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE research_sessions
                SET cancel_requested = 1,
                    status = CASE WHEN status = 'queued' THEN 'cancelled' ELSE status END,
                    error = CASE WHEN status = 'queued' THEN 'Cancelled.' ELSE error END,
                    finished_at = CASE WHEN status = 'queued' THEN ? ELSE finished_at END,
                    updated_at = ?,
                    lease_until = CASE WHEN status = 'queued' THEN NULL ELSE lease_until END,
                    lease_owner = CASE WHEN status = 'queued' THEN '' ELSE lease_owner END
                WHERE id = ? AND status IN ('queued', 'running', 'halted')
                """,
                (now, now, session_id),
            )
        return self.get(session_id)

    def cancel_requested(self, session_id: str) -> bool:
        session = self.get(session_id)
        return bool(session is not None and session.cancel_requested)

    def finish_cancelled(self, session_id: str, result: dict, state: ResearchState) -> None:
        now = _now()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE research_sessions
                SET status = 'cancelled', action = ?, report = ?,
                    sufficient_evidence = ?, steps = ?, gaps = ?,
                    contradictions = ?, cost = ?, state_json = ?,
                    result_json = ?, error = 'Cancelled.',
                    finished_at = ?, updated_at = ?, lease_until = NULL,
                    lease_owner = '', cancel_requested = 1
                WHERE id = ?
                """,
                (
                    str(result.get("action", "stop")),
                    str(result.get("report", state.report)),
                    1 if result.get("sufficient_evidence") else 0,
                    json.dumps(list(result.get("steps", state.steps))),
                    json.dumps(list(result.get("gaps", state.gaps))),
                    json.dumps(list(result.get("contradictions", state.contradictions))),
                    int(result.get("cost", state.cost) or 0),
                    json.dumps(state.to_payload()),
                    json.dumps(result),
                    now,
                    now,
                    session_id,
                ),
            )

    def claim(
        self,
        stale_after_seconds: float,
        *,
        owner: str = "worker",
        user_concurrency: int = 8,
    ) -> ResearchSession | None:
        """Take the highest-priority queued cycle the caller may still run."""
        self.requeue_expired(stale_after_seconds)
        connection = sqlite3.connect(self.path, timeout=5, isolation_level=None)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT * FROM research_sessions
                WHERE status = 'queued'
                  AND cancel_requested = 0
                  AND user_id NOT IN (
                    SELECT user_id FROM research_sessions
                    WHERE status = 'running'
                    GROUP BY user_id
                    HAVING COUNT(*) >= ?
                  )
                ORDER BY priority DESC, created_at, id
                LIMIT 1
                """,
                (max(1, user_concurrency),),
            ).fetchone()
            if row is None:
                connection.execute("COMMIT")
                return None
            started = _now()
            cursor = connection.execute(
                """
                UPDATE research_sessions
                SET status = 'running', started_at = ?, attempts = attempts + 1,
                    updated_at = ?, lease_until = ?, lease_owner = ?
                WHERE id = ? AND status = 'queued' AND cancel_requested = 0
                """,
                (started, started, _now_plus(stale_after_seconds), owner, row["id"]),
            )
            claimed = cursor.rowcount == 1
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()
        if not claimed:
            return None
        stored = self.get(row["id"])
        return stored

    def job_counts(self) -> dict[str, int]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT status, COUNT(*) AS count FROM research_sessions GROUP BY status"
            ).fetchall()
        return {row["status"]: int(row["count"]) for row in rows}

    def _bump_effect(self, connection: sqlite3.Connection, session_id: str, request: ActionRequest) -> None:
        connection.execute(
            """
            INSERT INTO research_side_effects (session_id, fingerprint, action, runs)
            VALUES (?, ?, ?, 1)
            ON CONFLICT(session_id, fingerprint)
            DO UPDATE SET runs = runs + 1
            """,
            (session_id, fingerprint_key(request), request.action),
        )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5)
        connection.row_factory = sqlite3.Row
        return connection


def _session(row: sqlite3.Row) -> ResearchSession:
    state_payload = _load_json(row["state_json"], {})
    result_payload = _load_json(row["result_json"], {})
    steps = tuple(_load_json(row["steps"], []))
    gaps = tuple(_load_json(row["gaps"], []))
    contradictions = tuple(_load_json(row["contradictions"], []))
    return ResearchSession(
        id=row["id"],
        user_id=row["user_id"],
        question=row["question"],
        top_k=int(row["top_k"]),
        status=row["status"],
        action=row["action"] or "",
        report=row["report"] or "",
        sufficient_evidence=bool(row["sufficient_evidence"]),
        steps=steps,
        gaps=gaps,
        contradictions=contradictions,
        cost=int(row["cost"] or 0),
        state=ResearchState.from_payload(state_payload),
        result=result_payload or None,
        error=row["error"],
        attempts=int(row["attempts"] or 0),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        trace_id=row["trace_id"] or "",
        lease_until=row["lease_until"] if "lease_until" in row.keys() else None,
        lease_owner=row["lease_owner"] if "lease_owner" in row.keys() else "",
        priority=int(row["priority"]) if "priority" in row.keys() else 0,
        cancel_requested=bool(row["cancel_requested"]) if "cancel_requested" in row.keys() else False,
    )


def _load_json(raw: str | None, default):
    if not raw:
        return default
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return default


def fingerprint_key(request: ActionRequest) -> str:
    return json.dumps(
        [request.action, request.query.strip().casefold(), request.claim.strip().casefold()],
        separators=(",", ":"),
    )


def _outcome_payload(outcome: ActionOutcome) -> dict:
    return {
        "action": outcome.action,
        "ok": outcome.ok,
        "error": outcome.error,
        "evidence": [
            {
                "evidence_id": item.evidence_id,
                "document_id": item.document_id,
                "source": item.source,
                "text": item.text,
            }
            for item in outcome.evidence
        ],
        "verdicts": [
            {
                "claim": item.claim,
                "verdict": item.verdict,
                "evidence_ids": list(item.evidence_ids),
            }
            for item in outcome.verdicts
        ],
        "gaps": list(outcome.gaps),
        "contradictions": list(outcome.contradictions),
        "provenance": list(outcome.provenance),
        "report": outcome.report,
        "cost": outcome.cost,
    }


def _ensure_session_columns(connection: sqlite3.Connection) -> None:
    columns = {row[1] for row in connection.execute("PRAGMA table_info(research_sessions)")}
    if "lease_until" not in columns:
        connection.execute("ALTER TABLE research_sessions ADD COLUMN lease_until TEXT")
    if "lease_owner" not in columns:
        connection.execute(
            "ALTER TABLE research_sessions ADD COLUMN lease_owner TEXT NOT NULL DEFAULT ''"
        )
    if "priority" not in columns:
        connection.execute(
            "ALTER TABLE research_sessions ADD COLUMN priority INTEGER NOT NULL DEFAULT 0"
        )
    if "cancel_requested" not in columns:
        connection.execute(
            "ALTER TABLE research_sessions ADD COLUMN cancel_requested INTEGER NOT NULL DEFAULT 0"
        )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _now_plus(seconds: float) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=seconds)).isoformat()


def _now_minus(seconds: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(seconds=seconds)).isoformat()
