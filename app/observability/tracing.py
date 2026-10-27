"""Traces that survive the API process, a worker, and a resume."""

import json
import sqlite3
import threading
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS research_traces (
    id TEXT PRIMARY KEY,
    question TEXT NOT NULL DEFAULT '',
    user_id TEXT NOT NULL DEFAULT '',
    session_id TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS research_spans (
    trace_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    span_id TEXT NOT NULL,
    parent_id TEXT NOT NULL DEFAULT '',
    action TEXT NOT NULL,
    payload TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (trace_id, seq)
);
CREATE INDEX IF NOT EXISTS idx_research_spans_trace ON research_spans(trace_id, seq);
"""

_SUMMARY = (
    "duration_ms",
    "cost",
    "status",
    "reliability",
    "quality_confident",
    "quality_failures",
)


class TraceLog:
    """Record nested research spans. A sqlite path is shared across processes."""

    def __init__(self, path: Path | None = None, limit: int = 200) -> None:
        self._limit = limit
        self._path = path
        self._lock = threading.Lock()
        self._traces: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._parents: dict[str, str] = {}
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as connection:
                _prepare(connection)

    def start(self, trace_id: str | None = None, **fields: Any) -> str:
        """Open a trace. Pass ``trace_id`` to continue one from another process."""
        tid = (trace_id or "").strip() or uuid.uuid4().hex[:16]
        record = {
            "id": tid,
            "steps": [],
            "question": str(fields.get("question", "") or ""),
            "user_id": str(fields.get("user_id", "") or ""),
            "session_id": str(fields.get("session_id", "") or ""),
        }
        for key in _SUMMARY:
            if key in fields and fields[key] is not None:
                record[key] = fields[key]
        with self._lock:
            existing = self._lookup(tid)
            if existing is None:
                self._traces[tid] = record
                self._traces.move_to_end(tid)
                while len(self._traces) > self._limit:
                    old, _ = self._traces.popitem(last=False)
                    self._parents.pop(old, None)
                self._persist_trace(record)
            else:
                for key in ("question", "user_id", "session_id"):
                    value = record.get(key, "")
                    if value:
                        existing[key] = value
                self._persist_trace(existing)
        return tid

    def update(self, trace_id: str, **fields: Any) -> None:
        with self._lock:
            trace = self._lookup(trace_id)
            if trace is None:
                return
            for key in ("question", "user_id", "session_id"):
                if key in fields and fields[key] is not None:
                    trace[key] = str(fields[key])
            for key in _SUMMARY:
                if key in fields and fields[key] is not None:
                    trace[key] = fields[key]
            self._persist_trace(trace)

    def step(self, trace_id: str, **fields: Any) -> str:
        """Append a span. The previous span is the parent unless one is supplied."""
        action = str(fields.pop("action", "step") or "step")
        span_id = str(fields.pop("span_id", "") or uuid.uuid4().hex[:12])
        with self._lock:
            trace = self._lookup(trace_id)
            if trace is None:
                return ""
            parent_id = str(fields.pop("parent_id", None) or self._parents.get(trace_id, ""))
            span = {"action": action, "span_id": span_id, "parent_id": parent_id, **fields}
            trace["steps"].append(span)
            self._parents[trace_id] = span_id
            self._persist_span(trace_id, len(trace["steps"]) - 1, span)
        return span_id

    def get(self, trace_id: str) -> dict[str, Any] | None:
        with self._lock:
            trace = self._lookup(trace_id)
            if trace is None:
                return None
            payload = {
                "id": trace["id"],
                "question": trace.get("question", ""),
                "user_id": trace.get("user_id", ""),
                "session_id": trace.get("session_id", ""),
                "steps": [dict(step) for step in trace["steps"]],
            }
            for key in _SUMMARY:
                if key in trace:
                    payload[key] = trace[key]
            return payload

    def _lookup(self, trace_id: str) -> dict[str, Any] | None:
        stored = self._traces.get(trace_id)
        if stored is not None:
            return stored
        loaded = self._load(trace_id)
        if loaded is not None:
            self._traces[trace_id] = loaded
            self._traces.move_to_end(trace_id)
            if loaded["steps"]:
                self._parents[trace_id] = str(loaded["steps"][-1].get("span_id", ""))
        return loaded

    def _persist_trace(self, trace: dict[str, Any]) -> None:
        if self._path is None:
            return
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO research_traces (id, question, user_id, session_id, created_at, summary)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    question = excluded.question,
                    user_id = excluded.user_id,
                    session_id = excluded.session_id,
                    summary = excluded.summary
                """,
                (
                    trace["id"],
                    trace.get("question", ""),
                    trace.get("user_id", ""),
                    trace.get("session_id", ""),
                    _now(),
                    json.dumps({key: trace[key] for key in _SUMMARY if key in trace}, default=str),
                ),
            )

    def _persist_span(self, trace_id: str, seq: int, span: dict[str, Any]) -> None:
        if self._path is None:
            return
        payload = {
            key: value
            for key, value in span.items()
            if key not in {"action", "span_id", "parent_id"}
        }
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO research_spans
                    (trace_id, seq, span_id, parent_id, action, payload)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    trace_id,
                    seq,
                    span.get("span_id", ""),
                    span.get("parent_id", ""),
                    span.get("action", "step"),
                    json.dumps(payload, default=str),
                ),
            )

    def _load(self, trace_id: str) -> dict[str, Any] | None:
        if self._path is None:
            return None
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM research_traces WHERE id = ?",
                (trace_id,),
            ).fetchone()
            if row is None:
                return None
            steps = []
            for item in connection.execute(
                """
                SELECT span_id, parent_id, action, payload
                FROM research_spans WHERE trace_id = ? ORDER BY seq
                """,
                (trace_id,),
            ).fetchall():
                payload = json.loads(item["payload"] or "{}")
                steps.append(
                    {
                        "action": item["action"],
                        "span_id": item["span_id"],
                        "parent_id": item["parent_id"],
                        **payload,
                    }
                )
        loaded = {
            "id": row["id"],
            "question": row["question"] or "",
            "user_id": row["user_id"] or "",
            "session_id": row["session_id"] or "",
            "steps": steps,
        }
        keys = row.keys()
        if "summary" in keys:
            loaded.update(json.loads(row["summary"] or "{}"))
        return loaded

    def _connect(self) -> sqlite3.Connection:
        assert self._path is not None
        connection = sqlite3.connect(self._path, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        return connection


def _prepare(connection: sqlite3.Connection) -> None:
    connection.executescript(_SCHEMA)
    names = {row["name"] for row in connection.execute("PRAGMA table_info(research_traces)")}
    if "summary" not in names:
        connection.execute("ALTER TABLE research_traces ADD COLUMN summary TEXT NOT NULL DEFAULT '{}'")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
