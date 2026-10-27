"""Production readiness checks, operator-approved releases, and rollback."""

import json
import sqlite3
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from app.engineering.loop import LOOP_MANDATORY, ContinuousImprovementLoop, LoopSnapshot
from app.ingestion.errors import KnowledgeError

_SCHEMA = """
CREATE TABLE IF NOT EXISTS engineering_releases (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    planner_version TEXT NOT NULL,
    metrics TEXT NOT NULL,
    actor TEXT NOT NULL,
    recorded_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS engineering_audit (
    id TEXT PRIMARY KEY,
    action TEXT NOT NULL,
    planner_version TEXT NOT NULL,
    actor TEXT NOT NULL,
    allowed INTEGER NOT NULL,
    detail TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    metrics TEXT NOT NULL DEFAULT '{}'
);
"""


@dataclass(frozen=True)
class ReadinessProbes:
    metadata: Callable[[], object] = lambda: None
    vectors: Callable[[], object] = lambda: None
    guardrails: Callable[[], bool] = lambda: True


@dataclass(frozen=True)
class GovernancePolicy:
    hold: bool = False
    require_approval: bool = True
    operators: frozenset[str] = frozenset()
    floors: dict[str, float] = field(default_factory=lambda: {name: 1.0 for name in LOOP_MANDATORY})


@dataclass(frozen=True)
class ReadinessCheck:
    name: str
    passed: bool
    detail: str = ""

    def to_payload(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ReadinessReport:
    ready: bool
    production_ready: bool
    serving: str = ""
    hold: bool = False
    checks: tuple[ReadinessCheck, ...] = ()

    def to_payload(self) -> dict:
        return {
            "ready": self.ready,
            "production_ready": self.production_ready,
            "serving": self.serving,
            "hold": self.hold,
            "checks": [item.to_payload() for item in self.checks],
        }


@dataclass(frozen=True)
class AuditEvent:
    id: str
    action: str
    planner_version: str
    actor: str
    allowed: bool
    detail: str
    recorded_at: str
    metrics: dict[str, float] = field(default_factory=dict)

    def to_payload(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class GovernanceDecision:
    allowed: bool
    ready: bool
    promoted: bool
    planner_version: str
    actor: str
    checks: tuple[ReadinessCheck, ...]
    regressions: tuple[str, ...] = ()
    detail: str = ""
    audit_id: str = ""
    serving: str = ""

    def to_payload(self) -> dict:
        payload = asdict(self)
        payload["checks"] = [item.to_payload() for item in self.checks]
        payload["regressions"] = list(self.regressions)
        return payload


class ProductionGovernance:
    """Gate a planner release on readiness, floors, and an approved operator."""

    def __init__(
        self,
        path: Path | None = None,
        loop: ContinuousImprovementLoop | None = None,
        *,
        policy: GovernancePolicy | None = None,
        probes: ReadinessProbes | None = None,
    ) -> None:
        self._path = path
        self.loop = loop if loop is not None else ContinuousImprovementLoop(path)
        self.policy = policy or GovernancePolicy()
        self.probes = probes or ReadinessProbes()
        self._releases: list[LoopSnapshot] = []
        self._memory_audit: list[AuditEvent] = []
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as connection:
                connection.executescript(_SCHEMA)

    def readiness(self, metrics: dict[str, float] | None = None) -> ReadinessReport:
        checks = self._checks(metrics)
        infra = all(item.passed for item in checks if item.name in {"metadata", "vectors", "guardrails"})
        production = infra and all(item.passed for item in checks)
        baseline = self.loop.baseline()
        return ReadinessReport(
            ready=infra,
            production_ready=production,
            serving=baseline.planner_version if baseline is not None else "",
            hold=self.policy.hold,
            checks=checks,
        )

    def release(self, planner_version: str, metrics: dict[str, float], *, actor: str) -> GovernanceDecision:
        self._authorize(actor)
        report = self.readiness(metrics)
        blocked = [item.name for item in report.checks if not item.passed]
        if blocked:
            return self._decide(
                action="release",
                planner_version=planner_version,
                metrics=metrics,
                actor=actor,
                allowed=False,
                ready=report.ready,
                promoted=False,
                checks=report.checks,
                regressions=tuple(blocked),
                detail="Release blocked: " + ", ".join(blocked) + ".",
            )
        snapshot = self.loop.promote(planner_version, metrics)
        if not snapshot.allowed:
            return self._decide(
                action="release",
                planner_version=planner_version,
                metrics=metrics,
                actor=actor,
                allowed=False,
                ready=report.ready,
                promoted=False,
                checks=report.checks,
                regressions=snapshot.regressions,
                detail="Mandatory loop checks regressed: " + ", ".join(snapshot.regressions) + ".",
            )
        self._push_release(snapshot, actor)
        return self._decide(
            action="release",
            planner_version=planner_version,
            metrics=metrics,
            actor=actor,
            allowed=True,
            ready=True,
            promoted=True,
            checks=report.checks,
            detail="Planner revision released.",
            serving=snapshot.planner_version,
        )

    def rollback(self, *, actor: str) -> GovernanceDecision:
        self._authorize(actor)
        report = self.readiness()
        previous = self._previous_release()
        if previous is None:
            return self._decide(
                action="rollback",
                planner_version=report.serving,
                metrics={},
                actor=actor,
                allowed=False,
                ready=report.ready,
                promoted=False,
                checks=report.checks,
                detail="No previous release to restore.",
            )
        restored = self.loop.restore_baseline(previous)
        self._pop_release()
        return self._decide(
            action="rollback",
            planner_version=restored.planner_version,
            metrics=restored.metrics,
            actor=actor,
            allowed=True,
            ready=True,
            promoted=True,
            checks=report.checks,
            detail="Previous planner revision restored.",
            serving=restored.planner_version,
        )

    def audit(self) -> list[AuditEvent]:
        if self._path is None:
            return list(self._memory_audit)
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM engineering_audit ORDER BY recorded_at").fetchall()
        return [_audit(row) for row in rows]

    def _checks(self, metrics: dict[str, float] | None) -> tuple[ReadinessCheck, ...]:
        return (
            _probe("metadata", self.probes.metadata),
            _probe("vectors", self.probes.vectors),
            ReadinessCheck("guardrails", bool(self.probes.guardrails()), "ok" if self.probes.guardrails() else "off"),
            ReadinessCheck(
                "hold",
                not self.policy.hold,
                "releases are paused" if self.policy.hold else "ok",
            ),
            self._floor_check(metrics),
        )

    def _floor_check(self, metrics: dict[str, float] | None) -> ReadinessCheck:
        scores = metrics if metrics is not None else (self.loop.baseline().metrics if self.loop.baseline() else None)
        if scores is None:
            return ReadinessCheck("loop_floors", False, "no production baseline")
        missing = [
            name
            for name, floor in self.policy.floors.items()
            if float(scores.get(name, 0.0) or 0.0) + 1e-9 < float(floor)
        ]
        if missing:
            return ReadinessCheck("loop_floors", False, ", ".join(missing))
        return ReadinessCheck("loop_floors", True, "ok")

    def _authorize(self, actor: str) -> None:
        name = (actor or "").strip()
        if self.policy.require_approval and not name:
            raise KnowledgeError("Caller is not permitted.", status_code=403)
        if self.policy.operators and name not in self.policy.operators:
            raise KnowledgeError("Caller is not permitted.", status_code=403)

    def _decide(
        self,
        *,
        action: str,
        planner_version: str,
        metrics: dict[str, float],
        actor: str,
        allowed: bool,
        ready: bool,
        promoted: bool,
        checks: tuple[ReadinessCheck, ...],
        regressions: tuple[str, ...] = (),
        detail: str,
        serving: str = "",
    ) -> GovernanceDecision:
        event = AuditEvent(
            id=uuid.uuid4().hex,
            action=action,
            planner_version=planner_version,
            actor=actor,
            allowed=allowed,
            detail=detail,
            recorded_at=_now(),
            metrics={key: float(value) for key, value in metrics.items()},
        )
        self._write_audit(event)
        baseline = self.loop.baseline()
        return GovernanceDecision(
            allowed=allowed,
            ready=ready,
            promoted=promoted,
            planner_version=planner_version,
            actor=actor,
            checks=checks,
            regressions=regressions,
            detail=detail,
            audit_id=event.id,
            serving=serving or (baseline.planner_version if baseline is not None else ""),
        )

    def _push_release(self, snapshot: LoopSnapshot, actor: str) -> None:
        if self._path is None:
            self._releases.append(snapshot)
            return
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO engineering_releases (planner_version, metrics, actor, recorded_at)
                VALUES (?, ?, ?, ?)
                """,
                (snapshot.planner_version, json.dumps(snapshot.metrics), actor, snapshot.recorded_at),
            )

    def _previous_release(self) -> LoopSnapshot | None:
        releases = self._load_releases()
        if len(releases) < 2:
            return None
        return releases[-2]

    def _pop_release(self) -> None:
        if self._path is None:
            if self._releases:
                self._releases.pop()
            return
        with self._connect() as connection:
            row = connection.execute("SELECT seq FROM engineering_releases ORDER BY seq DESC LIMIT 1").fetchone()
            if row is not None:
                connection.execute("DELETE FROM engineering_releases WHERE seq = ?", (row["seq"],))

    def _load_releases(self) -> list[LoopSnapshot]:
        if self._path is None:
            return list(self._releases)
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM engineering_releases ORDER BY seq").fetchall()
        return [
            LoopSnapshot(
                planner_version=row["planner_version"],
                metrics=json.loads(row["metrics"] or "{}"),
                allowed=True,
                promoted=True,
                recorded_at=row["recorded_at"],
            )
            for row in rows
        ]

    def _write_audit(self, event: AuditEvent) -> None:
        if self._path is None:
            self._memory_audit.append(event)
            return
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO engineering_audit
                    (id, action, planner_version, actor, allowed, detail, recorded_at, metrics)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.id,
                    event.action,
                    event.planner_version,
                    event.actor,
                    int(event.allowed),
                    event.detail,
                    event.recorded_at,
                    json.dumps(event.metrics),
                ),
            )

    def _connect(self) -> sqlite3.Connection:
        assert self._path is not None
        connection = sqlite3.connect(self._path, timeout=5)
        connection.row_factory = sqlite3.Row
        return connection


def _probe(name: str, callback: Callable[[], object]) -> ReadinessCheck:
    try:
        callback()
    except Exception as exc:
        return ReadinessCheck(name, False, type(exc).__name__)
    return ReadinessCheck(name, True, "ok")


def _audit(row: sqlite3.Row) -> AuditEvent:
    return AuditEvent(
        id=row["id"],
        action=row["action"],
        planner_version=row["planner_version"],
        actor=row["actor"],
        allowed=bool(row["allowed"]),
        detail=row["detail"],
        recorded_at=row["recorded_at"],
        metrics=json.loads(row["metrics"] or "{}"),
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
