"""Remember which research actions paid off and reuse them on the next similar case."""

import sqlite3
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from app.generation.grounding import is_refusal
from app.research.contracts import FILL_GAPS, RESOLVE_CONTRADICTIONS, REVERIFY

GAP = "gap"
CONFLICT = "conflict"
COMPLETE = "complete"
UNSUPPORTED = "unsupported"
MIN_SAMPLES = 2
ALLOWED = frozenset({FILL_GAPS, RESOLVE_CONTRADICTIONS, REVERIFY, "stop"})

_DECISION_ACTIONS = {
    "retry": FILL_GAPS,
    "probe_conflict": RESOLVE_CONTRADICTIONS,
    FILL_GAPS: FILL_GAPS,
    RESOLVE_CONTRADICTIONS: RESOLVE_CONTRADICTIONS,
    REVERIFY: REVERIFY,
    "report": "stop",
    "stop": "stop",
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS research_strategy (
    situation TEXT NOT NULL,
    action TEXT NOT NULL,
    wins INTEGER NOT NULL DEFAULT 0,
    losses INTEGER NOT NULL DEFAULT 0,
    last_reward REAL NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (situation, action)
);
"""


@dataclass(frozen=True)
class StrategyStat:
    situation: str
    action: str
    wins: int = 0
    losses: int = 0
    last_reward: float = 0.0

    @property
    def samples(self) -> int:
        return self.wins + self.losses

    @property
    def score(self) -> float:
        return (self.wins + 1.0) / (self.samples + 2.0)

    def to_payload(self) -> dict:
        payload = asdict(self)
        payload["score"] = self.score
        payload["samples"] = self.samples
        return payload


def classify_situation(
    *,
    gaps: tuple[str, ...] | list[str] = (),
    conflicts: tuple | list = (),
    sufficient: bool = False,
) -> str:
    if conflicts:
        return CONFLICT
    if gaps:
        return GAP
    if sufficient:
        return COMPLETE
    return UNSUPPORTED


def action_from_report(report) -> str:
    quality = str(getattr(report, "quality_action", "") or "").strip()
    if quality in ALLOWED:
        return quality
    decisions = list(getattr(report, "decisions", ()) or ())
    last = str(decisions[-1] if decisions else getattr(report, "decision", "") or "").strip()
    return _DECISION_ACTIONS.get(last, "stop")


def reward_from_report(report) -> float:
    reliability = float(getattr(report, "reliability", 0.0) or 0.0)
    confident = bool(getattr(report, "quality_confident", False))
    if confident and reliability >= 1.0:
        return 1.0
    text = getattr(report, "report", "") or ""
    if confident and is_refusal(text):
        return 1.0
    if getattr(report, "conflict", False) and getattr(report, "action", "") == "research":
        return 1.0
    return 0.0


class StrategyMemory:
    """Count wins and losses for each situation and action, then pick the better one."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._rows: dict[tuple[str, str], StrategyStat] = {}
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as connection:
                connection.executescript(_SCHEMA)

    def observe(self, situation: str, action: str, reward: float) -> StrategyStat:
        key_situation = (situation or UNSUPPORTED).strip() or UNSUPPORTED
        key_action = _DECISION_ACTIONS.get((action or "stop").strip(), "stop")
        if key_action not in ALLOWED:
            key_action = "stop"
        with self._lock:
            current = self._row(key_situation, key_action)
            won = current.wins + (1 if reward >= 0.5 else 0)
            lost = current.losses + (0 if reward >= 0.5 else 1)
            updated = StrategyStat(key_situation, key_action, won, lost, float(reward))
            self._rows[(key_situation, key_action)] = updated
            self._persist(updated)
            return updated

    def learn(self, report) -> StrategyStat:
        return self.observe(
            classify_situation(
                gaps=tuple(getattr(report, "gaps", ()) or ()),
                conflicts=tuple(getattr(report, "conflicts", ()) or getattr(report, "contradictions", ()) or ()),
                sufficient=bool(getattr(report, "sufficient_evidence", False)),
            ),
            action_from_report(report),
            reward_from_report(report),
        )

    def recommend(
        self,
        situation: str,
        fallback: str,
        *,
        denied: frozenset[str] | set[str] = frozenset(),
    ) -> str:
        allowed = {item for item in ALLOWED if item not in denied}
        default = fallback if fallback in allowed else ("stop" if "stop" in allowed else fallback)
        with self._lock:
            scored: list[tuple[float, int, str]] = []
            for action in allowed:
                row = self._row(situation, action)
                if row.samples < MIN_SAMPLES:
                    continue
                scored.append((row.score, row.wins, action))
        if not scored:
            return default
        scored.sort(reverse=True)
        return scored[0][2]

    def snapshot(self) -> list[StrategyStat]:
        with self._lock:
            if not self._rows and self._path is not None:
                self._rows = self._load()
            return sorted(self._rows.values(), key=lambda item: (item.situation, item.action))

    def _row(self, situation: str, action: str) -> StrategyStat:
        key = (situation, action)
        if key in self._rows:
            return self._rows[key]
        if self._path is not None and not self._rows:
            self._rows = self._load()
        return self._rows.get(key, StrategyStat(situation, action))

    def _persist(self, row: StrategyStat) -> None:
        if self._path is None:
            return
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO research_strategy (situation, action, wins, losses, last_reward, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(situation, action) DO UPDATE SET
                    wins = excluded.wins,
                    losses = excluded.losses,
                    last_reward = excluded.last_reward,
                    updated_at = excluded.updated_at
                """,
                (row.situation, row.action, row.wins, row.losses, row.last_reward, _now()),
            )

    def _load(self) -> dict[tuple[str, str], StrategyStat]:
        if self._path is None:
            return {}
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM research_strategy").fetchall()
        return {
            (row["situation"], row["action"]): StrategyStat(
                row["situation"],
                row["action"],
                int(row["wins"]),
                int(row["losses"]),
                float(row["last_reward"]),
            )
            for row in rows
        }

    def _connect(self) -> sqlite3.Connection:
        assert self._path is not None
        connection = sqlite3.connect(self._path, timeout=5)
        connection.row_factory = sqlite3.Row
        return connection


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
