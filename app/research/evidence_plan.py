"""Decide DAG changes from evidence quality, gaps, claims, and contradictions."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.research.contracts import (
    FILL_GAPS,
    RESOLVE_CONTRADICTIONS,
    REVERIFY,
    ResearchState,
)
from app.research.strategy import StrategyMemory, classify_situation

_NUMBERS = re.compile(r"\d+(?:\.\d+)?")
_WEAK = frozenset({"insufficient", "unsupported", "contradicted"})


@dataclass(frozen=True)
class EvidenceSignals:
    """What the current evidence says about the investigation."""

    quality: float
    unresolved_claims: tuple[str, ...]
    gaps: tuple[str, ...]
    contradictions: tuple[str, ...]
    contradiction_severity: str
    supported: int
    judged: int


@dataclass(frozen=True)
class PlanDecision:
    """One evidence-driven change to the research DAG."""

    action: str
    verb: str
    reason: str
    targets: tuple[str, ...] = ()
    severity: str = "none"
    quality: float = 0.0
    signals: dict = field(default_factory=dict)

    def to_payload(self) -> dict:
        return {
            "action": self.action,
            "verb": self.verb,
            "reason": self.reason,
            "targets": list(self.targets),
            "severity": self.severity,
            "quality": self.quality,
            "signals": dict(self.signals),
        }


class EvidenceDrivenPlanner:
    """Turn evidence quality into add, remove, reprioritize, or repeat."""

    def inspect(self, state: ResearchState) -> EvidenceSignals:
        judged = tuple(state.verdicts)
        supported = tuple(item for item in judged if item.verdict == "supported")
        unresolved = tuple(
            item.claim.strip()
            for item in judged
            if item.verdict in _WEAK and item.claim.strip()
        )
        gaps = tuple(gap for gap in state.gaps if gap.strip())
        contradictions = tuple(item for item in state.contradictions if item.strip())
        quality = (len(supported) / len(judged)) if judged else (0.5 if state.evidence else 0.0)
        return EvidenceSignals(
            quality=quality,
            unresolved_claims=unresolved,
            gaps=gaps,
            contradictions=contradictions,
            contradiction_severity=_severity(contradictions),
            supported=len(supported),
            judged=len(judged),
        )

    def decide(
        self,
        state: ResearchState,
        *,
        pending_actions: tuple[str, ...] = (),
        strategy: StrategyMemory | None = None,
    ) -> PlanDecision:
        signals = self.inspect(state)
        action, verb, reason, targets = self._choose(signals, pending_actions)
        if strategy is not None:
            action = strategy.recommend(
                classify_situation(gaps=signals.gaps, conflicts=signals.contradictions),
                action,
            )
            if action == "stop":
                verb, reason, targets = "remove", "Strategy stopped further research.", ()
        return PlanDecision(
            action=action,
            verb=verb,
            reason=reason,
            targets=targets,
            severity=signals.contradiction_severity,
            quality=signals.quality,
            signals=_signals_payload(signals),
        )

    def _choose(
        self,
        signals: EvidenceSignals,
        pending_actions: tuple[str, ...],
    ) -> tuple[str, str, str, tuple[str, ...]]:
        pending = set(pending_actions)
        if signals.contradiction_severity in {"high", "medium"}:
            verb = "reprioritize" if RESOLVE_CONTRADICTIONS in pending else "add"
            return (
                RESOLVE_CONTRADICTIONS,
                verb,
                f"{signals.contradiction_severity.capitalize()} contradictions need a resolve step.",
                signals.contradictions,
            )
        if signals.gaps:
            verb = "reprioritize" if FILL_GAPS in pending else "add"
            return (
                FILL_GAPS,
                verb,
                "Information gaps remain; the DAG adds or keeps a fill.",
                signals.gaps,
            )
        if signals.unresolved_claims:
            claim = signals.unresolved_claims[0]
            verb = "repeat" if REVERIFY in pending else "add"
            return (
                REVERIFY,
                verb,
                "An unresolved claim needs another verification.",
                (claim,),
            )
        if signals.judged and signals.quality < 0.5:
            return (
                REVERIFY,
                "repeat",
                "Evidence quality is weak; a claim should be verified again.",
                signals.unresolved_claims,
            )
        if not signals.gaps and not signals.contradictions and FILL_GAPS in pending:
            return "stop", "remove", "The missing evidence has arrived; drop stale fills.", ()
        return "stop", "remove", "Evidence is sufficient to synthesize.", ()


def _severity(contradictions: tuple[str, ...]) -> str:
    if not contradictions:
        return "none"
    numeric = 0
    for item in contradictions:
        values = set(_NUMBERS.findall(item))
        files = item.count(".md") + item.count(".txt") + item.count(".pdf")
        if len(values) >= 2 or (files >= 2 and values):
            numeric += 1
    if numeric:
        return "high"
    return "medium" if len(contradictions) > 1 else "low"


def _signals_payload(signals: EvidenceSignals) -> dict:
    return {
        "quality": signals.quality,
        "unresolved_claims": list(signals.unresolved_claims),
        "gaps": list(signals.gaps),
        "contradictions": list(signals.contradictions),
        "contradiction_severity": signals.contradiction_severity,
        "supported": signals.supported,
        "judged": signals.judged,
    }
