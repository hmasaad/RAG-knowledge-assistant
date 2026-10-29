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
from app.research.bayes import SequentialDesigner
from app.research.voi import ValueOfInformation, record_selection

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
    voi: float = 0.0
    cost: int = 0
    kind: str = ""
    candidates: tuple[dict, ...] = ()
    signals: dict = field(default_factory=dict)

    def to_payload(self) -> dict:
        return {
            "action": self.action,
            "verb": self.verb,
            "reason": self.reason,
            "targets": list(self.targets),
            "severity": self.severity,
            "quality": self.quality,
            "voi": self.voi,
            "cost": self.cost,
            "kind": self.kind,
            "candidates": list(self.candidates),
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
        remaining_budget: int = 16,
        spent_cost: int = 0,
        iterations: int = 0,
        max_iterations: int = 8,
        max_cost: int = 0,
        deadline: str = "",
    ) -> PlanDecision:
        signals = self.inspect(state)
        ranked = ValueOfInformation().candidates(
            state,
            remaining_budget=remaining_budget,
            spent_cost=spent_cost or state.cost,
            max_cost=max_cost,
            iterations=iterations or state.iterations,
            max_iterations=max_iterations,
            deadline=deadline,
        )
        designed = SequentialDesigner().plan(state, ranked, remaining_budget=remaining_budget)
        ranked = designed.ranked
        chosen = None if designed.stop else (ranked[0] if ranked else None)
        action, verb, reason, targets = self._choose(signals, pending_actions, chosen, ranked)
        if designed.stop and chosen is None and not (signals.gaps or signals.unresolved_claims or signals.contradictions):
            reason = designed.reason
        if strategy is not None:
            action = strategy.recommend(
                classify_situation(
                    gaps=signals.gaps,
                    conflicts=signals.contradictions,
                    remaining_budget=remaining_budget,
                ),
                action,
            )
            if action == "stop":
                verb, reason, targets = "remove", "Strategy stopped further research.", ()
                chosen = None
        if chosen is not None:
            record_selection(state, chosen, signals.quality)
        return PlanDecision(
            action=action,
            verb=verb,
            reason=reason,
            targets=targets,
            severity=signals.contradiction_severity,
            quality=signals.quality,
            voi=chosen.voi if chosen is not None else 0.0,
            cost=chosen.cost if chosen is not None else 0,
            kind=chosen.kind if chosen is not None else "",
            candidates=tuple(item.to_payload() for item in ranked),
            signals={
                **_signals_payload(signals),
                "design": designed.to_payload(),
            },
        )

    def _choose(
        self,
        signals: EvidenceSignals,
        pending_actions: tuple[str, ...],
        chosen,
        ranked=(),
    ) -> tuple[str, str, str, tuple[str, ...]]:
        pending = set(pending_actions)
        if chosen is None:
            mandatory_left = signals.unresolved_claims or signals.contradiction_severity in {"high", "medium"}
            if mandatory_left:
                return "stop", "remove", "Mandatory checks remain but budget, time, or tool limits block them.", ()
            if not signals.gaps and not signals.contradictions and FILL_GAPS in pending:
                return "stop", "remove", "The missing evidence has arrived; drop stale fills.", ()
            return "stop", "remove", "Evidence is sufficient to synthesize.", ()
        action = chosen.action
        targets = (chosen.target,) if chosen.target else ()
        if action == RESOLVE_CONTRADICTIONS:
            verb = "reprioritize" if RESOLVE_CONTRADICTIONS in pending else "add"
            return action, verb, chosen.reason, signals.contradictions or targets
        if action == FILL_GAPS:
            verb = "reprioritize" if FILL_GAPS in pending else "add"
            return action, verb, chosen.reason, signals.gaps or targets
        if action == REVERIFY:
            verb = "repeat" if REVERIFY in pending else "add"
            return action, verb, chosen.reason, targets or signals.unresolved_claims
        return "stop", "remove", chosen.reason, ()


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
