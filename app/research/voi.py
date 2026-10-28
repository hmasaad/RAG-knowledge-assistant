"""Estimate how much a research action would improve the final decision."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from app.research.contracts import (
    FILL_GAPS,
    RESOLVE_CONTRADICTIONS,
    REVERIFY,
    ResearchState,
)

MIN_VOI = 0.1
MIN_CONFIDENCE = 0.4
_COST = {RESOLVE_CONTRADICTIONS: 1, FILL_GAPS: 1, REVERIFY: 1}
_SEVERITY_GAIN = {"high": 0.85, "medium": 0.55, "low": 0.3, "none": 0.0}
_KIND = {
    RESOLVE_CONTRADICTIONS: "document",
    FILL_GAPS: "search",
    REVERIFY: "verification",
}
_MANDATORY = {RESOLVE_CONTRADICTIONS, REVERIFY}


@dataclass(frozen=True)
class BudgetWindow:
    """Spending, time, and tool-call limits for one selection."""

    remaining_budget: int = 16
    spent_cost: int = 0
    max_cost: int = 0
    iterations: int = 0
    max_iterations: int = 8
    deadline: str = ""
    now: str = ""


@dataclass(frozen=True)
class ValueCandidate:
    """Expected decision gain of one action, divided by its budget cost."""

    action: str
    kind: str
    target: str
    gain: float
    cost: int
    voi: float
    reason: str
    calibrated_voi: float = 0.0
    confidence: float = 1.0
    reliability: float = 1.0
    mandatory: bool = False
    blocked_by: str = ""
    eig: float = 0.0

    def to_payload(self) -> dict:
        return {
            "action": self.action,
            "kind": self.kind,
            "target": self.target,
            "gain": self.gain,
            "cost": self.cost,
            "voi": self.voi,
            "calibrated_voi": self.calibrated_voi,
            "confidence": self.confidence,
            "reliability": self.reliability,
            "mandatory": self.mandatory,
            "blocked_by": self.blocked_by,
            "eig": self.eig,
            "reason": self.reason,
        }


class ValueOfInformation:
    """Rank search, document, and verification work by value under a budget."""

    def __init__(self, min_voi: float = MIN_VOI, min_confidence: float = MIN_CONFIDENCE) -> None:
        self.min_voi = min_voi
        self.min_confidence = min_confidence

    def candidates(
        self,
        state: ResearchState,
        *,
        remaining_budget: int = 16,
        spent_cost: int = 0,
        max_cost: int = 0,
        iterations: int = 0,
        max_iterations: int = 8,
        deadline: str = "",
        now: str = "",
    ) -> list[ValueCandidate]:
        from app.research.evidence_plan import EvidenceDrivenPlanner

        signals = EvidenceDrivenPlanner().inspect(state)
        window = BudgetWindow(
            remaining_budget=max(0, remaining_budget),
            spent_cost=max(0, spent_cost or state.cost),
            max_cost=max_cost,
            iterations=max(0, iterations or state.iterations),
            max_iterations=max_iterations,
            deadline=deadline,
            now=now or _now(),
        )
        reliability = source_reliability(state)
        confidence = estimate_confidence(signals.judged, reliability)
        rows: list[ValueCandidate] = []
        if signals.contradictions:
            rows.append(
                self._score(
                    RESOLVE_CONTRADICTIONS,
                    signals.contradictions[0],
                    _SEVERITY_GAIN[signals.contradiction_severity],
                    f"{signals.contradiction_severity.capitalize()} contradictions could change the answer.",
                    window,
                    reliability=reliability,
                    confidence=confidence,
                    mandatory=signals.contradiction_severity in {"high", "medium"},
                )
            )
        uncertainty = max(0.0, 1.0 - signals.quality)
        for gap in signals.gaps:
            rows.append(
                self._score(
                    FILL_GAPS,
                    gap,
                    max(0.35, 0.7 * uncertainty) * reliability,
                    "A targeted search could close an information gap.",
                    window,
                    reliability=reliability,
                    confidence=confidence,
                    mandatory=False,
                )
            )
        for claim in signals.unresolved_claims:
            covered = _gap_covers(claim, signals.gaps)
            rows.append(
                self._score(
                    REVERIFY,
                    claim,
                    max(0.3, 0.5 * uncertainty),
                    "Another verification could settle an unresolved claim.",
                    window,
                    reliability=reliability,
                    confidence=confidence,
                    mandatory=not covered,
                )
            )
        runnable = [item for item in rows if _runnable(item, self.min_voi, self.min_confidence)]
        order = {RESOLVE_CONTRADICTIONS: 0, FILL_GAPS: 1, REVERIFY: 2}
        runnable.sort(
            key=lambda item: (
                not item.mandatory,
                order.get(item.action, 3),
                -item.calibrated_voi,
                -item.gain,
            )
        )
        return runnable

    def select(
        self,
        state: ResearchState,
        *,
        remaining_budget: int = 16,
        **limits,
    ) -> ValueCandidate | None:
        ranked = self.candidates(state, remaining_budget=remaining_budget, **limits)
        return ranked[0] if ranked else None

    def _score(
        self,
        action: str,
        target: str,
        gain: float,
        reason: str,
        window: BudgetWindow,
        *,
        reliability: float,
        confidence: float,
        mandatory: bool,
    ) -> ValueCandidate:
        cost = _COST[action]
        blocked = _blocked(cost, window)
        usable = 0.0 if blocked else max(0.0, gain)
        denom = float(cost) if cost > 0 else 0.25
        raw = usable / denom
        calibrated = raw * confidence
        if not mandatory and action != FILL_GAPS and confidence < self.min_confidence:
            calibrated = min(calibrated, raw * self.min_confidence)
            if calibrated < self.min_voi:
                blocked = blocked or "calibration"
                calibrated = 0.0
                raw = 0.0
        elif confidence < self.min_confidence:
            calibrated = min(calibrated, raw * max(confidence, 0.15))
        return ValueCandidate(
            action=action,
            kind=_KIND[action],
            target=target,
            gain=round(usable, 4),
            cost=cost,
            voi=round(raw, 4),
            reason=reason,
            calibrated_voi=round(calibrated, 4),
            confidence=round(confidence, 4),
            reliability=round(reliability, 4),
            mandatory=mandatory,
            blocked_by=blocked,
        )


def source_reliability(state: ResearchState) -> float:
    """Lower score when sources repeat or look like the same passage."""
    passages = [item.text.strip().casefold() for item in state.evidence if item.text.strip()]
    if not passages:
        return 1.0
    docs = {item.document_id or item.source for item in state.evidence if item.document_id or item.source}
    unique_text = len(set(passages)) / len(passages)
    unique_docs = (len(docs) / len(state.evidence)) if state.evidence else 1.0
    return max(0.15, (unique_text + unique_docs) / 2)


def estimate_confidence(judged: int, reliability: float) -> float:
    """Few verdicts or noisy sources make a high VoI score untrustworthy."""
    sample = min(1.0, judged / 3) if judged else 0.35
    return max(0.15, sample * (0.5 + 0.5 * reliability))


def _gap_covers(claim: str, gaps: tuple[str, ...]) -> bool:
    needle = claim.strip().casefold()
    if not needle:
        return False
    return any(needle in gap.casefold() or gap.casefold() in needle for gap in gaps)


def can_stop(state: ResearchState, ranked: list[ValueCandidate]) -> bool:
    """Stop only when optional work is cheap and mandatory checks are done."""
    from app.research.evidence_plan import EvidenceDrivenPlanner

    signals = EvidenceDrivenPlanner().inspect(state)
    mandatory_open = bool(signals.unresolved_claims) or signals.contradiction_severity in {"high", "medium"}
    if mandatory_open and any(item.mandatory and not item.blocked_by for item in ranked):
        return False
    if mandatory_open and not ranked:
        return True
    optional = [item for item in ranked if not item.mandatory and item.calibrated_voi >= MIN_VOI]
    return not optional


def record_selection(state: ResearchState, candidate: ValueCandidate, quality_before: float) -> dict:
    entry = {
        "action": candidate.action,
        "target": candidate.target,
        "kind": candidate.kind,
        "estimated_voi": candidate.voi,
        "calibrated_voi": candidate.calibrated_voi,
        "estimated_gain": candidate.gain,
        "cost": candidate.cost,
        "confidence": candidate.confidence,
        "reliability": candidate.reliability,
        "mandatory": candidate.mandatory,
        "reason": candidate.reason,
        "quality_before": quality_before,
        "quality_after": None,
        "actual_gain": None,
        "improved": None,
    }
    state.voi_log.append(entry)
    return entry


def observe_result(state: ResearchState, quality_after: float) -> dict | None:
    if not state.voi_log:
        return None
    last = state.voi_log[-1]
    if last.get("quality_after") is not None:
        return last
    last["quality_after"] = quality_after
    last["actual_gain"] = round(quality_after - float(last.get("quality_before") or 0), 4)
    last["improved"] = last["actual_gain"] > 0
    return last


def _blocked(cost: int, window: BudgetWindow) -> str:
    if window.deadline and window.now >= window.deadline:
        return "time"
    if window.remaining_budget < cost:
        return "budget"
    limit = window.max_cost if window.max_cost > 0 else 0
    if limit and window.spent_cost + cost > limit:
        return "spending"
    if window.max_iterations and window.iterations + 1 > window.max_iterations:
        return "tool_calls"
    return ""


def _runnable(item: ValueCandidate, min_voi: float, min_confidence: float) -> bool:
    if item.blocked_by:
        return False
    if item.mandatory or item.action == FILL_GAPS:
        return True
    if item.confidence < min_confidence and item.calibrated_voi < min_voi:
        return False
    return item.calibrated_voi >= min_voi


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
