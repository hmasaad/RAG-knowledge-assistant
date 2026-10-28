"""Bayesian experimental design and sequential research decisions."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

from app.research.contracts import (
    FILL_GAPS,
    RESOLVE_CONTRADICTIONS,
    REVERIFY,
    ResearchState,
)
from app.research.voi import MIN_VOI, ValueCandidate, can_stop

MIN_EIG = 0.08
_WEAK = frozenset({"insufficient", "unsupported", "contradicted"})


@dataclass(frozen=True)
class ClaimBelief:
    """Beta posterior over whether a claim is supported."""

    claim: str
    alpha: float = 1.0
    beta: float = 1.0

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def entropy(self) -> float:
        return _binary_entropy(self.mean)

    def updated(self, supported: bool) -> "ClaimBelief":
        if supported:
            return replace(self, alpha=self.alpha + 1)
        return replace(self, beta=self.beta + 1)


@dataclass(frozen=True)
class BeliefState:
    """Current beliefs about claims, gaps, and contradictions."""

    claims: tuple[ClaimBelief, ...] = ()
    gaps: tuple[str, ...] = ()
    contradiction_mass: float = 0.0

    def uncertainty(self) -> float:
        claim_h = sum(item.entropy() for item in self.claims) / max(1, len(self.claims))
        if not self.claims:
            claim_h = 1.0 if self.gaps else 0.25
        return claim_h + (0.35 if self.gaps else 0.0) + self.contradiction_mass

    def to_payload(self) -> dict:
        return {
            "uncertainty": round(self.uncertainty(), 4),
            "contradiction_mass": self.contradiction_mass,
            "gaps": list(self.gaps),
            "claims": [
                {
                    "claim": item.claim,
                    "mean": round(item.mean, 4),
                    "entropy": round(item.entropy(), 4),
                    "alpha": item.alpha,
                    "beta": item.beta,
                }
                for item in self.claims
            ],
        }


@dataclass(frozen=True)
class SequentialPlan:
    """The next action and a short look-ahead sequence as beliefs evolve."""

    stop: bool
    reason: str
    beliefs: BeliefState
    ranked: list[ValueCandidate] = field(default_factory=list)
    sequence: tuple[dict, ...] = ()

    def to_payload(self) -> dict:
        return {
            "stop": self.stop,
            "reason": self.reason,
            "beliefs": self.beliefs.to_payload(),
            "sequence": list(self.sequence),
            "ranked": [item.to_payload() for item in self.ranked],
        }


class SequentialDesigner:
    """Choose the information-gathering sequence that cuts uncertainty most."""

    def __init__(self, min_eig: float = MIN_EIG, horizon: int = 3) -> None:
        self.min_eig = min_eig
        self.horizon = max(1, horizon)

    def beliefs(self, state: ResearchState) -> BeliefState:
        claims: list[ClaimBelief] = []
        seen: set[str] = set()
        for verdict in state.verdicts:
            key = verdict.claim.strip()
            if not key or key.casefold() in seen:
                continue
            seen.add(key.casefold())
            alpha = 1.0
            beta = 1.0
            for item in state.verdicts:
                if item.claim.strip().casefold() != key.casefold():
                    continue
                if item.verdict == "supported":
                    alpha += 1
                elif item.verdict in _WEAK:
                    beta += 1
            claims.append(ClaimBelief(key, alpha=alpha, beta=beta))
        for gap in state.gaps:
            if gap.strip() and gap.strip().casefold() not in seen:
                claims.append(ClaimBelief(gap.strip(), alpha=1.0, beta=1.0))
                seen.add(gap.strip().casefold())
        mass = 0.0
        if state.contradictions:
            mass = 0.45 if len(state.contradictions) > 1 else 0.35
            text = " ".join(state.contradictions)
            if any(ch.isdigit() for ch in text):
                mass = 0.55
        return BeliefState(claims=tuple(claims), gaps=tuple(state.gaps), contradiction_mass=mass)

    def expected_information_gain(self, candidate: ValueCandidate, beliefs: BeliefState) -> float:
        before = beliefs.uncertainty()
        after = self._imagine(beliefs, candidate.action, candidate.target).uncertainty()
        return max(0.0, before - after)

    def plan(
        self,
        state: ResearchState,
        candidates: list[ValueCandidate],
        *,
        remaining_budget: int = 16,
    ) -> SequentialPlan:
        beliefs = self.beliefs(state)
        ranked = self._rank(candidates, beliefs)
        best_eig = ranked[0].eig if ranked else 0.0
        stop = can_stop(state, ranked) and best_eig < self.min_eig
        if not ranked:
            stop = True
        if ranked and ranked[0].mandatory:
            stop = False
        if ranked and ranked[0].action == FILL_GAPS and remaining_budget >= ranked[0].cost:
            stop = False
        sequence = self._lookahead(ranked, beliefs, remaining_budget)
        if stop:
            reason = "Expected information gain is too small and mandatory checks are done."
        elif ranked:
            reason = f"Next observation of {ranked[0].action} has EIG {ranked[0].eig:.3f}."
        else:
            reason = "No informative experiment remains."
        return SequentialPlan(stop=stop, reason=reason, beliefs=beliefs, ranked=ranked, sequence=sequence)

    def _rank(self, candidates: list[ValueCandidate], beliefs: BeliefState) -> list[ValueCandidate]:
        scored: list[ValueCandidate] = []
        for item in candidates:
            eig = round(self.expected_information_gain(item, beliefs), 4)
            scored.append(replace(item, eig=eig, reason=item.reason))
        order = {RESOLVE_CONTRADICTIONS: 0, FILL_GAPS: 1, REVERIFY: 2}
        scored.sort(
            key=lambda item: (
                not item.mandatory,
                order.get(item.action, 3),
                -item.eig,
                -item.calibrated_voi,
            )
        )
        return scored

    def _lookahead(
        self,
        ranked: list[ValueCandidate],
        beliefs: BeliefState,
        remaining_budget: int,
    ) -> tuple[dict, ...]:
        sequence: list[dict] = []
        current = beliefs
        budget = remaining_budget
        remaining = list(ranked)
        for _ in range(self.horizon):
            affordable = [item for item in remaining if item.cost <= budget and not item.blocked_by]
            if not affordable:
                break
            nxt = max(affordable, key=lambda item: (item.mandatory, item.eig, item.calibrated_voi))
            if nxt.eig < self.min_eig and not nxt.mandatory and nxt.action != FILL_GAPS:
                break
            sequence.append(
                {
                    "action": nxt.action,
                    "target": nxt.target,
                    "eig": nxt.eig,
                    "cost": nxt.cost,
                    "mandatory": nxt.mandatory,
                    "uncertainty_before": round(current.uncertainty(), 4),
                }
            )
            current = self._imagine(current, nxt.action, nxt.target)
            budget -= nxt.cost
            remaining = [
                replace(item, eig=round(self.expected_information_gain(item, current), 4))
                for item in remaining
                if not (item.action == nxt.action and item.target == nxt.target)
            ]
        return tuple(sequence)

    def _imagine(self, beliefs: BeliefState, action: str, target: str) -> BeliefState:
        claims = list(beliefs.claims)
        gaps = list(beliefs.gaps)
        mass = beliefs.contradiction_mass
        if action == FILL_GAPS:
            gaps = [gap for gap in gaps if gap.casefold() != target.casefold()]
            claims = [
                item.updated(True) if item.claim.casefold() == target.casefold() else item
                for item in claims
            ]
            if not any(item.claim.casefold() == target.casefold() for item in claims):
                claims.append(ClaimBelief(target, alpha=2.0, beta=1.0))
        elif action == REVERIFY:
            claims = [
                item.updated(item.mean >= 0.5)
                if item.claim.casefold() == target.casefold()
                else item
                for item in claims
            ]
        elif action == RESOLVE_CONTRADICTIONS:
            mass = 0.0
        return BeliefState(claims=tuple(claims), gaps=tuple(gaps), contradiction_mass=mass)


def _binary_entropy(probability: float) -> float:
    p = min(1.0 - 1e-9, max(1e-9, probability))
    return -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))
