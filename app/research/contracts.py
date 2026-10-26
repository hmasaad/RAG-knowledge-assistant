"""Provider-neutral contracts for research actions.

Adapters speak this language. They do not import an LLM client, a vector
store, or a metadata engine.
"""

from dataclasses import dataclass, field
from typing import Protocol

ACTIONS = (
    "search",
    "fill_gaps",
    "resolve_contradictions",
    "reverify",
    "synthesize",
)

SEARCH = "search"
FILL_GAPS = "fill_gaps"
RESOLVE_CONTRADICTIONS = "resolve_contradictions"
REVERIFY = "reverify"
SYNTHESIZE = "synthesize"


@dataclass(frozen=True)
class ActionRequest:
    """One worker action. Fields not required by that action stay empty."""

    action: str
    query: str = ""
    claim: str = ""
    evidence_ids: tuple[str, ...] = ()
    iteration: int = 0
    user_id: str = "local"
    top_k: int = 4


@dataclass(frozen=True)
class EvidenceRef:
    evidence_id: str
    document_id: str
    source: str
    text: str = ""


@dataclass(frozen=True)
class ClaimVerdict:
    claim: str
    verdict: str
    evidence_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ActionOutcome:
    action: str
    ok: bool
    error: str = ""
    evidence: tuple[EvidenceRef, ...] = ()
    verdicts: tuple[ClaimVerdict, ...] = ()
    gaps: tuple[str, ...] = ()
    contradictions: tuple[str, ...] = ()
    provenance: tuple[str, ...] = ()
    report: str = ""
    cost: int = 1


class ActionAdapter(Protocol):
    """Run one action. Implementations may call RAG, a judge, or a fake."""

    action: str

    def run(self, request: ActionRequest) -> ActionOutcome: ...


@dataclass
class ResearchState:
    """What the worker has learned so far, including cost and failures."""

    evidence: list[EvidenceRef] = field(default_factory=list)
    verdicts: list[ClaimVerdict] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)
    provenance: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    executed: set[tuple[str, str, str]] = field(default_factory=set)
    cost: int = 0
    iterations: int = 0
    report: str = ""
    steps: list[str] = field(default_factory=list)

    def to_payload(self) -> dict:
        return {
            "evidence": [
                {
                    "evidence_id": item.evidence_id,
                    "document_id": item.document_id,
                    "source": item.source,
                    "text": item.text,
                }
                for item in self.evidence
            ],
            "verdicts": [
                {
                    "claim": item.claim,
                    "verdict": item.verdict,
                    "evidence_ids": list(item.evidence_ids),
                }
                for item in self.verdicts
            ],
            "gaps": list(self.gaps),
            "contradictions": list(self.contradictions),
            "provenance": list(self.provenance),
            "errors": list(self.errors),
            "executed": [list(item) for item in sorted(self.executed)],
            "cost": self.cost,
            "iterations": self.iterations,
            "report": self.report,
            "steps": list(self.steps),
        }

    @classmethod
    def from_payload(cls, payload: dict | None) -> "ResearchState":
        data = payload or {}
        executed = {
            (str(item[0]), str(item[1]).strip().casefold(), str(item[2]).strip().casefold())
            for item in data.get("executed", [])
            if isinstance(item, (list, tuple)) and len(item) == 3
        }
        return cls(
            evidence=[
                EvidenceRef(
                    evidence_id=str(item.get("evidence_id", "")),
                    document_id=str(item.get("document_id", "")),
                    source=str(item.get("source", "")),
                    text=str(item.get("text", "")),
                )
                for item in data.get("evidence", [])
            ],
            verdicts=[
                ClaimVerdict(
                    claim=str(item.get("claim", "")),
                    verdict=str(item.get("verdict", "")),
                    evidence_ids=tuple(str(value) for value in item.get("evidence_ids", ())),
                )
                for item in data.get("verdicts", [])
            ],
            gaps=list(data.get("gaps", [])),
            contradictions=list(data.get("contradictions", [])),
            provenance=list(data.get("provenance", [])),
            errors=list(data.get("errors", [])),
            executed=executed,
            cost=int(data.get("cost", 0) or 0),
            iterations=int(data.get("iterations", 0) or 0),
            report=str(data.get("report", "")),
            steps=list(data.get("steps", [])),
        )

    def fingerprint(self, request: ActionRequest) -> tuple[str, str, str]:
        return (request.action, request.query.strip().casefold(), request.claim.strip().casefold())

    def apply(self, request: ActionRequest, outcome: ActionOutcome) -> None:
        if self.fingerprint(request) in self.executed:
            return
        self.iterations += 1
        self.cost += max(0, outcome.cost)
        self.executed.add(self.fingerprint(request))
        if outcome.error:
            self.errors.append(outcome.error)
        self.evidence.extend(outcome.evidence)
        self.verdicts.extend(outcome.verdicts)
        self.provenance.extend(outcome.provenance)
        if outcome.report:
            self.report = outcome.report
        if outcome.ok:
            for gap in outcome.gaps:
                if gap not in self.gaps:
                    self.gaps.append(gap)
            for item in outcome.contradictions:
                if item not in self.contradictions:
                    self.contradictions.append(item)
            self._clear_resolved_gaps(outcome)
        elif outcome.gaps:
            for gap in outcome.gaps:
                if gap not in self.gaps:
                    self.gaps.append(gap)

    def _clear_resolved_gaps(self, outcome: ActionOutcome) -> None:
        if not outcome.evidence and not outcome.verdicts:
            return
        resolved = {
            verdict.claim.casefold()
            for verdict in outcome.verdicts
            if verdict.verdict == "supported"
        }
        self.gaps = [
            gap
            for gap in self.gaps
            if gap.casefold() not in resolved and gap not in {item.text for item in outcome.evidence}
        ]
