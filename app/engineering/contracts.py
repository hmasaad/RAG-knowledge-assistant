"""Typed evaluation report for one agent run."""

from dataclasses import asdict, dataclass, field

EVALUATOR_VERSION = "1.0.0"

OUTCOMES = ("success", "failure", "partial", "interrupted", "unknown")


@dataclass(frozen=True)
class EvaluationDimension:
    name: str
    score: float
    passed: bool
    detail: str = ""


@dataclass(frozen=True)
class EvaluationFinding:
    code: str
    detail: str
    evidence_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvaluationReport:
    """Outcome, scored dimensions, findings, and the evidence they rest on."""

    run_id: str
    outcome: str
    dimensions: tuple[EvaluationDimension, ...]
    findings: tuple[EvaluationFinding, ...]
    evidence_refs: tuple[str, ...]
    confidence: float
    evaluator_version: str = EVALUATOR_VERSION
    agent_version: str = ""
    failure_class: str = ""
    recovery: str = ""

    def to_payload(self) -> dict:
        return asdict(self)

    @classmethod
    def from_payload(cls, payload: dict) -> "EvaluationReport":
        return cls(
            run_id=str(payload.get("run_id", "")),
            outcome=str(payload.get("outcome", "unknown")),
            dimensions=tuple(
                EvaluationDimension(
                    name=str(item.get("name", "")),
                    score=float(item.get("score", 0) or 0),
                    passed=bool(item.get("passed", False)),
                    detail=str(item.get("detail", "")),
                )
                for item in payload.get("dimensions", []) or []
            ),
            findings=tuple(
                EvaluationFinding(
                    code=str(item.get("code", "")),
                    detail=str(item.get("detail", "")),
                    evidence_ids=tuple(str(value) for value in item.get("evidence_ids", ()) or ()),
                )
                for item in payload.get("findings", []) or []
            ),
            evidence_refs=tuple(str(item) for item in payload.get("evidence_refs", ()) or ()),
            confidence=float(payload.get("confidence", 0) or 0),
            evaluator_version=str(payload.get("evaluator_version", EVALUATOR_VERSION)),
            agent_version=str(payload.get("agent_version", "")),
            failure_class=str(payload.get("failure_class", "")),
            recovery=str(payload.get("recovery", "")),
        )


def dimension(name: str, score: float, *, passed: bool | None = None, detail: str = "") -> EvaluationDimension:
    value = max(0.0, min(1.0, score))
    return EvaluationDimension(
        name=name,
        score=value,
        passed=value >= 1.0 if passed is None else passed,
        detail=detail,
    )
