"""Compare a changed agent against a fixed benchmark and block regressions."""

from dataclasses import dataclass

from app.engineering.contracts import EvaluationReport

MANDATORY = ("safety", "correctness")


@dataclass(frozen=True)
class ImprovementDecision:
    allowed: bool
    regressions: tuple[str, ...]
    detail: str


def compare_reports(
    baseline: EvaluationReport,
    candidate: EvaluationReport,
    *,
    mandatory: tuple[str, ...] = MANDATORY,
) -> ImprovementDecision:
    """Block a change when a mandatory safety or correctness score falls."""
    base = {item.name: item for item in baseline.dimensions}
    next_up = {item.name: item for item in candidate.dimensions}
    regressions: list[str] = []
    for name in mandatory:
        before = base.get(name)
        after = next_up.get(name)
        if before is None or after is None:
            regressions.append(name)
            continue
        if after.score + 1e-9 < before.score or (before.passed and not after.passed):
            regressions.append(name)
    if regressions:
        return ImprovementDecision(
            False,
            tuple(regressions),
            "Mandatory checks regressed: " + ", ".join(regressions) + ".",
        )
    return ImprovementDecision(True, (), "No mandatory safety or correctness regression.")
