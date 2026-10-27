"""Aggregate run outcomes for the evaluation dashboard."""

from dataclasses import dataclass, field

from app.engineering.classifier import classify_failure
from app.engineering.contracts import EvaluationReport
from app.engineering.recovery import analyze_recovery
from app.engineering.trajectory import Trajectory


@dataclass
class VersionStats:
    agent_version: str
    runs: int = 0
    successes: int = 0
    duration_ms: float = 0.0
    token_usage: int = 0
    estimated_cost: float = 0.0
    recovered: int = 0
    failures: dict[str, int] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        return 0.0 if self.runs == 0 else self.successes / self.runs


@dataclass
class Dashboard:
    versions: list[VersionStats]
    runs: list[dict]


def build_dashboard(
    trajectories: list[Trajectory],
    reports: dict[str, EvaluationReport] | None = None,
) -> Dashboard:
    reports = reports or {}
    grouped: dict[str, VersionStats] = {}
    rows: list[dict] = []
    for run in trajectories:
        report = reports.get(run.id)
        classification = classify_failure(run, workflow_state=run.status)
        recovery = analyze_recovery(run)
        stats = grouped.setdefault(run.agent_version or "unknown", VersionStats(run.agent_version or "unknown"))
        stats.runs += 1
        stats.duration_ms += run.duration_ms
        stats.token_usage += run.token_usage
        stats.estimated_cost += run.estimated_cost
        succeeded = run.status == "completed" and (report.outcome == "success" if report else True)
        if report is not None:
            succeeded = report.outcome == "success"
        if succeeded:
            stats.successes += 1
        if not succeeded:
            label = report.failure_class if report and report.failure_class else classification.label
            stats.failures[label] = stats.failures.get(label, 0) + 1
        if recovery.status == "recovered":
            stats.recovered += 1
        rows.append(
            {
                "run_id": run.id,
                "agent_version": run.agent_version,
                "session_id": run.session_id,
                "status": run.status,
                "duration_ms": run.duration_ms,
                "token_usage": run.token_usage,
                "estimated_cost": run.estimated_cost,
                "outcome": report.outcome if report else run.status,
                "failure_class": report.failure_class if report else classification.label,
                "recovery": report.recovery if report else recovery.status,
            }
        )
    versions = sorted(grouped.values(), key=lambda item: item.agent_version)
    return Dashboard(versions=versions, runs=rows)
