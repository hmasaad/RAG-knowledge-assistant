"""Replay a fixed research benchmark and compare two agent versions."""

import tempfile
from pathlib import Path

from app.engineering.contracts import EvaluationReport
from app.engineering.improve import ImprovementDecision, compare_reports
from evaluation.harness import _services

ROOT = Path(__file__).resolve().parents[1]


def replay_benchmark(
    *,
    agent_version: str,
    root: Path | None = None,
    directory: Path | None = None,
) -> list[EvaluationReport]:
    """Run the retry-file cycle once and return a typed evaluation report."""
    base = root or ROOT
    owned = tempfile.TemporaryDirectory() if directory is None else None
    path = directory if directory is not None else Path(owned.name)
    try:
        services = _services(path)
        payload = (base / "samples" / "api-errors.md").read_bytes()
        services.ingestion.ingest("api-errors.md", payload)
        report = services.research.cycle("which file implements the retry logic", top_k=3)
        run_id = report.session_id or report.trace_id
        stored = services.engineering.trajectory(run_id)
        if stored is None:
            run_id = services.engineering.start_run(
                agent_version=agent_version,
                session_id=report.session_id,
                trace_id=report.trace_id,
                question=report.question,
            )
            services.engineering.finish_run(
                run_id,
                status=report.status,
                estimated_cost=float(max(0, 16 - report.remaining_budget)),
            )
        return [
            services.engineering.evaluate(
                run_id,
                workflow_state=report.status,
                reliability=report.reliability,
                quality_confident=report.quality_confident,
                sufficient=report.sufficient_evidence,
            )
        ]
    finally:
        if owned is not None:
            owned.cleanup()


def decide_improvement(baseline: EvaluationReport, candidate: EvaluationReport) -> ImprovementDecision:
    return compare_reports(baseline, candidate)
