"""AI Engineering Task Manager: trajectories, evaluation, recovery, and replay."""

from pathlib import Path

from fastapi.testclient import TestClient

from app.engineering.classifier import MULTIPLE_CAUSES, UNKNOWN, classify_failure
from app.engineering.contracts import EVALUATOR_VERSION, EvaluationReport, dimension
from app.engineering.improve import compare_reports
from app.engineering.recovery import RECOVERY_LOOP, RETRY_EXHAUSTED, analyze_recovery
from app.engineering.service import EngineeringService
from app.engineering.trajectory import TOOL_OUTCOME, TrajectoryRecorder

ROOT = Path(__file__).resolve().parents[1]


def test_trajectory_keeps_event_order_after_an_interrupted_run(tmp_path: Path) -> None:
    path = tmp_path / "metadata.db"
    first = TrajectoryRecorder(path)
    run_id = first.start(question="retry logic", session_id="s1", agent_version="0.1.0")
    first.record(run_id, TOOL_OUTCOME, action="search", ok=True, query="retry")
    first.record(run_id, TOOL_OUTCOME, action="reverify", ok=False, error="timed out")
    first.finish(run_id, status="halted")

    reloaded = TrajectoryRecorder(path).get(run_id)

    assert reloaded is not None
    assert [event.kind for event in reloaded.events][:3] == ["run_started", "tool_outcome", "tool_outcome"]
    assert [event.seq for event in reloaded.events] == list(range(len(reloaded.events)))
    assert reloaded.events[1].action == "search"
    assert reloaded.events[2].error == "timed out"
    assert reloaded.status == "halted"
    assert reloaded.events[-1].kind == "interrupted"


def test_evaluation_report_includes_required_fields() -> None:
    service = EngineeringService()
    run_id = service.start_run(question="retry", agent_version="0.1.0")
    service.record_tool(run_id, "search", ok=True, query="retry")
    service.record_verification(run_id, "retry uses three attempts", ok=True, evidence_ids=("e1",))
    service.finish_run(run_id, status="completed")
    report = service.evaluate(run_id, workflow_state="completed", reliability=1.0, sufficient=True)

    names = {item.name for item in report.dimensions}
    assert report.outcome == "success"
    assert report.evaluator_version == EVALUATOR_VERSION
    assert names == {"correctness", "safety", "recovery"}
    assert report.evidence_refs == ("e1",)
    assert 0 <= report.confidence <= 1
    assert EvaluationReport.from_payload(report.to_payload()).run_id == run_id


def test_failure_classifier_allows_unknown_and_multiple_causes() -> None:
    service = EngineeringService()
    empty = service.start_run(question="none")
    mixed = service.start_run(question="mixed")
    service.record_policy(mixed, "Caller is not permitted.", action="search")
    service.record_tool(mixed, "search", ok=False, error="Search budget exhausted.")
    service.finish_run(mixed, status="failed")

    unknown = classify_failure(service.trajectory(empty), workflow_state="running")
    several = classify_failure(service.trajectory(mixed), test_results=("failed",), workflow_state="failed")

    assert unknown.label == UNKNOWN
    assert several.label == MULTIPLE_CAUSES
    assert "policy" in several.causes
    assert "budget" in several.causes


def test_recovery_analysis_enforces_retry_budget_and_detects_loops() -> None:
    service = EngineeringService(retry_budget=1)
    exhausted = service.start_run(question="budget")
    service.record_tool(exhausted, "search", ok=False, query="wifi", error="no hits")
    service.record_tool(exhausted, "search", ok=False, query="wifi", error="no hits")
    looped = service.start_run(question="loop")
    for action in ("search", "fill_gaps", "search", "fill_gaps"):
        service.record_tool(looped, action, ok=False, query=action)

    budget = analyze_recovery(service.trajectory(exhausted), retry_budget=1)
    loop = analyze_recovery(service.trajectory(looped))

    assert budget.status == RETRY_EXHAUSTED
    assert budget.terminal is True
    assert loop.status == RECOVERY_LOOP
    assert loop.terminal is True


def test_dashboard_and_compare_block_safety_regressions(client: TestClient) -> None:
    payload = (ROOT / "samples" / "api-errors.md").read_bytes()
    client.post("/api/documents", files={"file": ("api-errors.md", payload, "text/markdown")})
    cycled = client.post(
        "/api/research/cycle",
        json={"question": "which file implements the retry logic", "top_k": 3},
    )
    dashboard = client.get("/api/engineering/dashboard")
    run = client.get(f"/api/engineering/runs/{cycled.json()['session_id']}")
    baseline = EvaluationReport(
        run_id="base",
        outcome="success",
        dimensions=(dimension("safety", 1.0), dimension("correctness", 1.0)),
        findings=(),
        evidence_refs=(),
        confidence=1.0,
    )
    worse = EvaluationReport(
        run_id="next",
        outcome="failure",
        dimensions=(dimension("safety", 0.0), dimension("correctness", 1.0)),
        findings=(),
        evidence_refs=(),
        confidence=0.4,
    )
    blocked = client.post(
        "/api/engineering/compare",
        json={"baseline": baseline.to_payload(), "candidate": worse.to_payload()},
    )

    assert cycled.status_code == 200
    assert dashboard.status_code == 200
    assert dashboard.json()["versions"]
    assert dashboard.json()["versions"][0]["runs"] >= 1
    assert "success_rate" in dashboard.json()["versions"][0]
    assert run.status_code == 200
    assert run.json()["events"][0]["kind"] == "run_started"
    assert [event["seq"] for event in run.json()["events"]] == list(range(len(run.json()["events"])))
    assert any(event["kind"] == "tool_outcome" for event in run.json()["events"])
    assert blocked.status_code == 200
    assert blocked.json()["allowed"] is False
    assert "safety" in blocked.json()["regressions"]
    assert compare_reports(baseline, baseline).allowed is True


def test_replay_benchmark_blocks_a_weaker_candidate() -> None:
    from evaluation.task_manager import decide_improvement, replay_benchmark

    reports = replay_benchmark(agent_version="0.1.0")
    weaker = EvaluationReport(
        run_id="changed",
        outcome="failure",
        dimensions=(dimension("safety", 0.0), dimension("correctness", 0.0)),
        findings=(),
        evidence_refs=(),
        confidence=0.2,
        agent_version="changed-prompt",
    )

    assert reports
    assert reports[0].evaluator_version == EVALUATOR_VERSION
    assert decide_improvement(reports[0], weaker).allowed is False
