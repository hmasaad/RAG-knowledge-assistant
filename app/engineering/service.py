"""Record, evaluate, classify, and summarize agent engineering runs."""

from pathlib import Path

from app.engineering.classifier import classify_failure
from app.engineering.contracts import (
    EVALUATOR_VERSION,
    EvaluationFinding,
    EvaluationReport,
    dimension,
)
from app.engineering.dashboard import Dashboard, build_dashboard
from app.engineering.improve import ImprovementDecision, compare_reports
from app.engineering.governance import (
    GovernanceDecision,
    GovernancePolicy,
    ProductionGovernance,
    ReadinessProbes,
    ReadinessReport,
)
from app.engineering.loop import ContinuousImprovementLoop, LoopSnapshot, LoopState, compare_metrics
from app.engineering.recovery import analyze_recovery
from app.engineering.trajectory import (
    ERROR,
    POLICY,
    TOOL_OUTCOME,
    VERIFICATION,
    Trajectory,
    TrajectoryRecorder,
)


class EngineeringService:
    """One facade for trajectories, evaluation contracts, and the dashboard."""

    def __init__(
        self,
        path: Path | None = None,
        *,
        retry_budget: int = 2,
        policy: GovernancePolicy | None = None,
        probes: ReadinessProbes | None = None,
    ) -> None:
        self.recorder = TrajectoryRecorder(path)
        self.retry_budget = retry_budget
        self._reports: dict[str, EvaluationReport] = {}
        self.loop = ContinuousImprovementLoop(path)
        self.governance = ProductionGovernance(path, self.loop, policy=policy, probes=probes)

    def start_run(self, **fields) -> str:
        return self.recorder.start(**fields)

    def record_tool(self, run_id: str, action: str, *, ok: bool, error: str = "", query: str = "", cost: int = 0) -> None:
        self.recorder.record(
            run_id,
            TOOL_OUTCOME,
            action=action,
            ok=ok,
            error=error,
            query=query,
            cost=cost,
        )

    def record_verification(self, run_id: str, claim: str, *, ok: bool, evidence_ids: tuple[str, ...] = ()) -> None:
        self.recorder.record(
            run_id,
            VERIFICATION,
            action="verify",
            ok=ok,
            claim=claim,
            evidence_ids=list(evidence_ids),
        )

    def record_error(self, run_id: str, error: str, *, action: str = "") -> None:
        self.recorder.record(run_id, ERROR, action=action, ok=False, error=error)

    def record_policy(self, run_id: str, error: str, *, action: str = "") -> None:
        self.recorder.record(run_id, POLICY, action=action, ok=False, error=error)

    def finish_run(self, run_id: str, **fields) -> Trajectory | None:
        return self.recorder.finish(run_id, **fields)

    def trajectory(self, run_id: str) -> Trajectory | None:
        return self.recorder.get(run_id)

    def evaluate(
        self,
        run_id: str,
        *,
        workflow_state: str = "",
        test_results: tuple[str, ...] = (),
        policy_events: tuple[str, ...] = (),
        reliability: float | None = None,
        quality_confident: bool | None = None,
        sufficient: bool | None = None,
        leaked: bool = False,
        budget_ok: bool | None = None,
    ) -> EvaluationReport:
        run = self.recorder.get(run_id)
        status = workflow_state or (run.status if run is not None else "unknown")
        classification = classify_failure(
            run,
            test_results=test_results,
            policy_events=policy_events,
            workflow_state=status,
        )
        recovery = analyze_recovery(run, retry_budget=self.retry_budget)
        tools = [event for event in (run.events if run else []) if event.kind == TOOL_OUTCOME]
        verifies = [event for event in (run.events if run else []) if event.kind == VERIFICATION]
        evidence = [
            str(item)
            for event in verifies
            for item in event.payload.get("evidence_ids", [])
        ]
        correctness = 1.0 if sufficient else (0.0 if sufficient is False else (1.0 if status == "completed" else 0.0))
        if reliability is not None:
            correctness = reliability
        safety = 0.0 if leaked or classification.label == "policy" else 1.0
        recovery_score = 1.0 if recovery.status in {"recovered", "none"} and not recovery.terminal else 0.0
        findings: list[EvaluationFinding] = []
        if classification.label not in {"", "unknown"} and status != "completed":
            findings.append(EvaluationFinding(classification.label, classification.detail, tuple(evidence)))
        if recovery.terminal:
            findings.append(EvaluationFinding(recovery.status, recovery.detail))
        if quality_confident is False:
            findings.append(EvaluationFinding("low_confidence", "The quality gate was not confident."))
        if status == "completed" and correctness >= 1.0 and safety >= 1.0:
            outcome = "success"
        elif status in {"halted", "cancelled", "interrupted"}:
            outcome = "interrupted"
        elif status == "completed":
            outcome = "partial"
        elif status == "unknown" and run is None:
            outcome = "unknown"
        else:
            outcome = "failure"
        report = EvaluationReport(
            run_id=run_id,
            outcome=outcome,
            dimensions=(
                dimension("correctness", correctness),
                dimension("safety", safety, detail="leak or policy" if safety < 1 else ""),
                dimension("recovery", recovery_score, passed=not recovery.terminal),
                dimension("budget", 1.0 if budget_ok is not False else 0.0),
            ),
            findings=tuple(findings),
            evidence_refs=tuple(evidence),
            confidence=0.9 if verifies or tools else 0.4,
            evaluator_version=EVALUATOR_VERSION,
            agent_version=run.agent_version if run is not None else "",
            failure_class=classification.label if outcome != "success" else "",
            recovery=recovery.status,
        )
        self._reports[run_id] = report
        self.recorder.save_report(run_id, report.to_payload())
        return report

    def dashboard(self) -> Dashboard:
        stored = {
            run_id: EvaluationReport.from_payload(payload)
            for run_id, payload in self.recorder.load_reports().items()
        }
        stored.update(self._reports)
        return build_dashboard(self.recorder.list_runs(), stored)

    def compare(self, baseline: EvaluationReport, candidate: EvaluationReport) -> ImprovementDecision:
        return compare_reports(baseline, candidate)

    def record_loop(self, planner_version: str, metrics: dict[str, float], *, promote: bool = False) -> LoopSnapshot:
        return self.loop.record(planner_version, metrics, promote=promote)

    def promote_loop(self, planner_version: str, metrics: dict[str, float]) -> LoopSnapshot:
        return self.loop.promote(planner_version, metrics)

    def loop_state(self) -> LoopState:
        return self.loop.state()

    def compare_loop(self, baseline: dict[str, float], candidate: dict[str, float]) -> ImprovementDecision:
        return compare_metrics(baseline, candidate)

    def readiness(self, metrics: dict[str, float] | None = None) -> ReadinessReport:
        return self.governance.readiness(metrics)

    def release(self, planner_version: str, metrics: dict[str, float], *, actor: str) -> GovernanceDecision:
        return self.governance.release(planner_version, metrics, actor=actor)

    def rollback(self, *, actor: str) -> GovernanceDecision:
        return self.governance.rollback(actor=actor)

    def audit(self):
        return self.governance.audit()
