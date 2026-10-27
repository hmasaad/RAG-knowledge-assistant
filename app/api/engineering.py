"""AI Engineering Task Manager API."""

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict

from app.api.dependencies import get_services
from app.engineering.contracts import EvaluationReport
from app.ingestion.errors import KnowledgeError
from app.wiring import Services

router = APIRouter(tags=["engineering"])


class DimensionResponse(BaseModel):
    name: str
    score: float
    passed: bool
    detail: str = ""


class FindingResponse(BaseModel):
    code: str
    detail: str
    evidence_ids: list[str] = []


class EvaluationResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    run_id: str
    outcome: str
    dimensions: list[DimensionResponse]
    findings: list[FindingResponse]
    evidence_refs: list[str]
    confidence: float
    evaluator_version: str
    agent_version: str = ""
    failure_class: str = ""
    recovery: str = ""


class EventResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    seq: int
    kind: str
    action: str = ""
    ok: bool | None = None
    error: str = ""
    payload: dict = {}
    created_at: str = ""


class TrajectoryResponse(BaseModel):
    id: str
    agent_version: str = ""
    session_id: str = ""
    trace_id: str = ""
    question: str = ""
    status: str
    duration_ms: float = 0
    token_usage: int = 0
    estimated_cost: float = 0
    events: list[EventResponse]


class VersionStatsResponse(BaseModel):
    agent_version: str
    runs: int
    success_rate: float
    duration_ms: float
    token_usage: int
    estimated_cost: float
    recovered: int
    failures: dict[str, int] = {}


class DashboardResponse(BaseModel):
    versions: list[VersionStatsResponse]
    runs: list[dict]


class CompareRequest(BaseModel):
    baseline: dict
    candidate: dict


class CompareResponse(BaseModel):
    allowed: bool
    regressions: list[str]
    detail: str


@router.get("/engineering/dashboard", response_model=DashboardResponse)
def read_dashboard(services: Services = Depends(get_services)) -> DashboardResponse:
    board = services.engineering.dashboard()
    return DashboardResponse(
        versions=[
            VersionStatsResponse(
                agent_version=item.agent_version,
                runs=item.runs,
                success_rate=item.success_rate,
                duration_ms=item.duration_ms,
                token_usage=item.token_usage,
                estimated_cost=item.estimated_cost,
                recovered=item.recovered,
                failures=item.failures,
            )
            for item in board.versions
        ],
        runs=board.runs,
    )


@router.get("/engineering/runs/{run_id}", response_model=TrajectoryResponse)
def read_run(run_id: str, services: Services = Depends(get_services)) -> TrajectoryResponse:
    run = services.engineering.trajectory(run_id)
    if run is None:
        raise KnowledgeError("Engineering run not found.", status_code=404)
    return TrajectoryResponse(
        id=run.id,
        agent_version=run.agent_version,
        session_id=run.session_id,
        trace_id=run.trace_id,
        question=run.question,
        status=run.status,
        duration_ms=run.duration_ms,
        token_usage=run.token_usage,
        estimated_cost=run.estimated_cost,
        events=[
            EventResponse(
                seq=event.seq,
                kind=event.kind,
                action=event.action,
                ok=event.ok,
                error=event.error,
                payload=event.payload,
                created_at=event.created_at,
            )
            for event in run.events
        ],
    )


@router.post("/engineering/runs/{run_id}/evaluate", response_model=EvaluationResponse)
def evaluate_run(run_id: str, services: Services = Depends(get_services)) -> EvaluationResponse:
    run = services.engineering.trajectory(run_id)
    if run is None:
        raise KnowledgeError("Engineering run not found.", status_code=404)
    report = services.engineering.evaluate(run_id, workflow_state=run.status)
    return _evaluation_response(report)


@router.post("/engineering/compare", response_model=CompareResponse)
def compare_evaluations(
    body: CompareRequest,
    services: Services = Depends(get_services),
) -> CompareResponse:
    decision = services.engineering.compare(
        EvaluationReport.from_payload(body.baseline),
        EvaluationReport.from_payload(body.candidate),
    )
    return CompareResponse(
        allowed=decision.allowed,
        regressions=list(decision.regressions),
        detail=decision.detail,
    )


def _evaluation_response(report: EvaluationReport) -> EvaluationResponse:
    return EvaluationResponse(
        run_id=report.run_id,
        outcome=report.outcome,
        dimensions=[
            DimensionResponse(
                name=item.name,
                score=item.score,
                passed=item.passed,
                detail=item.detail,
            )
            for item in report.dimensions
        ],
        findings=[
            FindingResponse(code=item.code, detail=item.detail, evidence_ids=list(item.evidence_ids))
            for item in report.findings
        ],
        evidence_refs=list(report.evidence_refs),
        confidence=report.confidence,
        evaluator_version=report.evaluator_version,
        agent_version=report.agent_version,
        failure_class=report.failure_class,
        recovery=report.recovery,
    )
