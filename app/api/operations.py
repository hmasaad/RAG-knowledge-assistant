"""Traces and index recovery."""

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict

from app.api.dependencies import get_services
from app.ingestion.errors import KnowledgeError
from app.wiring import Services

router = APIRouter(tags=["operations"])


class TraceStepResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    action: str
    query: str | None = None
    sufficient_evidence: bool | None = None
    hit_count: int | None = None
    tool: str | None = None
    span_id: str | None = None
    parent_id: str | None = None
    duration_ms: float | None = None
    cost: int | None = None
    status: str | None = None
    error: str | None = None
    kind: str | None = None


class TraceResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str
    steps: list[TraceStepResponse]
    question: str = ""
    user_id: str = ""
    session_id: str = ""
    duration_ms: float | None = None
    cost: int | None = None
    status: str = ""
    reliability: float | None = None
    quality_confident: bool | None = None
    quality_failures: list[str] = []


class RecoveryResponse(BaseModel):
    requeued_failed: int
    requeued_missing: int
    reindexed: int


@router.get("/traces/{trace_id}", response_model=TraceResponse)
def read_trace(trace_id: str, services: Services = Depends(get_services)) -> TraceResponse:
    trace = services.traces.get(trace_id)
    if trace is None:
        raise KnowledgeError("Trace not found.", status_code=404)
    failures = trace.get("quality_failures") or []
    return TraceResponse(
        id=trace["id"],
        question=trace.get("question", ""),
        user_id=trace.get("user_id", ""),
        session_id=trace.get("session_id", ""),
        duration_ms=trace.get("duration_ms"),
        cost=trace.get("cost"),
        status=str(trace.get("status", "") or ""),
        reliability=trace.get("reliability"),
        quality_confident=trace.get("quality_confident"),
        quality_failures=list(failures),
        steps=[TraceStepResponse(**step) for step in trace["steps"]],
    )


@router.post("/operations/recover", response_model=RecoveryResponse)
def recover(services: Services = Depends(get_services)) -> RecoveryResponse:
    report = services.ingestion.recover()
    return RecoveryResponse(
        requeued_failed=report.requeued_failed,
        requeued_missing=report.requeued_missing,
        reindexed=report.reindexed,
    )
