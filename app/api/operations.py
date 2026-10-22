"""Traces and index recovery."""

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.api.dependencies import get_services
from app.ingestion.errors import KnowledgeError
from app.wiring import Services

router = APIRouter(tags=["operations"])


class TraceStepResponse(BaseModel):
    action: str
    query: str | None = None
    sufficient_evidence: bool | None = None
    hit_count: int | None = None


class TraceResponse(BaseModel):
    id: str
    steps: list[TraceStepResponse]


class RecoveryResponse(BaseModel):
    requeued_failed: int
    requeued_missing: int
    reindexed: int


@router.get("/traces/{trace_id}", response_model=TraceResponse)
def read_trace(trace_id: str, services: Services = Depends(get_services)) -> TraceResponse:
    trace = services.traces.get(trace_id)
    if trace is None:
        raise KnowledgeError("Trace not found.", status_code=404)
    return TraceResponse(
        id=trace["id"],
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
