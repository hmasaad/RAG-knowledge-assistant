"""Source-trace API over the evidence graph."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.dependencies import get_services, get_user_id
from app.research.intelligence import run_research_loop
from app.research.resolution import approval_log
from app.research.trace import answer_from_trace, library_scene, named_demo_asset, trace_scene
from app.wiring import Services

router = APIRouter(tags=["trace"])


class TraceAssetModel(BaseModel):
    id: str
    name: str
    color: str


class TraceMarkerModel(BaseModel):
    n: int
    claim_id: str


class TraceVersionModel(BaseModel):
    version: int
    checksum: str
    label: str
    recorded_at: str = ""


class TraceLineageModel(BaseModel):
    kind: str
    id: str
    label: str
    version: int = 0
    recorded_at: str = ""
    checksum: str = ""


class TraceCitationModel(BaseModel):
    marker: int | None = None
    quote: str = ""
    filename: str = ""
    version: int = 0
    chunk_id: str = ""
    start_char: int = 0
    end_char: int = 0
    recorded_at: str = ""


class TraceAuditModel(BaseModel):
    decision: str
    detail: str = ""
    recorded_at: str = ""
    evidence_ids: list[str] = Field(default_factory=list)


class TraceProvenanceModel(BaseModel):
    document_id: str = ""
    filename: str = ""
    version: int = 0
    checksum: str = ""
    chunk_id: str = ""
    heading: str | None = None
    start_char: int = 0
    end_char: int = 0
    recorded_at: str = ""
    lineage: list[TraceLineageModel] = Field(default_factory=list)
    history: list[TraceVersionModel] = Field(default_factory=list)


class TraceClaimModel(BaseModel):
    id: str
    text: str
    short: str
    agent_id: str
    band: str
    upstream: list[str]
    upstream_count: int
    marker: int | None = None
    entities: list[str]
    contradicts: list[str]
    provenance: TraceProvenanceModel
    citation: TraceCitationModel | None = None
    audit: list[TraceAuditModel] = Field(default_factory=list)


class TraceAgentModel(BaseModel):
    id: str
    label: str
    role: str


class TraceEntityModel(BaseModel):
    id: str
    label: str


class TraceChunkModel(BaseModel):
    id: str
    agent_id: str
    filename: str
    version: int
    label: str
    checksum: str = ""
    chunk_id: str = ""
    document_id: str = ""


class TraceEdgeModel(BaseModel):
    source: str
    relation: str
    target: str


class TracePlanItemModel(BaseModel):
    id: str
    kind: str
    missing: str
    question: str
    strategy: str
    strategy_reason: str
    gain: float
    cost: int
    value: float
    priority: int
    claim_ids: list[str] = Field(default_factory=list)


class TracePlanModel(BaseModel):
    summary: str = ""
    next_id: str = ""
    items: list[TracePlanItemModel] = Field(default_factory=list)


class TraceConflictModel(BaseModel):
    id: str
    left_id: str
    right_id: str
    status: str
    winner_id: str = ""
    loser_id: str = ""
    reason: str = ""
    left_role: str = ""
    right_role: str = ""
    left_authority: int = 0
    right_authority: int = 0
    reviewer: str = ""


class TraceSceneModel(BaseModel):
    asset: str
    assets: list[TraceAssetModel]
    question: str
    answer: str
    markers: list[TraceMarkerModel]
    agents: list[TraceAgentModel]
    entities: list[TraceEntityModel]
    claims: list[TraceClaimModel]
    chunks: list[TraceChunkModel]
    edges: list[TraceEdgeModel]
    selected_agent: str
    selected_claim: str
    decisions: list[TraceAuditModel] = Field(default_factory=list)
    conflicts: list[TraceConflictModel] = Field(default_factory=list)
    plan: TracePlanModel = Field(default_factory=TracePlanModel)


class TraceApprovalRequest(BaseModel):
    asset: str = "NVDA"
    conflict_id: str = Field(min_length=1)
    winner_id: str = Field(min_length=1)
    reviewer: str = ""


class TraceAskRequest(BaseModel):
    question: str = Field(min_length=1)
    asset: str = "NVDA"


class TraceAskModel(BaseModel):
    asset: str
    answer: str
    markers: list[TraceMarkerModel]
    scene: TraceSceneModel


class LoopStepModel(BaseModel):
    action: str
    status: str
    question: str = ""
    strategy: str = ""
    detail: str = ""
    searches_used: int = 0
    tokens_used: int = 0
    elapsed_seconds: float = 0
    confidence: float = 0


class LoopMetricsModel(BaseModel):
    confidence: float = 0
    quality: float = 0
    completeness: float = 0
    factual_support: float = 0
    citation_validity: float = 0
    source_diversity: float = 0
    freshness: float = 0
    unresolved_contradictions: float = 0
    sufficient: bool = False


class LoopConflictModel(BaseModel):
    id: str = ""
    status: str = ""
    reason: str = ""
    left: str = ""
    right: str = ""
    winner_id: str = ""


class LoopRequest(BaseModel):
    question: str = Field(min_length=1)
    asset: str = "NVDA"
    max_searches: int = Field(default=4, ge=1, le=8)
    max_tokens: int = Field(default=2500, ge=1, le=20000)
    max_seconds: float = Field(default=15, gt=0, le=60)


class LoopReportModel(BaseModel):
    question: str
    asset: str
    answer: str = ""
    stop_reason: str
    sufficient: bool
    searches_used: int
    search_budget: int
    tokens_used: int
    token_budget: int
    elapsed_seconds: float
    time_budget: float
    steps: list[LoopStepModel]
    metrics: LoopMetricsModel
    contradictions: list[LoopConflictModel]
    scene: TraceSceneModel | None = None


def _scene_or_404(asset: str) -> dict:
    scene = trace_scene(asset)
    if scene is None:
        raise HTTPException(status_code=404, detail=f"Unknown asset {asset.strip().upper()}.")
    return scene


@router.get("/trace", response_model=TraceSceneModel)
def read_trace(asset: str = "NVDA") -> dict:
    return _scene_or_404(asset)


@router.post("/trace/ask", response_model=TraceAskModel)
def ask_trace(
    body: TraceAskRequest,
    services: Services = Depends(get_services),
    user_id: str = Depends(get_user_id),
) -> dict:
    question = body.question.strip()
    if named_demo_asset(question) is None:
        scene = library_scene(services.generation.answer(question, 4, user_id=user_id), services.metadata)
        if scene is not None:
            return {
                "asset": scene["asset"],
                "answer": scene["answer"],
                "markers": scene["markers"],
                "scene": scene,
            }
    result = answer_from_trace(question, body.asset)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Unknown asset {body.asset.strip().upper()}.")
    return result


@router.post("/trace/conflicts/approve", response_model=TraceSceneModel)
def approve_conflict(
    body: TraceApprovalRequest,
    user_id: str = Depends(get_user_id),
) -> dict:
    scene = _scene_or_404(body.asset)
    conflict = next((item for item in scene["conflicts"] if item["id"] == body.conflict_id), None)
    if conflict is None:
        raise HTTPException(status_code=404, detail="Unknown conflict.")
    if body.winner_id not in {conflict["left_id"], conflict["right_id"]}:
        raise HTTPException(status_code=400, detail="The winner has to be one of the two claims.")
    if conflict["status"] not in {"pending", "approved"}:
        raise HTTPException(status_code=409, detail="Supersession and authority already settled this conflict.")
    approval_log.decide(body.asset, body.conflict_id, body.winner_id, body.reviewer or user_id)
    return _scene_or_404(body.asset)


@router.post("/trace/loop", response_model=LoopReportModel)
def research_loop(
    body: LoopRequest,
    services: Services = Depends(get_services),
    user_id: str = Depends(get_user_id),
) -> dict:
    def search(question: str, asset: str) -> dict:
        if asset not in {"NVDA", "TSLA", "GOLD"}:
            return {"asset": asset, "answer": "", "markers": [], "scene": None}
        if named_demo_asset(question) is None:
            scene = library_scene(services.generation.answer(question, 4, user_id=user_id), services.metadata)
            if scene is not None:
                return {
                    "asset": scene["asset"],
                    "answer": scene["answer"],
                    "markers": scene["markers"],
                    "scene": scene,
                }
        result = answer_from_trace(question, asset)
        if result is None:
            raise LookupError(f"Unknown asset {asset.strip().upper()}.")
        return result

    def outside_search(question: str, asset: str) -> dict:
        scene = library_scene(services.generation.answer(question, 4, user_id=user_id), services.metadata)
        if scene is None:
            return {"asset": asset, "answer": "", "markers": [], "scene": None}
        return {
            "asset": scene["asset"],
            "answer": scene["answer"],
            "markers": scene["markers"],
            "scene": scene,
        }

    return run_research_loop(
        body.question,
        search,
        asset=body.asset,
        max_searches=body.max_searches,
        max_tokens=body.max_tokens,
        max_seconds=body.max_seconds,
        outside_search=outside_search,
    )
