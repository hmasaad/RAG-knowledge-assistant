"""Source-trace API over the evidence graph."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.dependencies import get_services, get_user_id
from app.research.intelligence import run_research_loop
from app.research.learning import research_memory
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
    top_k: int = 0
    agent: str = ""


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


class TrajectoryDecisionModel(BaseModel):
    index: int
    action: str
    question: str = ""
    strategy: str = ""
    verdict: str
    reason: str


class TrajectoryFindingModel(BaseModel):
    kind: str
    detail: str


class TrajectoryEvaluationModel(BaseModel):
    score: float
    outcome: str
    summary: str
    gain_rate: float
    efficiency: float
    decisions: list[TrajectoryDecisionModel]
    findings: list[TrajectoryFindingModel]


class StrategyStatModel(BaseModel):
    strategy: str
    runs: int
    useful: int
    useful_rate: float


class PolicyDecisionModel(BaseModel):
    action: str
    strategy: str = ""
    reason: str


class RetrievalChoiceModel(BaseModel):
    top_k: int
    strategy: str = ""
    reason: str
    basis: str = "default"


class CounterfactualAlternativeModel(BaseModel):
    strategy: str = ""
    question: str = ""
    expected: float = 0
    basis: str = ""


class CounterfactualForkModel(BaseModel):
    taken_strategy: str = ""
    taken_question: str = ""
    useful: bool = False
    actual: float = 0
    verdict: str
    summary: str
    alternatives: list[CounterfactualAlternativeModel] = []


class CounterfactualModel(BaseModel):
    verdict: str
    summary: str
    forks: list[CounterfactualForkModel] = []


class ResearchAgentModel(BaseModel):
    id: str
    name: str
    task: str = ""
    searches: int = 0
    useful: int = 0


class HandoffModel(BaseModel):
    from_agent: str
    to_agent: str
    strategy: str = ""
    question: str = ""


class AggregatorNoteModel(BaseModel):
    name: str
    decision: str
    reason: str


class CollaborationModel(BaseModel):
    agents: list[ResearchAgentModel]
    handoffs: list[HandoffModel]
    aggregator: AggregatorNoteModel
    summary: str


class BenchmarkModel(BaseModel):
    runs: int
    latest_score: float
    mean_score: float
    delta: float
    success_rate: float
    trend: str
    summary: str
    improvement: str


class LearningModel(BaseModel):
    strategies: list[StrategyStatModel]
    best: str = ""
    summary: str
    policy: PolicyDecisionModel
    retrieval: RetrievalChoiceModel


class FailurePredictionModel(BaseModel):
    label: str
    probability: float
    at_search: int = 0
    warned_at_search: int | None = None
    reasons: list[str]
    basis: str
    history_runs: int = 0


class AdvancedNoteModel(BaseModel):
    component: str
    decision: str
    reason: str


class AdvancedModel(BaseModel):
    summary: str
    notes: list[AdvancedNoteModel]


class DiscoveryReviewModel(BaseModel):
    agent: str
    stance: str
    finding: str


class DiscoveryHypothesisModel(BaseModel):
    statement: str
    status: str


class DiscoveryConclusionModel(BaseModel):
    verdict: str
    text: str


class DiscoveryRecordModel(BaseModel):
    label: str
    detail: str


class DiscoveryPhaseModel(BaseModel):
    phase: int
    component: str
    title: str
    status: str
    summary: str
    records: list[DiscoveryRecordModel]


class SocietyNoteModel(BaseModel):
    component: str
    decision: str
    reason: str


class SocietyModel(BaseModel):
    summary: str
    notes: list[SocietyNoteModel]


class DiscoveryModel(BaseModel):
    summary: str
    hypothesis: DiscoveryHypothesisModel
    reviews: list[DiscoveryReviewModel]
    conclusion: DiscoveryConclusionModel
    phases: list[DiscoveryPhaseModel]
    society: SocietyModel


class AcceptanceCriterionModel(BaseModel):
    criterion: str
    result: str
    score: float
    reason: str


class AcceptanceModel(BaseModel):
    passed: int
    total: int
    met: bool
    summary: str
    criteria: list[AcceptanceCriterionModel]


class ProductionRecordModel(BaseModel):
    label: str
    detail: str


class ProductionAreaModel(BaseModel):
    priority: int
    component: str
    title: str
    status: str
    score: float
    summary: str
    records: list[ProductionRecordModel]


class ProductionModel(BaseModel):
    summary: str
    release: str
    revision: str = ""
    areas: list[ProductionAreaModel]


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
    evaluation: TrajectoryEvaluationModel
    prediction: FailurePredictionModel
    learning: LearningModel
    counterfactual: CounterfactualModel
    collaboration: CollaborationModel
    benchmark: BenchmarkModel
    advanced: AdvancedModel
    discovery: DiscoveryModel
    acceptance: AcceptanceModel
    production: ProductionModel


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
    retrieval = {"top_k": 4}

    def search(question: str, asset: str) -> dict:
        if asset not in {"NVDA", "TSLA", "GOLD"}:
            return {"asset": asset, "answer": "", "markers": [], "scene": None}
        if named_demo_asset(question) is None:
            scene = library_scene(
                services.generation.answer(question, int(retrieval["top_k"]), user_id=user_id),
                services.metadata,
            )
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
        scene = library_scene(
            services.generation.answer(question, int(retrieval["top_k"]), user_id=user_id),
            services.metadata,
        )
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
        memory=research_memory,
        retrieval=retrieval,
    )
