"""HTTP schemas for documents and search."""

from pydantic import BaseModel, Field, field_validator

from app.ingestion.models import DocumentRecord, VersionRecord


class DocumentResponse(BaseModel):
    id: str
    filename: str
    content_type: str
    checksum: str
    status: str
    chunk_count: int
    byte_size: int
    created_at: str
    error: str | None = None
    already_indexed: bool = False
    owner: str = "local"
    visibility: str = "private"
    version: int = 1
    indexed_version: int = 0
    updated_at: str = ""

    @classmethod
    def from_record(
        cls,
        record: DocumentRecord,
        *,
        already_indexed: bool = False,
    ) -> "DocumentResponse":
        return cls(
            id=record.id,
            filename=record.filename,
            content_type=record.content_type,
            checksum=record.checksum,
            status=record.status,
            chunk_count=record.chunk_count,
            byte_size=record.byte_size,
            created_at=record.created_at,
            error=record.error,
            already_indexed=already_indexed,
            owner=record.owner,
            visibility=record.visibility,
            version=record.version,
            indexed_version=record.indexed_version,
            updated_at=record.updated_at or record.created_at,
        )


class ChunkResponse(BaseModel):
    id: str
    index: int
    text: str
    start_char: int
    end_char: int
    page: int | None
    heading: str | None


class DocumentDetailResponse(DocumentResponse):
    chunks: list[ChunkResponse]


class DocumentListResponse(BaseModel):
    documents: list[DocumentResponse]


class VersionResponse(BaseModel):
    version: int
    filename: str
    content_type: str
    checksum: str
    byte_size: int
    chunk_count: int
    status: str
    created_at: str
    error: str | None = None

    @classmethod
    def from_record(cls, record: VersionRecord) -> "VersionResponse":
        return cls(
            version=record.version,
            filename=record.filename,
            content_type=record.content_type,
            checksum=record.checksum,
            byte_size=record.byte_size,
            chunk_count=record.chunk_count,
            status=record.status,
            created_at=record.created_at,
            error=record.error,
        )


class VersionListResponse(BaseModel):
    versions: list[VersionResponse]


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=20)

    @field_validator("query")
    @classmethod
    def query_not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Query must not be empty.")
        return stripped


class SearchHitResponse(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    content_type: str
    text: str
    score: float
    chunk_index: int
    start_char: int
    end_char: int
    page: int | None
    heading: str | None


class SearchResponse(BaseModel):
    query: str
    hits: list[SearchHitResponse]


class SourceResponse(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    content_type: str
    chunk_index: int
    start_char: int
    end_char: int
    page: int | None
    heading: str | None


class PassageResponse(BaseModel):
    rank: int
    text: str
    score: float
    vector_score: float
    lexical_score: float
    answers_question: bool
    matched_terms: list[str]
    instruction_detected: bool = False
    rerank_score: float = 0.0
    source: SourceResponse


class RetrieveResponse(BaseModel):
    query: str
    sufficient_evidence: bool
    conflict: bool = False
    inspection: str
    passages: list[PassageResponse]
    queries: list[str] = []


class AnswerRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=4, ge=1, le=20)
    halt_after: int | None = Field(default=None, ge=1, le=32)
    priority: int = Field(default=0, ge=0, le=9)

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Question must not be empty.")
        return stripped


class CitationResponse(BaseModel):
    marker: str
    chunk_id: str
    document_id: str
    filename: str
    content_type: str
    heading: str | None
    page: int | None
    quote: str


class AnswerResponse(BaseModel):
    question: str
    answer: str
    sufficient_evidence: bool
    conflict: bool = False
    citations: list[CitationResponse]
    passages: list[PassageResponse]
    action: str = "answer"
    trace_id: str = ""


class ResearchSubquestionResponse(BaseModel):
    question: str
    tool: str
    status: str
    evidence: str
    completion: str = ""
    complete: bool = False


class EvidenceResponse(BaseModel):
    evidence_id: str
    claim_id: str
    claim: str
    passage_id: str
    document_id: str
    filename: str
    content_type: str
    heading: str | None = None
    page: int | None = None
    start_char: int
    end_char: int
    score: float
    created_at: str
    verdict: str = "supported"
    support: float | None = None


class ClaimLinkResponse(BaseModel):
    claim_id: str
    passage_id: str
    evidence_id: str


class ClaimJudgeRequest(BaseModel):
    claim: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=4, ge=1, le=20)

    @field_validator("claim")
    @classmethod
    def claim_not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Claim must not be empty.")
        return stripped


class JudgeEvidenceResponse(BaseModel):
    evidence_id: str
    document_id: str
    filename: str
    text: str
    role: str
    score: float
    start_char: int
    end_char: int
    heading: str | None = None
    page: int | None = None
    instruction_detected: bool = False


class GraphNodeResponse(BaseModel):
    id: str
    kind: str
    label: str
    document_id: str = ""
    version: int = 0
    filename: str = ""
    content_type: str = ""
    checksum: str = ""
    chunk_id: str = ""
    heading: str | None = None
    page: int | None = None
    start_char: int = 0
    end_char: int = 0


class GraphEdgeResponse(BaseModel):
    source: str
    relation: str
    target: str


class EvidenceGraphResponse(BaseModel):
    nodes: list[GraphNodeResponse] = []
    edges: list[GraphEdgeResponse] = []


class ClaimJudgeResponse(BaseModel):
    claim: str
    verdict: str
    evidence_ids: list[str]
    support_ids: list[str]
    oppose_ids: list[str]
    contradiction: bool
    reason: str
    evidence: list[JudgeEvidenceResponse] = []
    accepted: bool = True
    graph: EvidenceGraphResponse = Field(default_factory=EvidenceGraphResponse)


class ConflictResponse(BaseModel):
    summary: str
    left_filename: str
    right_filename: str
    left_values: list[int]
    right_values: list[int]
    left_claim_id: str = ""
    right_claim_id: str = ""


class PlanTaskResponse(BaseModel):
    id: str
    action: str
    depends_on: list[str] = []
    query: str = ""
    claim: str = ""
    status: str = "pending"
    agent_id: str = ""
    role: str = ""


class ResearchPlanResponse(BaseModel):
    question: str
    version: int
    reason: str
    revisions: list[str] = []
    tools: list[str] = []
    budget: int = 16
    actions: list[str] = []
    tasks: list[PlanTaskResponse]


class EvidenceVerdictRequest(BaseModel):
    claim: str
    verdict: str
    evidence_ids: list[str] = []


class EvidencePlanRequest(BaseModel):
    gaps: list[str] = []
    contradictions: list[str] = []
    verdicts: list[EvidenceVerdictRequest] = []
    pending_actions: list[str] = []
    remaining_budget: int | None = None


class ValueCandidateResponse(BaseModel):
    action: str
    kind: str
    target: str
    gain: float
    cost: int
    voi: float
    calibrated_voi: float = 0.0
    confidence: float = 1.0
    reliability: float = 1.0
    mandatory: bool = False
    blocked_by: str = ""
    eig: float = 0.0
    reason: str = ""


class EvidencePlanResponse(BaseModel):
    action: str
    verb: str
    reason: str
    targets: list[str] = []
    severity: str = "none"
    quality: float = 0.0
    voi: float = 0.0
    cost: int = 0
    kind: str = ""
    candidates: list[ValueCandidateResponse] = []
    signals: dict = {}


class ValueSelectionResponse(BaseModel):
    selected: ValueCandidateResponse | None = None
    remaining_budget: int = 16
    candidates: list[ValueCandidateResponse] = []


class SequentialDesignResponse(BaseModel):
    stop: bool
    reason: str
    beliefs: dict = {}
    sequence: list[dict] = []
    ranked: list[ValueCandidateResponse] = []


class AgentAssignmentResponse(BaseModel):
    agent_id: str
    role: str
    query: str
    tool: str
    action: str
    depends_on: list[str] = []


class AgentFindingResponse(BaseModel):
    agent_id: str
    role: str
    claim: str
    verdict: str
    source: str = ""
    values: list[int] = []
    evidence_ids: list[str] = []


class ConflictResolutionResponse(BaseModel):
    decision: str
    summary: str
    kept_claims: list[str] = []
    dropped_claims: list[str] = []
    left_agent: str = ""
    right_agent: str = ""


class CoordinationResponse(BaseModel):
    question: str
    assignments: list[AgentAssignmentResponse]
    findings: list[AgentFindingResponse] = []
    resolutions: list[ConflictResolutionResponse] = []
    consensus: list[str] = []
    resolved: bool = True


class CoordinateRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    findings: list[AgentFindingResponse] = []

    @field_validator("question")
    @classmethod
    def coordinate_question_not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Question must not be empty.")
        return stripped


class StrategyObserveRequest(BaseModel):
    situation: str = Field(min_length=1, max_length=32)
    action: str = Field(min_length=1, max_length=64)
    reward: float = Field(ge=0, le=1)


class StrategyRecommendRequest(BaseModel):
    situation: str = Field(min_length=1, max_length=32)
    fallback: str = "stop"


class StrategyStatResponse(BaseModel):
    situation: str
    action: str
    wins: int
    losses: int
    last_reward: float = 0
    score: float = 0
    samples: int = 0


class StrategyRecommendResponse(BaseModel):
    situation: str
    action: str
    fallback: str = "stop"


class ResearchResponse(BaseModel):
    question: str
    report: str
    sufficient_evidence: bool
    conflict: bool = False
    tools: list[str]
    subquestions: list[ResearchSubquestionResponse]
    contradictions: list[str]
    gaps: list[str]
    catalog: list[str]
    citations: list[CitationResponse]
    passages: list[PassageResponse]
    evidence: list[EvidenceResponse] = []
    links: list[ClaimLinkResponse] = []
    conflicts: list[ConflictResponse] = []
    graph: EvidenceGraphResponse = Field(default_factory=EvidenceGraphResponse)
    decision: str = "report"
    decisions: list[str] = []
    remaining_budget: int = 0
    action: str
    trace_id: str = ""
    session_id: str = ""
    status: str = "completed"
    side_effects: list["SideEffectResponse"] = []
    quality_action: str = "stop"
    quality_reason: str = ""
    quality_failures: list[str] = []
    quality_claims: list["QualityClaimResponse"] = []
    quality_confident: bool = True
    coordination: CoordinationResponse | None = None


class QualityClaimResponse(BaseModel):
    claim: str
    marker: int
    citation_exists: bool
    supports: bool
    support_ratio: float
    filename: str = ""
    failures: list[str] = []


class SideEffectResponse(BaseModel):
    action: str
    fingerprint: str = ""
    runs: int = 0


class ResearchSessionListItem(BaseModel):
    session_id: str
    question: str
    status: str
    action: str = ""
    created_at: str = ""
    updated_at: str = ""
    priority: int = 0


class ResearchSessionListResponse(BaseModel):
    sessions: list[ResearchSessionListItem]


class SessionAuditEventResponse(BaseModel):
    action: str
    query: str = ""
    claim: str = ""
    fingerprint: str = ""
    created_at: str = ""
    ok: bool = True
    error: str = ""
    evidence: list[dict] = []
    verdicts: list[dict] = []
    gaps: list[str] = []
    contradictions: list[str] = []
    cost: int = 0


class SessionAuditResponse(BaseModel):
    session_id: str
    question: str
    status: str
    plan: list[dict] = []
    events: list[SessionAuditEventResponse] = []
    evidence: list[dict] = []
    verdicts: list[dict] = []
    contradictions: list[str] = []
    gaps: list[str] = []
    voi: list[dict] = []
