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
    action: str
    trace_id: str = ""
