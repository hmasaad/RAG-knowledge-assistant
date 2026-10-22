"""HTTP schemas for documents and search."""

from pydantic import BaseModel, Field, field_validator

from app.ingestion.models import DocumentRecord


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
