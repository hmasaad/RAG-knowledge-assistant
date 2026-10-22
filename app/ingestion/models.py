"""Records produced by ingestion and stored with the collection."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ParsedPage:
    """One page of extracted text. ``number`` is 1-based for PDFs."""

    number: int | None
    text: str


@dataclass(frozen=True)
class ParsedDocument:
    content_type: str
    pages: list[ParsedPage]


@dataclass(frozen=True)
class ChunkDraft:
    """A slice of cleaned page text before ids are assigned.

    ``start`` and ``end`` are offsets into that cleaned page, not the raw file.
    """

    text: str
    start: int
    end: int
    heading: str | None


@dataclass(frozen=True)
class Chunk:
    id: str
    document_id: str
    filename: str
    content_type: str
    text: str
    index: int
    start_char: int
    end_char: int
    page: int | None
    heading: str | None


@dataclass(frozen=True)
class DocumentRecord:
    id: str
    filename: str
    content_type: str
    checksum: str
    status: str
    chunk_count: int
    byte_size: int
    created_at: str
    error: str | None = None
