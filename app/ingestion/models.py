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
    section_id: str = ""
    section_title: str = ""
    section_number: str = ""
    parent_section: str = ""
    chunk_type: str = "section"
    bullet_index: int = -1


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
    version: int = 1
    parent_id: str = ""
    parent_text: str = ""
    parent_index: int = -1
    parent_start: int = -1
    parent_end: int = -1
    context: str = ""
    section_id: str = ""
    section_title: str = ""
    section_number: str = ""
    parent_section: str = ""
    chunk_type: str = ""
    bullet_index: int = -1


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
    owner: str = "local"
    visibility: str = "private"
    version: int = 1
    indexed_version: int = 0
    updated_at: str = ""


@dataclass(frozen=True)
class VersionRecord:
    document_id: str
    version: int
    filename: str
    content_type: str
    checksum: str
    byte_size: int
    chunk_count: int
    status: str
    created_at: str
    error: str | None = None


@dataclass(frozen=True)
class JobRecord:
    id: str
    document_id: str
    version: int
    status: str
    attempts: int
