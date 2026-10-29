"""Search small slices and return the larger chunk they came from."""

from app.ingestion.chunking import build_chunks
from app.ingestion.models import Chunk


def split_children(
    chunks: list[Chunk],
    *,
    child_chunk_size: int,
    overlap: int,
) -> list[Chunk]:
    """Replace a long chunk with the smaller slices used for search.

    Each child records the parent text. Retrieval expands a matching child
    back to that parent so the answer sees the whole section.
    """
    if child_chunk_size < 1:
        raise ValueError("child_chunk_size must be positive")
    produced: list[Chunk] = []
    child_overlap = min(overlap, max(0, child_chunk_size - 1))
    for parent in chunks:
        if len(parent.text) <= child_chunk_size:
            produced.append(parent)
            continue
        drafts = build_chunks(
            parent.text,
            content_type=parent.content_type,
            chunk_size=child_chunk_size,
            overlap=child_overlap,
            strategy="recursive",
        )
        if len(drafts) <= 1:
            produced.append(parent)
            continue
        parent_id = f"{parent.document_id}:{parent.version}:p{parent.index}"
        for offset, draft in enumerate(drafts):
            produced.append(
                Chunk(
                    id=f"{parent.id}:{offset}",
                    document_id=parent.document_id,
                    filename=parent.filename,
                    content_type=parent.content_type,
                    text=draft.text,
                    index=len(produced),
                    start_char=parent.start_char + draft.start,
                    end_char=parent.start_char + draft.end,
                    page=parent.page,
                    heading=parent.heading,
                    version=parent.version,
                    parent_id=parent_id,
                    parent_text=parent.text,
                    parent_index=parent.index,
                    parent_start=parent.start_char,
                    parent_end=parent.end_char,
                    context=parent.context,
                    section_id=parent.section_id,
                    section_title=parent.section_title,
                    section_number=parent.section_number,
                    parent_section=parent.parent_section,
                    chunk_type=parent.chunk_type or "requirement_bullet",
                    bullet_index=offset,
                )
            )
    return produced
