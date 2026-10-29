"""Reconstruct a chunk's section and neighbors from stored relationships.

Retrieval may return one bullet. This resolver uses section ids, parent
headings, page numbers, and character spans to restore the surrounding
evidence before a claim is judged.
"""

from dataclasses import dataclass

from app.ingestion.context import situate
from app.research.contracts import EvidenceRef


@dataclass(frozen=True)
class ResolvedContext:
    """The seed chunk plus the section it belongs to."""

    seed: EvidenceRef
    section_id: str
    heading: str
    section_title: str
    members: tuple[EvidenceRef, ...]
    neighbors: tuple[EvidenceRef, ...]
    inherited: str
    combined_text: str


def resolve_context(seed: EvidenceRef, pool: list[EvidenceRef]) -> ResolvedContext:
    """Rebuild section membership and adjacent spans from indexed metadata."""
    section_id = seed.section_id.strip()
    members = _section_members(seed, pool, section_id)
    heading, title = _section_labels(members)
    inherited = _inherited(seed, heading, title)
    neighbors = _neighbors(seed, pool, {item.evidence_id for item in members})
    bodies = [item.text.strip() for item in members if item.text.strip()]
    combined = inherited
    if bodies:
        combined = f"{inherited}\n\n" + "\n".join(bodies) if inherited else "\n".join(bodies)
    return ResolvedContext(
        seed=seed,
        section_id=section_id,
        heading=heading,
        section_title=title,
        members=members,
        neighbors=neighbors,
        inherited=inherited,
        combined_text=combined.strip(),
    )


def _section_members(seed: EvidenceRef, pool: list[EvidenceRef], section_id: str) -> tuple[EvidenceRef, ...]:
    if not section_id:
        return (seed,)
    found = [
        item
        for item in pool
        if item.document_id == seed.document_id and item.section_id == section_id
    ]
    if seed not in found:
        found.append(seed)
    found.sort(key=lambda item: (item.start_char, item.chunk_index, item.evidence_id))
    return tuple(found)


def _section_labels(members: tuple[EvidenceRef, ...]) -> tuple[str, str]:
    heading = next((item.heading for item in members if item.heading), "")
    title = next((item.section_title for item in members if item.section_title), heading)
    return heading, title


def _inherited(seed: EvidenceRef, heading: str, title: str) -> str:
    prefix = situate(seed.source, heading or None, section_title=title)
    if seed.section_number and title:
        return f"{prefix} Section {seed.section_number}: {title}.".strip()
    number = next((item.section_number for item in (seed,) if item.section_number), "")
    if number and title:
        return f"{prefix} Section {number}: {title}.".strip()
    return prefix


def _neighbors(
    seed: EvidenceRef,
    pool: list[EvidenceRef],
    member_ids: set[str],
) -> tuple[EvidenceRef, ...]:
    same = [item for item in pool if item.document_id == seed.document_id]
    same.sort(key=lambda item: (item.start_char, item.chunk_index, item.evidence_id))
    index = next((i for i, item in enumerate(same) if item.evidence_id == seed.evidence_id), -1)
    if index < 0:
        return ()
    adjacent: list[EvidenceRef] = []
    if index > 0 and same[index - 1].evidence_id not in member_ids:
        adjacent.append(same[index - 1])
    if index + 1 < len(same) and same[index + 1].evidence_id not in member_ids:
        adjacent.append(same[index + 1])
    return tuple(adjacent)
