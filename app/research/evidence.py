"""Evidence collected while a research plan is executing."""

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from app.research.rag import RagPassage


@dataclass(frozen=True)
class SourceReference:
    document_id: str
    chunk_id: str
    filename: str
    content_type: str
    heading: str | None
    page: int | None
    start_char: int
    end_char: int


@dataclass(frozen=True)
class EvidenceRecord:
    evidence_id: str
    claim_id: str
    claim: str
    passage_id: str
    source: SourceReference
    score: float
    created_at: str
    verdict: str = "supported"
    support: float | None = None


@dataclass(frozen=True)
class ClaimLink:
    """A claim and the passage that supports it."""

    claim_id: str
    passage_id: str
    evidence_id: str


class EvidenceStore:
    """Record claims and the passages they cite for one research run."""

    def __init__(self) -> None:
        self.records: list[EvidenceRecord] = []
        self.links: list[ClaimLink] = []

    def record_claim(
        self,
        claim: str,
        passage: RagPassage,
        *,
        verdict: str = "supported",
        support: float | None = None,
    ) -> EvidenceRecord | None:
        text = claim.strip()
        if not text:
            return None
        source = passage.source
        evidence_id = uuid.uuid4().hex[:16]
        claim_id = uuid.uuid4().hex[:16]
        record = EvidenceRecord(
            evidence_id=evidence_id,
            claim_id=claim_id,
            claim=text,
            passage_id=source.chunk_id,
            source=SourceReference(
                document_id=source.document_id,
                chunk_id=source.chunk_id,
                filename=source.filename,
                content_type=source.content_type,
                heading=source.heading,
                page=source.page,
                start_char=source.start_char,
                end_char=source.end_char,
            ),
            score=passage.score,
            created_at=datetime.now(timezone.utc).isoformat(),
            verdict=verdict,
            support=support,
        )
        self.records.append(record)
        self.links.append(
            ClaimLink(claim_id=claim_id, passage_id=source.chunk_id, evidence_id=evidence_id)
        )
        return record
