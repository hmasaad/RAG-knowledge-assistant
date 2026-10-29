"""Score relevance, contextual meaning, support, and contradictions separately."""

from app.generation.grounding import SUPPORT_THRESHOLD, claim_support
from app.research.context_evidence import (
    has_evidence_body,
    section_matches_question,
    support_with_context,
)
from app.research.context_resolver import ResolvedContext
from app.research.contracts import EvidenceRef
from app.research.ledger import LedgerEntry, SourceSpan
from app.retrieval.conflicts import numeric_facts
from app.retrieval.search import RankedPassage, SourceMetadata
from app.retrieval.text import content_tokens, focus_tokens


def verify_resolved(claim: str, resolved: ResolvedContext, pool: list[EvidenceRef]) -> LedgerEntry:
    """Judge one seed chunk after its section context has been restored."""
    passage = _as_passage(resolved.seed, resolved)
    relevance = _relevance(claim, resolved, passage)
    contextual = support_with_context(claim, passage)
    support = claim_support(claim, resolved.seed.text)
    contradicted = _contradicted(claim, resolved.seed, pool)
    verdict = _verdict(relevance, contextual, support, contradicted, resolved.seed.text)
    return LedgerEntry(
        claim=claim.strip(),
        evidence_id=resolved.seed.evidence_id,
        verdict=verdict,
        relevance=relevance,
        contextual_meaning=contextual,
        support=support,
        contradicted=contradicted,
        inherited=resolved.inherited,
        span=SourceSpan(
            document_id=resolved.seed.document_id,
            evidence_id=resolved.seed.evidence_id,
            section_id=resolved.section_id,
            heading=resolved.heading or resolved.section_title,
            page=resolved.seed.page,
            start_char=resolved.seed.start_char,
            end_char=resolved.seed.end_char or resolved.seed.start_char + len(resolved.seed.text),
        ),
    )


def _relevance(claim: str, resolved: ResolvedContext, passage: RankedPassage) -> float | None:
    title = resolved.section_title or resolved.heading
    if section_matches_question(title, claim):
        return 1.0
    asked = set(focus_tokens(claim))
    if not asked:
        return None
    haystack = set(content_tokens(resolved.combined_text or passage.text))
    return sum(1 for token in asked if token in haystack) / len(asked)


def _contradicted(claim: str, seed: EvidenceRef, pool: list[EvidenceRef]) -> bool:
    values = numeric_facts(claim)
    body_values = numeric_facts(seed.text)
    if values and body_values and not values & body_values:
        return True
    return False


def _verdict(
    relevance: float | None,
    contextual: float | None,
    support: float | None,
    contradicted: bool,
    body: str,
) -> str:
    if not has_evidence_body(body):
        return "unverifiable"
    if contradicted:
        return "unsupported"
    meaning = contextual if contextual is not None else support
    if meaning is None:
        return "unverifiable"
    if meaning >= SUPPORT_THRESHOLD and (relevance or 0) > 0:
        return "supported"
    if support is not None and support >= SUPPORT_THRESHOLD:
        return "supported"
    if (relevance or 0) >= 1.0 and (contextual or 0) > 0:
        return "supported"
    return "unsupported"


def _as_passage(item: EvidenceRef, resolved: ResolvedContext) -> RankedPassage:
    heading = resolved.heading or item.heading or None
    title = resolved.section_title or item.section_title
    return RankedPassage(
        rank=1,
        text=item.text,
        score=1.0,
        vector_score=0.0,
        lexical_score=0.0,
        answers_question=True,
        matched_terms=(),
        source=SourceMetadata(
            chunk_id=item.evidence_id,
            document_id=item.document_id,
            filename=item.source,
            content_type="text/markdown",
            chunk_index=item.chunk_index,
            start_char=item.start_char,
            end_char=item.end_char or len(item.text),
            page=item.page,
            heading=heading,
            section_id=resolved.section_id or item.section_id,
            section_title=title,
            section_number=item.section_number,
        ),
    )
