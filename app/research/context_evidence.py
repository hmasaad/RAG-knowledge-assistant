"""Context-preserving evidence verification.

This layer sits between retrieval and synthesis. It scores a claim against
the passage body plus inherited section metadata, so a bullet such as
"Salon discovery and search" can support "what is MVP scope?" when it
belongs to ``11. MVP Scope``. Heading membership does not waive a body
check or invent facts that are not in the source text.
"""

from dataclasses import dataclass

from app.generation.grounding import SUPPORT_THRESHOLD, claim_support
from app.ingestion.context import situate
from app.retrieval.conflicts import numeric_facts
from app.retrieval.search import RankedPassage
from app.retrieval.text import content_tokens, focus_tokens

_MIN_BODY = 12


@dataclass(frozen=True)
class ContextualEvidence:
    """One retrieved passage with the section context used to verify it."""

    body: str
    inherited: str
    combined: str
    used_inherited: bool


def inherited_section_context(passage: RankedPassage) -> str:
    """Document and parent heading that this chunk inherited at index time."""
    source = passage.source
    title = source.section_title or ""
    heading = source.heading
    prefix = situate(source.filename, heading, section_title=title)
    extra = []
    if source.section_number and title:
        extra.append(f"Section {source.section_number}: {title}.")
    elif title and (not heading or title.casefold() not in heading.casefold()):
        extra.append(f"Parent heading: {title}.")
    return " ".join(part for part in (prefix, *extra) if part).strip()


def bind_context(passage: RankedPassage) -> ContextualEvidence:
    """Attach inherited section context to the stored source text."""
    body = (passage.model_text() or passage.text or "").strip()
    inherited = inherited_section_context(passage)
    combined = f"{inherited}\n\n{body}".strip() if inherited else body
    return ContextualEvidence(
        body=body,
        inherited=inherited,
        combined=combined,
        used_inherited=bool(inherited) and bool(body),
    )


def section_matches_question(section_title: str, question: str) -> bool:
    """True when stored section metadata names the same topic as the question."""
    title = (section_title or "").strip()
    if not title:
        return False
    asked = set(focus_tokens(question))
    titled = set(content_tokens(title))
    if not asked or not titled:
        return False
    return bool(asked & titled)


def has_evidence_body(text: str) -> bool:
    """True when the chunk still contains usable source text, not only a heading."""
    compact = " ".join((text or "").split())
    return len(compact) >= _MIN_BODY


def support_with_context(claim: str, passage: RankedPassage) -> float | None:
    """Share of claim words found in the body plus inherited section context.

    Numbers stated in the claim must appear in the body. A parent heading
    may supply topic words such as ``MVP``, but it cannot invent a fact.
    """
    bound = bind_context(passage)
    if not has_evidence_body(bound.body):
        return None
    body_ratio = claim_support(claim, bound.body)
    full_ratio = claim_support(claim, bound.combined)
    if numeric_facts(claim) and not numeric_facts(bound.body):
        return body_ratio
    return full_ratio if full_ratio is not None else body_ratio


def should_accept_evidence(claim: str, passage: RankedPassage) -> bool:
    """Keep a passage only when contextual support clears the citation bar."""
    ratio = support_with_context(claim, passage)
    return ratio is not None and ratio >= SUPPORT_THRESHOLD
