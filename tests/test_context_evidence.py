"""Inherited section context can support a bullet that does not repeat the topic."""

from app.research.context_evidence import (
    bind_context,
    should_accept_evidence,
    support_with_context,
)
from app.research.verification import assess_claim
from app.retrieval.search import RankedPassage, SourceMetadata


def test_mvp_bullet_is_verified_through_parent_heading() -> None:
    passage = _passage(
        "Salon discovery and search.",
        heading="11. MVP Scope",
        section_title="MVP Scope",
        section_number="11",
    )
    bound = bind_context(passage)

    assert "MVP Scope" in bound.inherited
    assert "Salon discovery" in bound.body
    assert support_with_context("what is MVP scope?", passage) >= 0.6
    assert assess_claim("what is MVP scope?", passage).verdict == "supported"


def test_heading_does_not_invent_unrelated_facts() -> None:
    passage = _passage(
        "Salon discovery and search.",
        heading="11. MVP Scope",
        section_title="MVP Scope",
        section_number="11",
    )

    assert not should_accept_evidence("what is the office wifi password", passage)
    assert not should_accept_evidence("The retry limit is three attempts.", passage)
    assert assess_claim("The retry limit is three attempts.", passage).verdict == "unsupported"


def test_heading_only_chunk_is_not_evidence() -> None:
    passage = _passage("MVP", heading="11. MVP Scope", section_title="MVP Scope")

    assert not should_accept_evidence("what is MVP scope?", passage)


def _passage(
    text: str,
    *,
    heading: str,
    section_title: str,
    section_number: str = "",
) -> RankedPassage:
    return RankedPassage(
        rank=1,
        text=text,
        score=1.0,
        vector_score=1.0,
        lexical_score=1.0,
        answers_question=True,
        matched_terms=(),
        source=SourceMetadata(
            chunk_id="c1",
            document_id="doc",
            filename="salon.pdf",
            content_type="application/pdf",
            chunk_index=0,
            start_char=0,
            end_char=len(text),
            page=11,
            heading=heading,
            section_id="section_11",
            section_title=section_title,
            section_number=section_number,
        ),
    )
