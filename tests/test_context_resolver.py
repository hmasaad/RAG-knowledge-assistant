"""Section ids and spans restore neighboring bullets before verification."""

from app.research.claim_verifier import verify_resolved
from app.research.context_resolver import resolve_context
from app.research.contracts import ActionRequest, EvidenceRef, ResearchState
from app.research.ledger import EvidenceLedger, SourceSpan
from app.research.synthesis_guard import preserve_supported
from app.research.verify_actions import VerifyAction


def test_resolver_rebuilds_the_mvp_section_from_one_bullet() -> None:
    heading, first, second = _mvp_pool()
    resolved = resolve_context(second, [heading, first, second])

    assert resolved.section_id == "section_11"
    assert {item.evidence_id for item in resolved.members} == {"h", "b1", "b2"}
    assert "MVP Scope" in resolved.inherited
    assert "Salon discovery" in resolved.combined_text
    assert "Customer registration" in resolved.combined_text


def test_ledger_records_spans_and_independent_scores() -> None:
    _, first, second = _mvp_pool()
    resolved = resolve_context(second, list(_mvp_pool()))
    entry = verify_resolved("what is MVP scope?", resolved, list(_mvp_pool()))
    ledger = EvidenceLedger()
    ledger.record(entry)

    assert entry.verdict == "supported"
    assert entry.relevance == 1.0
    assert entry.contextual_meaning is not None and entry.contextual_meaning >= 0.6
    assert entry.span == SourceSpan(
        document_id="doc",
        evidence_id="b2",
        section_id="section_11",
        heading="11. MVP Scope",
        page=11,
        start_char=80,
        end_char=110,
    )
    assert ledger.supported_ids("what is MVP scope?") == ("b2",)
    wifi = verify_resolved("what is the office wifi password", resolved, list(_mvp_pool()))
    assert wifi.verdict != "supported"


def test_reverify_consumes_resolver_and_writes_the_ledger() -> None:
    state = ResearchState(evidence=list(_mvp_pool()))
    outcome = VerifyAction().run(ActionRequest("reverify", claim="what is MVP scope?"), state)

    assert outcome.verdicts[0].verdict == "supported"
    assert set(outcome.verdicts[0].evidence_ids) == {"h", "b1", "b2"}
    assert len(outcome.ledger) == 3
    assert {row["span"]["section_id"] for row in outcome.ledger} == {"section_11"}
    assert all("inherited" in row for row in outcome.ledger)


def test_synthesis_retains_every_ledger_supported_span() -> None:
    state = ResearchState(evidence=list(_mvp_pool()))
    outcome = VerifyAction().run(ActionRequest("reverify", claim="what is MVP scope?"), state)
    state.verdicts.extend(outcome.verdicts)
    state.ledger.extend(outcome.ledger)

    before = [item.text for item in state.evidence]
    report = preserve_supported("what is MVP scope?", state)

    assert "Salon discovery" in report
    assert "Customer registration" in report
    assert report.count("[") >= 2
    assert [item.text for item in state.evidence] == before


def _mvp_pool() -> tuple[EvidenceRef, EvidenceRef, EvidenceRef]:
    heading = EvidenceRef(
        "h",
        "doc",
        "salon.pdf",
        "11. MVP Scope",
        heading="11. MVP Scope",
        section_title="MVP Scope",
        section_id="section_11",
        section_number="11",
        start_char=0,
        end_char=14,
        page=11,
        chunk_index=0,
    )
    first = EvidenceRef(
        "b1",
        "doc",
        "salon.pdf",
        "Customer registration/login and profile.",
        heading="11. MVP Scope",
        section_title="MVP Scope",
        section_id="section_11",
        section_number="11",
        start_char=14,
        end_char=56,
        page=11,
        chunk_index=1,
    )
    second = EvidenceRef(
        "b2",
        "doc",
        "salon.pdf",
        "Salon discovery and search.",
        heading="11. MVP Scope",
        section_title="MVP Scope",
        section_id="section_11",
        section_number="11",
        start_char=80,
        end_char=110,
        page=11,
        chunk_index=2,
    )
    return heading, first, second
