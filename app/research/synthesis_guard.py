"""Keep every ledger-supported span in the final report and check citations."""

from app.generation.answerers import _section_points, numbered_list
from app.generation.agent import wants_section_summary
from app.research.context_evidence import section_matches_question
from app.research.contracts import ResearchState
from app.research.ledger import EvidenceLedger, LedgerEntry
from app.security.content import neutralize_untrusted_text
from app.security.prompts import REFUSAL_TEXT


def preserve_supported(question: str, state: ResearchState, ledger: EvidenceLedger | None = None) -> str:
    """Quote all supported source spans. Do not drop a ledger-backed bullet."""
    book = ledger or EvidenceLedger.from_payload(getattr(state, "ledger", None))
    if not book.entries and getattr(state, "ledger", None):
        book = EvidenceLedger.from_payload(state.ledger)
    by_id = {item.evidence_id: item for item in state.evidence}
    supported_ids = _supported_ids(state, book, question, by_id)
    ordered: list[tuple[str, int]] = []
    used: set[str] = set()
    limit = 16 if wants_section_summary(question) else 8
    marker = 0
    for evidence_id in supported_ids:
        passage = by_id.get(evidence_id)
        if passage is None:
            continue
        marker += 1
        inherited = " ".join(part for part in (passage.heading, passage.section_title) if part)
        body = neutralize_untrusted_text(passage.text).text or ""
        points = _section_points(body, (), require_topic=False, inherited=inherited)
        if not points and body.strip():
            points = [body.strip()]
        for point in points:
            key = point.casefold()
            if key in used:
                continue
            used.add(key)
            ordered.append((point, marker))
            if len(ordered) >= limit:
                break
        if len(ordered) >= limit:
            break
    if not ordered:
        return REFUSAL_TEXT
    text = numbered_list(ordered)
    if state.contradictions:
        text = f"{text}\nThe documents disagree."
    return _require_citations(text, ordered)


def _supported_ids(
    state: ResearchState,
    ledger: EvidenceLedger,
    question: str,
    by_id: dict | None = None,
) -> list[str]:
    ids = list(ledger.supported_ids(question)) or list(ledger.supported_ids())
    if not ids:
        seen: list[str] = []
        for verdict in state.verdicts:
            if verdict.verdict != "supported":
                continue
            for evidence_id in verdict.evidence_ids:
                if evidence_id not in seen:
                    seen.append(evidence_id)
        ids = seen
    known = by_id or {item.evidence_id: item for item in state.evidence}
    on_topic = [
        evidence_id
        for evidence_id in ids
        if _on_topic(known.get(evidence_id), question)
    ]
    return on_topic or ids


def _on_topic(passage, question: str) -> bool:
    if passage is None:
        return False
    heading = " ".join(part for part in (passage.heading, passage.section_title) if part)
    if section_matches_question(heading, question):
        return True
    from app.retrieval.text import content_tokens, focus_tokens

    asked = set(focus_tokens(question))
    return bool(asked) and bool(asked & set(content_tokens(passage.text + " " + heading)))


def _require_citations(text: str, ordered: list[tuple[str, int]]) -> str:
    if not ordered:
        return REFUSAL_TEXT
    if any(f"[{marker}]" not in text for _, marker in ordered):
        return REFUSAL_TEXT
    return text


def ledger_from_state(state: ResearchState) -> list[LedgerEntry]:
    return EvidenceLedger.from_payload(getattr(state, "ledger", None)).entries
