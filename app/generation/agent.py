"""Decide whether to search, retry with a shorter query, or ask for clarification."""

from app.generation.grounding import AnswerResult
from app.retrieval.queries import rewrite_query
from app.retrieval.text import content_tokens

CLARIFICATION_TEXT = "Which document or topic should I search?"
_VAGUE = {"help", "please", "something", "anything", "this", "that", "it"}
_OVERVIEW_INTENT = {
    "about",
    "contents",
    "describe",
    "explain",
    "list",
    "outline",
    "overview",
    "summarize",
    "summary",
}
_OVERVIEW = _OVERVIEW_INTENT | {
    "all",
    "app",
    "document",
    "entire",
    "file",
    "full",
    "give",
    "me",
    "pdf",
    "requirement",
    "requirements",
    "say",
    "scope",
    "show",
    "spec",
    "specification",
    "srs",
    "system",
    "tell",
    "what",
}


def needs_clarification(question: str) -> bool:
    """True when the question does not name a topic worth searching."""
    tokens = content_tokens(question)
    if not tokens:
        return True
    return all(token in _VAGUE for token in tokens)


def wants_section_summary(question: str) -> bool:
    """True when the asker wants a section explained, listed, or summarized."""
    tokens = set(content_tokens(question))
    if tokens & _OVERVIEW_INTENT or "breakdown" in tokens:
        return True
    folded = question.casefold()
    return folded.startswith("what are") or (
        "what" in tokens and "requirements" in tokens
    )


def is_document_overview(question: str) -> bool:
    """True when the asker wants a summary of the indexed file, not a specific fact."""
    tokens = content_tokens(question)
    if not tokens or not all(token in _OVERVIEW or token in _VAGUE for token in tokens):
        return False
    return any(token in _OVERVIEW_INTENT or token in {"what", "tell", "say"} for token in tokens)


def reformulate(question: str) -> str | None:
    """A shorter search to try when the first retrieval does not cover the question.

    Politeness is removed first. The keyword form is used when it differs from
    both the original question and that rewrite.
    """
    rewritten = rewrite_query(question)
    keywords = " ".join(content_tokens(rewritten))
    original = question.strip()
    if keywords and keywords.casefold() not in {original.casefold(), rewritten.casefold()}:
        return keywords
    if rewritten.casefold() != original.casefold():
        return rewritten
    return None


def clarification(question: str) -> AnswerResult:
    return AnswerResult(
        query=question.strip(),
        answer=CLARIFICATION_TEXT,
        sufficient_evidence=False,
        citations=[],
        passages=[],
        action="clarify",
    )
