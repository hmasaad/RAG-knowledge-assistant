"""Decide whether to search, retry with a shorter query, or ask for clarification."""

from app.generation.grounding import AnswerResult
from app.retrieval.queries import rewrite_query
from app.retrieval.text import content_tokens

CLARIFICATION_TEXT = "Which document or topic should I search?"
_VAGUE = {"help", "please", "something", "anything", "this", "that", "it"}
_OVERVIEW = {
    "about",
    "contents",
    "describe",
    "document",
    "explain",
    "file",
    "overview",
    "pdf",
    "say",
    "summarize",
    "summary",
    "tell",
    "what",
}


def needs_clarification(question: str) -> bool:
    """True when the question does not name a topic worth searching."""
    tokens = content_tokens(question)
    if not tokens:
        return True
    return all(token in _VAGUE for token in tokens)


def is_document_overview(question: str) -> bool:
    """True when the asker wants a summary of the indexed file, not a specific fact."""
    tokens = content_tokens(question)
    return bool(tokens) and all(token in _OVERVIEW or token in _VAGUE for token in tokens)


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
