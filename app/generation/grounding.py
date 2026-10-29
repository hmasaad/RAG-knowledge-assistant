"""Turn a model reply into a cited answer, or the standard refusal."""

import re

from app.retrieval.search import RankedPassage
from app.retrieval.text import content_tokens, focus_tokens
from app.security.prompts import REFUSAL_TEXT

_MARKER = re.compile(r"\[(\d+)\]")
_QUOTE_LIMIT = 500
_BOILERPLATE = {"documents", "disagree", "document"}
SUPPORT_THRESHOLD = 0.6
_SUPPORT_THRESHOLD = SUPPORT_THRESHOLD


class Citation:
    def __init__(
        self,
        *,
        marker: str,
        chunk_id: str,
        document_id: str,
        filename: str,
        content_type: str,
        heading: str | None,
        page: int | None,
        quote: str,
    ) -> None:
        self.marker = marker
        self.chunk_id = chunk_id
        self.document_id = document_id
        self.filename = filename
        self.content_type = content_type
        self.heading = heading
        self.page = page
        self.quote = quote


class AnswerResult:
    def __init__(
        self,
        *,
        query: str,
        answer: str,
        sufficient_evidence: bool,
        citations: list[Citation],
        passages: list[RankedPassage],
        conflict: bool = False,
        action: str = "answer",
        trace_id: str = "",
    ) -> None:
        self.query = query
        self.answer = answer
        self.sufficient_evidence = sufficient_evidence
        self.citations = citations
        self.passages = passages
        self.conflict = conflict
        self.action = action
        self.trace_id = trace_id


def interpret_model_answer(
    question: str,
    model_text: str,
    passages: list[RankedPassage],
    *,
    conflict: bool = False,
) -> AnswerResult:
    """Keep a reply only when every citation is faithful to that passage."""
    text = (model_text or "").strip()
    if not text or not passages or is_refusal(text):
        return refusal(question, passages)

    seen: set[int] = set()
    numbers: list[int] = []
    for match in _MARKER.finditer(text):
        number = int(match.group(1))
        if number < 1 or number > len(passages) or number in seen:
            continue
        seen.add(number)
        numbers.append(number)
    if not numbers or not _citations_are_faithful(text, passages, numbers):
        return refusal(question, passages)

    citations = [_citation(number, passages[number - 1]) for number in numbers]
    return AnswerResult(
        query=question,
        answer=text,
        sufficient_evidence=True,
        citations=citations,
        passages=passages,
        conflict=conflict,
    )


def refusal(question: str, passages: list[RankedPassage]) -> AnswerResult:
    return AnswerResult(
        query=question,
        answer=REFUSAL_TEXT,
        sufficient_evidence=False,
        citations=[],
        passages=passages,
    )


def is_refusal(text: str) -> bool:
    normalized = " ".join(text.lower().split())
    return (
        "don't have enough evidence" in normalized
        or "do not have enough evidence" in normalized
    )


def claim_support(claim: str, passage: str) -> float | None:
    """Share of the claim's content words that appear in the passage.

    Short words and the conflict boilerplate are ignored. ``None`` means the
    claim has no content words left to check.
    """
    tokens = [
        token
        for token in focus_tokens(claim)
        if len(token) >= 4 and token not in _BOILERPLATE
    ]
    if not tokens:
        tokens = [
            token
            for token in content_tokens(claim)
            if len(token) >= 4 and token not in _BOILERPLATE
        ]
    if not tokens:
        return None
    haystack = set(content_tokens(passage))
    return sum(1 for token in tokens if token in haystack) / len(tokens)


def _citations_are_faithful(
    answer: str,
    passages: list[RankedPassage],
    numbers: list[int],
) -> bool:
    """Each cited claim must be mostly made of words from that passage."""
    supported = 0
    for claim, number in _segments(answer):
        if number not in numbers:
            continue
        ratio = claim_support(claim, passages[number - 1].model_text())
        if ratio is None:
            continue
        if ratio < _SUPPORT_THRESHOLD:
            return False
        supported += 1
    return supported > 0


def _segments(answer: str) -> list[tuple[str, int]]:
    segments: list[tuple[str, int]] = []
    previous = 0
    for match in _MARKER.finditer(answer):
        segments.append((answer[previous : match.start()], int(match.group(1))))
        previous = match.end()
    return segments


def _citation(number: int, passage: RankedPassage) -> Citation:
    source = passage.source
    return Citation(
        marker=f"[{number}]",
        chunk_id=source.chunk_id,
        document_id=source.document_id,
        filename=source.filename,
        content_type=source.content_type,
        heading=source.heading,
        page=source.page,
        quote=_quote(passage.model_text() or passage.text),
    )


def _quote(text: str) -> str:
    stripped = text.strip()
    if len(stripped) <= _QUOTE_LIMIT:
        return stripped
    return stripped[: _QUOTE_LIMIT - 1].rstrip() + "…"
