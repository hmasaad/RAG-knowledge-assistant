"""Score each claim against its passage and compare claims that disagree."""

import re
from dataclasses import dataclass
from itertools import combinations

from app.generation.grounding import SUPPORT_THRESHOLD, AnswerResult
from app.research.context_evidence import (
    section_matches_question,
    should_accept_evidence,
    support_with_context,
)

__all__ = (
    "ClaimAssessment",
    "Contradiction",
    "assess_claim",
    "section_matches_question",
    "should_accept_evidence",
    "verify_answer",
    "find_contradictions",
    "passage_contradiction",
)
from app.retrieval.conflicts import numeric_facts
from app.retrieval.search import RankedPassage

_MARKER = re.compile(r"\[(\d+)\]")


@dataclass(frozen=True)
class ClaimAssessment:
    claim: str
    verdict: str
    support: float | None
    filename: str
    passage: RankedPassage
    values: tuple[int, ...]
    claim_id: str = ""


@dataclass(frozen=True)
class Contradiction:
    summary: str
    left_filename: str
    right_filename: str
    left_values: tuple[int, ...]
    right_values: tuple[int, ...]
    left_claim_id: str = ""
    right_claim_id: str = ""


def assess_claim(claim: str, passage: RankedPassage) -> ClaimAssessment:
    """Compare one claim with the passage it cites.

    ``supported`` clears the citation bar. ``unsupported`` does not.
    ``unverifiable`` has no content words left after the short-word filter.
    """
    ratio = support_with_context(claim, passage)
    if ratio is None:
        verdict = "unverifiable"
    elif ratio >= SUPPORT_THRESHOLD:
        verdict = "supported"
    else:
        verdict = "unsupported"
    return ClaimAssessment(
        claim=claim.strip(),
        verdict=verdict,
        support=ratio,
        filename=passage.source.filename,
        passage=passage,
        values=tuple(sorted(numeric_facts(claim))),
        claim_id="",
    )


def verify_answer(answer: AnswerResult) -> list[ClaimAssessment]:
    """Supported claims from a cited answer.

    One unsupported claim rejects the answer. A claim with nothing to check
    is skipped, matching the grounded-answer rule.
    """
    if not answer.sufficient_evidence or not answer.citations:
        return []
    passages = {passage.source.chunk_id: passage for passage in answer.passages}
    accepted = {citation.marker: citation for citation in answer.citations}
    supported: list[ClaimAssessment] = []
    for claim, marker in _cited_segments(answer.answer):
        citation = accepted.get(marker)
        if citation is None:
            continue
        passage = passages.get(citation.chunk_id)
        if passage is None:
            return []
        assessment = assess_claim(claim, passage)
        if assessment.verdict == "unverifiable" or not assessment.claim:
            continue
        if assessment.verdict != "supported":
            return []
        supported.append(assessment)
    return supported


def find_contradictions(question: str, claims: list[ClaimAssessment]) -> list[Contradiction]:
    """Pairs of supported claims from different files whose numbers do not overlap."""
    grouped: dict[str, set[int]] = {}
    claim_ids: dict[str, list[str]] = {}
    for claim in claims:
        if claim.verdict != "supported" or not claim.values:
            continue
        grouped.setdefault(claim.filename, set()).update(claim.values)
        if claim.claim_id:
            claim_ids.setdefault(claim.filename, []).append(claim.claim_id)
    stated = [(name, values) for name, values in grouped.items() if values]
    if len(stated) < 2 or set.intersection(*(values for _, values in stated)):
        return []
    conflicts: list[Contradiction] = []
    for left, right in _pairs(sorted(name for name, _ in stated)):
        left_values = grouped[left]
        right_values = grouped[right]
        if left_values & right_values:
            continue
        conflicts.append(
            Contradiction(
                summary=_summary(question, ((left, left_values), (right, right_values))),
                left_filename=left,
                right_filename=right,
                left_values=tuple(sorted(left_values)),
                right_values=tuple(sorted(right_values)),
                left_claim_id=claim_ids.get(left, [""])[0],
                right_claim_id=claim_ids.get(right, [""])[0],
            )
        )
    return conflicts


def passage_contradiction(question: str, passages: list[RankedPassage]) -> Contradiction:
    """Describe a retrieval conflict when cited claims did not already do so."""
    grouped: dict[str, set[int]] = {}
    for passage in passages:
        if not passage.answers_question:
            continue
        values = numeric_facts(passage.model_text())
        if values:
            grouped.setdefault(passage.source.filename, set()).update(values)
    stated = [(name, values) for name, values in sorted(grouped.items()) if values]
    if len(stated) >= 2 and not set.intersection(*(values for _, values in stated)):
        return Contradiction(
            summary=_summary(question, tuple(stated)),
            left_filename=stated[0][0],
            right_filename=stated[1][0],
            left_values=tuple(sorted(stated[0][1])),
            right_values=tuple(sorted(stated[1][1])),
        )
    return Contradiction(
        summary=f'"{question}" is contradicted by the visible documents.',
        left_filename=stated[0][0] if stated else "",
        right_filename=stated[1][0] if len(stated) > 1 else "",
        left_values=tuple(sorted(stated[0][1])) if stated else (),
        right_values=tuple(sorted(stated[1][1])) if len(stated) > 1 else (),
    )


def _summary(question: str, sides: tuple[tuple[str, set[int]], ...]) -> str:
    details = "; ".join(
        f"{name} states {', '.join(str(value) for value in sorted(values))}"
        for name, values in sides
    )
    return f'"{question}" is contradicted: {details}.'


def _cited_segments(answer: str) -> list[tuple[str, str]]:
    segments: list[tuple[str, str]] = []
    previous = 0
    for match in _MARKER.finditer(answer):
        segments.append((answer[previous : match.start()], match.group(0)))
        previous = match.end()
    return segments


def _pairs(names: list[str]) -> list[tuple[str, str]]:
    return list(combinations(names, 2))
