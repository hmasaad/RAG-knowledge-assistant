"""Score final answers against a golden set with the evidence held fixed.

Completeness asks whether every required fact is in the answer. Citation
integrity asks whether that fact is cited to the passage that contains it.
Retrieval is not scored. A better search cannot raise these numbers, and it
cannot hide an answer that drops a fact or cites the wrong source.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from app.generation.answerers import LocalAnswerer
from app.generation.grounding import is_refusal
from app.retrieval.search import RankedPassage, RetrievalResult, SourceMetadata
from app.retrieval.text import content_tokens

_MARKER = re.compile(r"\[(\d+)\]")


@dataclass(frozen=True)
class GoldenEvidence:
    filename: str
    text: str


@dataclass(frozen=True)
class GoldenRequirement:
    detail: str
    source: str


@dataclass(frozen=True)
class GoldenCase:
    id: str
    question: str
    evidence: tuple[GoldenEvidence, ...]
    requirements: tuple[GoldenRequirement, ...]
    gold_answer: str
    expect_refusal: bool = False


@dataclass(frozen=True)
class GoldenScore:
    id: str
    completeness: float
    citation_integrity: float


def load_golden_cases(path: Path) -> list[GoldenCase]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = []
    for item in payload:
        cases.append(
            GoldenCase(
                id=item["id"],
                question=item["question"],
                evidence=tuple(
                    GoldenEvidence(filename=row["filename"], text=row["text"])
                    for row in item["evidence"]
                ),
                requirements=tuple(
                    GoldenRequirement(detail=row["detail"], source=row["source"])
                    for row in item.get("requirements", ())
                ),
                gold_answer=item["gold_answer"],
                expect_refusal=bool(item.get("expect_refusal", False)),
            )
        )
    return cases


def score_golden(case: GoldenCase, answer: str) -> GoldenScore:
    """Score one answer. The evidence list is the only source of truth."""
    if case.expect_refusal:
        credit = 1.0 if is_refusal(answer) else 0.0
        return GoldenScore(case.id, credit, credit)
    total = len(case.requirements)
    if total == 0:
        return GoldenScore(case.id, 0.0, 0.0)
    covered = sum(1 for item in case.requirements if _stated(answer, item.detail))
    cited = sum(1 for item in case.requirements if _cited(answer, case.evidence, item))
    return GoldenScore(
        case.id,
        round(covered / total, 2),
        round(cited / total, 2),
    )


def run_answer_golden(root: Path, cases: list[GoldenCase] | None = None) -> dict[str, float]:
    """Answer each case from its fixed evidence and average the two scores."""
    if cases is None:
        cases = load_golden_cases(root / "evaluation" / "datasets" / "answer_golden.json")
    scores = [score_golden(case, _answer(case)) for case in cases]
    return {
        "completeness": _mean(item.completeness for item in scores),
        "citation_integrity": _mean(item.citation_integrity for item in scores),
    }


def _answer(case: GoldenCase) -> str:
    passages = [_passage(index, item, answers=not case.expect_refusal) for index, item in enumerate(case.evidence)]
    retrieved = RetrievalResult(
        query=case.question,
        sufficient_evidence=not case.expect_refusal,
        inspection="golden evidence",
        passages=passages,
    )
    return LocalAnswerer().generate(case.question, retrieved).answer


def _passage(index: int, item: GoldenEvidence, *, answers: bool) -> RankedPassage:
    source = SourceMetadata(
        chunk_id=f"golden-{index}",
        document_id=f"golden-doc-{index}",
        filename=item.filename,
        content_type="text/markdown",
        chunk_index=index,
        start_char=0,
        end_char=len(item.text),
        page=None,
        heading=None,
    )
    return RankedPassage(
        rank=index + 1,
        text=item.text,
        score=1.0,
        vector_score=1.0,
        lexical_score=1.0,
        answers_question=answers,
        matched_terms=(),
        source=source,
        safe_text=item.text,
    )


def _stated(answer: str, detail: str) -> bool:
    if is_refusal(answer):
        return False
    return _covers(answer, detail)


def _cited(answer: str, evidence: tuple[GoldenEvidence, ...], requirement: GoldenRequirement) -> bool:
    if is_refusal(answer) or not _stated(answer, requirement.detail):
        return False
    for claim, number in _segments(answer):
        if number < 1 or number > len(evidence):
            continue
        passage = evidence[number - 1]
        if passage.filename != requirement.source:
            continue
        if _covers(claim, requirement.detail) and _covers(passage.text, requirement.detail):
            return True
    return False


def _covers(text: str, detail: str) -> bool:
    tokens = _tokens(detail)
    if not tokens:
        return False
    present = set(content_tokens(text))
    return all(token in present for token in tokens)


def _tokens(detail: str) -> list[str]:
    tokens = [token for token in content_tokens(detail) if len(token) >= 4]
    return tokens or content_tokens(detail)


def _segments(answer: str) -> list[tuple[str, int]]:
    found = []
    previous = 0
    for match in _MARKER.finditer(answer):
        found.append((answer[previous : match.start()], int(match.group(1))))
        previous = match.end()
    return found


def _mean(values) -> float:
    rows = list(values)
    if not rows:
        return 0.0
    return round(sum(rows) / len(rows), 2)
