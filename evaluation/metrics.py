"""Recall@K, MRR, nDCG, faithfulness, answer relevance, and citation correctness."""

import math
import re
from collections.abc import Mapping, Sequence

from app.generation.grounding import SUPPORT_THRESHOLD, claim_support, is_refusal
from app.retrieval.text import content_tokens

_MARKER = re.compile(r"\[(\d+)\]")


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant items that appear in the first ``k`` results."""
    if k < 1 or not relevant:
        return 0.0
    return len(set(_first_ranks(retrieved)[:k]) & relevant) / len(relevant)


def reciprocal_rank(retrieved: Sequence[str], relevant: set[str]) -> float:
    """1 / rank of the first relevant item, or 0 when none is retrieved."""
    for index, item in enumerate(_first_ranks(retrieved), start=1):
        if item in relevant:
            return 1.0 / index
    return 0.0


def mean_reciprocal_rank(ranks: Sequence[float]) -> float:
    if not ranks:
        return 0.0
    return sum(ranks) / len(ranks)


def ndcg_at_k(retrieved: Sequence[str], gains: Mapping[str, float], k: int) -> float:
    """Normalized discounted cumulative gain for the first ``k`` results.

    Gains are relevance grades. A perfect ranking of the highest grades scores 1.
    """
    if k < 1 or not any(gain > 0 for gain in gains.values()):
        return 0.0
    ranked = _first_ranks(retrieved)
    dcg = _dcg(ranked[:k], gains)
    ideal = sorted((gain for gain in gains.values() if gain > 0), reverse=True)[:k]
    idcg = sum((2**gain - 1) / math.log2(index + 1) for index, gain in enumerate(ideal, start=1))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def faithfulness(answer: str, passages: Sequence[str]) -> float:
    """Mean support of each cited claim. A refusal makes no unsupported claim."""
    if is_refusal(answer):
        return 1.0
    ratios: list[float] = []
    for claim, number in _cited_claims(answer):
        if number < 1 or number > len(passages):
            ratios.append(0.0)
            continue
        ratio = claim_support(claim, passages[number - 1])
        if ratio is None:
            continue
        ratios.append(ratio)
    if not ratios:
        return 0.0
    return sum(ratios) / len(ratios)


def answer_relevance(
    question: str,
    answer: str,
    passages: Sequence[str],
    *,
    expect_answer: bool,
) -> float:
    """How much of the answerable question shows up in the answer.

    Question words that never appear in the passages are left out, because the
    evidence cannot support them. A correct refusal scores 1. Answering when
    the question should be refused scores 0.
    """
    if is_refusal(answer):
        return 0.0 if expect_answer else 1.0
    if not expect_answer:
        return 0.0
    evidence = set()
    for passage in passages:
        evidence.update(content_tokens(passage))
    wanted = [
        token
        for token in dict.fromkeys(content_tokens(question))
        if len(token) >= 4 and token in evidence
    ]
    if not wanted:
        return 0.0
    answered = set(content_tokens(answer))
    return sum(1 for token in wanted if token in answered) / len(wanted)


def citation_correctness(
    answer: str,
    passages: Sequence[tuple[str, str]],
    relevant: set[str],
    *,
    expect_answer: bool,
) -> float:
    """Share of citations that name a relevant passage and are supported by it.

    ``passages`` are ``(filename, text)`` in citation order. A refusal is
    correct when no answer was expected.
    """
    if is_refusal(answer):
        return 0.0 if expect_answer else 1.0
    claims = list(_cited_claims(answer))
    if not claims:
        return 0.0
    correct = 0
    for claim, number in claims:
        if number < 1 or number > len(passages):
            continue
        filename, text = passages[number - 1]
        ratio = claim_support(claim, text)
        if filename in relevant and ratio is not None and ratio >= SUPPORT_THRESHOLD:
            correct += 1
    return correct / len(claims)


def _first_ranks(retrieved: Sequence[str]) -> list[str]:
    """Keep the first time each item appears. A repeated document is not a new rank."""
    seen: set[str] = set()
    ordered: list[str] = []
    for item in retrieved:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _dcg(retrieved: Sequence[str], gains: Mapping[str, float]) -> float:
    total = 0.0
    for index, item in enumerate(retrieved, start=1):
        gain = gains.get(item, 0.0)
        total += (2**gain - 1) / math.log2(index + 1)
    return total


def _cited_claims(answer: str) -> list[tuple[str, int]]:
    claims: list[tuple[str, int]] = []
    previous = 0
    for match in _MARKER.finditer(answer):
        claims.append((answer[previous : match.start()], int(match.group(1))))
        previous = match.end()
    return claims
