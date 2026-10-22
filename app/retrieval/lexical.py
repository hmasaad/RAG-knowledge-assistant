"""BM25 scores over tokenized passages."""

import math
from collections import Counter


def bm25_scores(
    query_tokens: list[str],
    documents: list[list[str]],
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[float]:
    """Score each document against the query. Higher means a closer lexical match."""
    if not documents:
        return []
    terms = list(dict.fromkeys(query_tokens))
    if not terms:
        return [0.0] * len(documents)

    count = len(documents)
    average_length = sum(len(document) for document in documents) / count or 1.0
    document_frequency: dict[str, int] = {}
    for document in documents:
        for term in set(document):
            document_frequency[term] = document_frequency.get(term, 0) + 1

    scores: list[float] = []
    for document in documents:
        frequencies = Counter(document)
        length = len(document) or 1
        score = 0.0
        for term in terms:
            frequency = frequencies.get(term, 0)
            if frequency == 0:
                continue
            containing = document_frequency.get(term, 0)
            idf = math.log(1 + (count - containing + 0.5) / (containing + 0.5))
            denominator = frequency + k1 * (1 - b + b * length / average_length)
            score += idf * (frequency * (k1 + 1)) / denominator
        scores.append(score)
    return scores
