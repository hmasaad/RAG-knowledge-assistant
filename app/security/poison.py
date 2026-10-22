"""Drop chunks that repeat one word often enough to dominate keyword search."""

from collections import Counter

from app.retrieval.text import content_tokens

_MIN_TOKENS = 12
_MIN_REPEATS = 8
_DOMINANCE = 0.45


def is_keyword_stuffed(text: str) -> bool:
    """True when one token is repeated so often that the chunk is a ranking trap.

    Ordinary prose stays searchable. A chunk built to flood BM25 with the
    question's words does not.
    """
    tokens = content_tokens(text)
    if len(tokens) < _MIN_TOKENS:
        return False
    count = Counter(tokens).most_common(1)[0][1]
    return count >= _MIN_REPEATS and count / len(tokens) >= _DOMINANCE
