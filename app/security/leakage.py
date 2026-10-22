"""Stop an answer from repeating a secret that belongs to another caller."""

import re

_SENSITIVE = re.compile(
    r"(?i)\b(?:canary|api[_-]?key|password|secret)\s*[:=]\s*([A-Za-z0-9][A-Za-z0-9._-]{5,})"
)


def sensitive_tokens(text: str) -> set[str]:
    """Return secret values marked as canary, api key, password, or secret."""
    return {match.group(1) for match in _SENSITIVE.finditer(text)}


def answer_leaks(answer: str, forbidden: set[str]) -> bool:
    """True when the answer contains a secret the caller must not see."""
    if not answer or not forbidden:
        return False
    lowered = answer.casefold()
    return any(token.casefold() in lowered for token in forbidden)
