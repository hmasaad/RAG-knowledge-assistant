"""Tokenize questions and passages for lexical retrieval."""

import re

_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "do",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "our",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "where",
    "which",
    "with",
}


_ASK_META = {
    "also",
    "breakdown",
    "can",
    "could",
    "describe",
    "explain",
    "give",
    "i",
    "just",
    "like",
    "may",
    "me",
    "might",
    "need",
    "please",
    "properly",
    "really",
    "should",
    "show",
    "so",
    "tell",
    "understand",
    "want",
    "we",
    "would",
    "you",
}


def content_tokens(text: str) -> list[str]:
    """Return lowercase word tokens, dropping common question words when possible."""
    tokens = _TOKEN.findall(text.lower())
    content = [token for token in tokens if token not in _STOPWORDS]
    return content or tokens


def focus_tokens(text: str) -> list[str]:
    """Question tokens that name the topic, not the request to explain it."""
    tokens = content_tokens(text)
    focused = [token for token in tokens if token not in _ASK_META]
    return focused or tokens
