"""Entities and relations extracted from indexed text."""

import re
from dataclasses import dataclass

_PATH = re.compile(
    r"\b[A-Za-z][A-Za-z0-9_-]*(?:/[A-Za-z0-9_.-]+)+\b"
    r"|\b[A-Za-z][A-Za-z0-9_-]*\.[A-Za-z0-9]+\b"
)


@dataclass(frozen=True)
class GraphEdge:
    source: str
    relation: str
    target: str
    chunk_id: str


def path_entities(text: str) -> list[str]:
    """File and module paths mentioned in the text, in order."""
    found: list[str] = []
    seen: set[str] = set()
    for match in _PATH.finditer(text):
        value = match.group(0)
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        found.append(value)
    return found


def extract_edges(text: str, chunk_id: str) -> list[GraphEdge]:
    """Link paths that share a sentence.

    ``calls``, ``implements``, and ``configured_in`` come from the sentence
    wording. Other co-mentioned paths are linked with ``mentions``.
    """
    edges: list[GraphEdge] = []
    seen: set[tuple[str, str, str]] = set()
    for sentence in _sentences(text):
        paths = path_entities(sentence)
        if len(paths) < 2:
            continue
        relation = _relation(sentence)
        key = (paths[0].casefold(), relation, paths[1].casefold())
        if key in seen:
            continue
        seen.add(key)
        edges.append(GraphEdge(source=paths[0], relation=relation, target=paths[1], chunk_id=chunk_id))
    return edges


def is_path(value: str) -> bool:
    return _PATH.fullmatch(value) is not None


def _relation(sentence: str) -> str:
    lowered = sentence.casefold()
    if "call" in lowered:
        return "calls"
    if "implement" in lowered:
        return "implements"
    if "configur" in lowered or "timeout" in lowered:
        return "configured_in"
    return "mentions"


def _sentences(text: str) -> list[str]:
    parts: list[str] = []
    start = 0
    length = len(text)
    for index, char in enumerate(text):
        boundary = char == "\n" or (
            char in ".!?" and (index + 1 == length or text[index + 1].isspace())
        )
        if boundary:
            parts.append(text[start : index + 1])
            start = index + 1
    if start < length:
        parts.append(text[start:])
    return [part.strip() for part in parts if part.strip()]
