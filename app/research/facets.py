"""Scope, dates, versions, quantities, and qualifiers for one claim."""

import re

from app.retrieval.conflicts import numeric_facts
from app.retrieval.text import content_tokens

_DATE = re.compile(r"\b(?:19|20)\d{2}\b|\b\d{4}-\d{2}-\d{2}\b")
_VERSION = re.compile(r"\bversion\s+(\d+)\b|\bv(\d+)\b", re.IGNORECASE)
_NEGATION = re.compile(r"\b(?:does not|do not|is not|isn['’]t|cannot|can not|never)\b", re.IGNORECASE)
_QUALIFIERS = (
    "default",
    "maximum",
    "minimum",
    "least",
    "most",
    "only",
    "approximately",
    "required",
    "optional",
)
_UNITS = {
    "second": "seconds",
    "seconds": "seconds",
    "minute": "minutes",
    "minutes": "minutes",
    "millisecond": "milliseconds",
    "milliseconds": "milliseconds",
    "hour": "hours",
    "hours": "hours",
    "day": "days",
    "days": "days",
    "attempt": "attempts",
    "attempts": "attempts",
    "retry": "retries",
    "retries": "retries",
    "percent": "percent",
    "%": "percent",
}
_POPULATIONS = (
    "production",
    "staging",
    "callers",
    "users",
    "clients",
    "developers",
    "operators",
)
_CONDITION = re.compile(r"\b(?:when|if|unless|after|before)\s+(.{1,80})")
_NUMBER_WORDS = {
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
}


class Facets:
    def __init__(
        self,
        *,
        scope: frozenset[str],
        dates: frozenset[str],
        versions: frozenset[str],
        quantities: frozenset[int],
        qualifiers: frozenset[str],
        units: frozenset[str],
        populations: frozenset[str],
        conditions: frozenset[str],
        negated: bool,
    ) -> None:
        self.scope = scope
        self.dates = dates
        self.versions = versions
        self.quantities = quantities
        self.qualifiers = qualifiers
        self.units = units
        self.populations = populations
        self.conditions = conditions
        self.negated = negated


def extract_facets(text: str) -> Facets:
    """Pull the parts of a sentence that decide whether two statements are the same fact."""
    dates = frozenset(_DATE.findall(text))
    versions = frozenset(group for match in _VERSION.finditer(text) for group in match.groups() if group)
    qualifiers = frozenset(name for name in _QUALIFIERS if re.search(rf"\b{name}\b", text, re.IGNORECASE))
    quantities = set(numeric_facts(text))
    for version in versions:
        quantities.discard(int(version))
    for date in dates:
        if date.isdigit():
            quantities.discard(int(date))
    units = _units(text)
    populations = frozenset(name for name in _POPULATIONS if re.search(rf"\b{name}\b", text, re.IGNORECASE))
    conditions = _conditions(text)
    skip = set(qualifiers) | set(units) | set(populations) | set(conditions) | {"version"} | _NUMBER_WORDS
    scope = frozenset(
        token
        for token in content_tokens(text)
        if token not in skip and not token.isdigit() and token not in dates
    )
    return Facets(
        scope=scope,
        dates=dates,
        versions=versions,
        quantities=frozenset(quantities),
        qualifiers=qualifiers,
        units=units,
        populations=populations,
        conditions=conditions,
        negated=bool(_NEGATION.search(text)),
    )


def same_fact(claim: Facets, passage: Facets) -> bool:
    """True when scope, dates, versions, and qualifiers describe one fact.

    A missing date, version, or qualifier on the passage does not by itself
    make the fact a different one. A different value does.
    """
    if claim.dates and passage.dates and claim.dates != passage.dates:
        return False
    if claim.versions and passage.versions and claim.versions != passage.versions:
        return False
    if claim.qualifiers and passage.qualifiers and claim.qualifiers != passage.qualifiers:
        return False
    if claim.units and passage.units and claim.units != passage.units:
        return False
    if claim.populations and passage.populations and claim.populations != passage.populations:
        return False
    if claim.conditions and passage.conditions:
        overlap = sum(1 for token in claim.conditions if token in passage.conditions) / len(
            claim.conditions
        )
        if overlap < 0.5:
            return False
    if not claim.scope:
        return False
    overlap = sum(1 for token in claim.scope if token in passage.scope) / len(claim.scope)
    return overlap >= 0.6


def missing_qualifier(claim: Facets, passage: Facets) -> bool:
    """The claim states a date, version, or qualifier the passage does not."""
    if claim.dates and not passage.dates:
        return True
    if claim.versions and not passage.versions:
        return True
    if claim.qualifiers and not passage.qualifiers:
        return True
    if claim.units and not passage.units:
        return True
    if claim.populations and not passage.populations:
        return True
    if claim.conditions and not passage.conditions:
        return True
    return False


def _units(text: str) -> frozenset[str]:
    found: set[str] = set()
    for token in content_tokens(text):
        unit = _UNITS.get(token)
        if unit:
            found.add(unit)
    if "%" in text:
        found.add("percent")
    return frozenset(found)


def _conditions(text: str) -> frozenset[str]:
    tokens: set[str] = set()
    for match in _CONDITION.finditer(text):
        tokens.update(content_tokens(match.group(1)))
    return frozenset(tokens)


def evidence_role(claim_text: str, passage_text: str) -> str:
    """Label a passage as support, oppose, partial, or unrelated."""
    claim = extract_facets(claim_text)
    passage = extract_facets(passage_text)
    if not same_fact(claim, passage):
        return "unrelated"
    if passage.negated:
        return "oppose"
    if claim.quantities and passage.quantities and claim.quantities.isdisjoint(passage.quantities):
        return "oppose"
    if missing_qualifier(claim, passage) or (
        claim.quantities and not claim.quantities <= passage.quantities
    ):
        return "partial"
    return "support"


def opposition_query(claim: str) -> str:
    """Topic words used to retrieve a passage that might disagree with the claim."""
    scope = extract_facets(claim).scope
    tokens = [token for token in content_tokens(claim) if token in scope]
    return " ".join(tokens) or claim.strip()
