"""Decide whether a retrieved passage covers the question."""


def covers_question(
    coverage: float,
    matched_terms: tuple[str, ...],
    *,
    min_coverage: float,
) -> bool:
    """A passage answers the question when it contains enough of the question's terms."""
    return bool(matched_terms) and coverage >= min_coverage
