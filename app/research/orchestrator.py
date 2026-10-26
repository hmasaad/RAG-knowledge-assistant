"""Charge each knowledge search against a fixed budget."""

from app.research.rag import RagClient, RagContractError, RagSearchResult, SearchFilters


class SearchBudget:
    """How many knowledge searches one research request may run."""

    def __init__(self, limit: int) -> None:
        if limit < 1:
            raise ValueError("Search budget must be at least 1.")
        self.limit = limit
        self.used = 0

    def allow(self) -> bool:
        if self.used >= self.limit:
            return False
        self.used += 1
        return True


def run_search(
    rag: RagClient,
    budget: SearchBudget,
    query: str,
    filters: SearchFilters,
    top_k: int,
) -> tuple[RagSearchResult | None, str]:
    """Run one search, or return an error when the budget or the contract fails.

    The error is empty when a result was returned. A budget miss and a contract
    failure both return no result. The caller decides whether a failed retry
    should keep an earlier result.
    """
    if not budget.allow():
        return None, "Search budget exhausted."
    try:
        return rag.search(query, filters, top_k), ""
    except RagContractError as exc:
        return None, exc.message
