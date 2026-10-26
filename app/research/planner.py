"""Split a research question and choose an approved source for each part."""

from dataclasses import dataclass

from app.ingestion.graph import path_entities
from app.retrieval.queries import decompose_query, rewrite_query

MAX_SUBQUESTIONS = 4
APPROVED_TOOLS = ("knowledge", "catalog", "graph")
KNOWLEDGE_COMPLETION = "Cite a claim supported by a retrieved passage."
CATALOG_COMPLETION = "List the documents visible to this caller."
GRAPH_COMPLETION = "Search each related path from the knowledge graph."
_CATALOG_PHRASES = (
    "which document",
    "which documents",
    "what documents",
    "which files",
    "what files",
    "which sources",
    "what sources",
    "available documents",
)


def subquestions(question: str) -> tuple[str, ...]:
    """Clauses that can be investigated on their own.

    A single clause stays as the caller wrote it, so a weak polite question
    can still be retried. Two or more clauses come from the rewritten question.
    """
    text = question.strip()
    clauses = decompose_query(rewrite_query(text))
    if len(clauses) < 2:
        return (text,)
    return clauses[:MAX_SUBQUESTIONS]


def pending_subquestions(question: str) -> tuple[str, ...]:
    """Clauses past the investigation limit. They are reported as gaps."""
    clauses = decompose_query(rewrite_query(question.strip()))
    if len(clauses) <= MAX_SUBQUESTIONS:
        return ()
    return clauses[MAX_SUBQUESTIONS:]


def select_tools(question: str) -> tuple[str, ...]:
    """Approved tools for this question. The knowledge collection is always included."""
    tools = ["knowledge"]
    folded = question.casefold()
    if any(phrase in folded for phrase in _CATALOG_PHRASES):
        tools.append("catalog")
    if path_entities(question):
        tools.append("graph")
    return tuple(tools)


@dataclass(frozen=True)
class PlannedTask:
    """One subquestion that can be executed without the others."""

    task_id: str
    question: str
    tool: str
    completion: str


@dataclass(frozen=True)
class ResearchPlan:
    question: str
    tools: tuple[str, ...]
    tasks: tuple[PlannedTask, ...]
    deferred: tuple[str, ...]
    budget: int
    catalog_completion: str = ""


def build_plan(question: str, *, budget: int) -> ResearchPlan:
    """Turn one request into independent tasks with a completion criterion each."""
    tools = select_tools(question)
    tasks = tuple(
        PlannedTask(
            task_id=f"q{index}",
            question=clause,
            tool="knowledge",
            completion=KNOWLEDGE_COMPLETION,
        )
        for index, clause in enumerate(subquestions(question), start=1)
    )
    return ResearchPlan(
        question=question.strip(),
        tools=tools,
        tasks=tasks,
        deferred=pending_subquestions(question),
        budget=budget,
        catalog_completion=CATALOG_COMPLETION if "catalog" in tools else "",
    )
