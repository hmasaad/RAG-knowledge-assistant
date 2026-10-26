"""Approved tools the research worker may call."""

from dataclasses import dataclass
from typing import Protocol

from app.ingestion.graph import path_entities
from app.ingestion.metadata_store import MetadataStore
from app.research.orchestrator import SearchBudget, run_search
from app.research.planner import APPROVED_TOOLS
from app.research.rag import RagClient, RagSearchResult, SearchFilters

GRAPH_LIMIT = 4


@dataclass(frozen=True)
class ToolContext:
    user_id: str
    top_k: int = 4
    filters: SearchFilters | None = None
    budget: SearchBudget | None = None
    document_ids: set[str] | None = None


@dataclass(frozen=True)
class ToolResult:
    tool: str
    query: str
    ok: bool
    error: str = ""
    names: tuple[str, ...] = ()
    search: RagSearchResult | None = None


class ResearchTool(Protocol):
    name: str

    def run(self, query: str, context: ToolContext) -> ToolResult: ...


class KnowledgeTool:
    """Search the indexed collection through the RAG contract."""

    name = "knowledge"

    def __init__(self, rag: RagClient) -> None:
        self._rag = rag

    def run(self, query: str, context: ToolContext) -> ToolResult:
        filters = context.filters or SearchFilters(user_id=context.user_id)
        if context.budget is None:
            try:
                found = self._rag.search(query, filters, context.top_k)
            except Exception as exc:
                return ToolResult(self.name, query, False, error=str(exc))
            return ToolResult(self.name, query, True, search=found)
        found, error = run_search(self._rag, context.budget, query, filters, context.top_k)
        if error or found is None:
            return ToolResult(self.name, query, False, error=error or "Search failed.")
        return ToolResult(self.name, query, True, search=found)


class CatalogTool:
    """List documents this caller can see."""

    name = "catalog"

    def __init__(self, metadata: MetadataStore) -> None:
        self._metadata = metadata

    def run(self, query: str, context: ToolContext) -> ToolResult:
        names = tuple(
            record.filename
            for record in self._metadata.list_visible(context.user_id)
            if record.indexed_version >= 1
        )
        return ToolResult(self.name, query, True, names=names)


class GraphTool:
    """Return related file paths from the knowledge graph."""

    name = "graph"

    def __init__(self, metadata: MetadataStore) -> None:
        self._metadata = metadata

    def run(self, query: str, context: ToolContext) -> ToolResult:
        names = path_entities(query)
        visible = context.document_ids
        if visible is None:
            visible = set(self._metadata.readable_versions(context.user_id))
        neighbors = tuple(self._metadata.neighbors(names, visible)[:GRAPH_LIMIT])
        return ToolResult(self.name, query, True, names=neighbors)


class ToolRegistry:
    """Look up an approved tool by name."""

    def __init__(self, tools: list[ResearchTool] | None = None) -> None:
        self._tools = {tool.name: tool for tool in tools or []}

    def register(self, tool: ResearchTool) -> None:
        self._tools[tool.name] = tool

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._tools))

    def get(self, name: str) -> ResearchTool | None:
        return self._tools.get(name)

    def run(self, name: str, query: str, context: ToolContext) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(name, query, False, error=f"Unknown tool: {name}.")
        return tool.run(query, context)


def approved_tools(*, rag: RagClient, metadata: MetadataStore) -> ToolRegistry:
    """The catalog, graph, and knowledge tools the research worker may use."""
    registry = ToolRegistry()
    for tool in (KnowledgeTool(rag), CatalogTool(metadata), GraphTool(metadata)):
        if tool.name in APPROVED_TOOLS:
            registry.register(tool)
    return registry
