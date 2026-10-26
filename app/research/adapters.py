"""Compose worker adapters. RAG, verification, and the graph are optional."""

from app.ingestion.metadata_store import MetadataStore
from app.research.contracts import (
    FILL_GAPS,
    SEARCH,
    SYNTHESIZE,
    ActionOutcome,
    ActionRequest,
    EvidenceRef,
    ResearchState,
)
from app.research.graph import EvidenceGraph, GraphBoundAdapter, ProvenanceEngine
from app.research.mocks import mock_action_adapters
from app.research.tools import ToolContext, ToolRegistry
from app.research.verify_actions import ResolveConflictsAction, VerifyAction


class SearchAction:
    """Retrieve passages through the knowledge tool. Nothing else."""

    action = SEARCH

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    def run(self, request: ActionRequest, state: ResearchState | None = None) -> ActionOutcome:
        return from_search(self._registry, self.action, request)


class FillGapsAction:
    """A second knowledge search for an unanswered clause."""

    action = FILL_GAPS

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    def run(self, request: ActionRequest, state: ResearchState | None = None) -> ActionOutcome:
        return from_search(self._registry, self.action, request)


class SynthesizeAction:
    action = SYNTHESIZE

    def run(self, request: ActionRequest, state: ResearchState | None = None) -> ActionOutcome:
        report = request.query.strip()
        if not report and state is not None:
            report = cited_answer(request.claim or request.query, state)
        return ActionOutcome(action=self.action, ok=True, report=report, cost=0)


def compose_action_adapters(
    *,
    registry: ToolRegistry | None = None,
    use_rag: bool = False,
    use_verification: bool = False,
    graph: EvidenceGraph | None = None,
    metadata: MetadataStore | None = None,
) -> list:
    """Start from mocks. Overlay one integration at a time.

    RAG replaces search and fill_gaps. Verification replaces reverify and
    resolve_contradictions. The graph wrapper records whatever those adapters
    return; it does not retrieve or judge.
    """
    adapters = {item.action: item for item in mock_action_adapters()}
    if use_rag:
        if registry is None:
            raise ValueError("RAG adapters need a tool registry.")
        adapters[SEARCH] = SearchAction(registry)
        adapters[FILL_GAPS] = FillGapsAction(registry)
    if use_verification:
        adapters["reverify"] = VerifyAction()
        adapters["resolve_contradictions"] = ResolveConflictsAction()
    adapters[SYNTHESIZE] = SynthesizeAction()
    built = list(adapters.values())
    if graph is None:
        return built
    engine = ProvenanceEngine(metadata)
    return [GraphBoundAdapter(item, graph, engine) for item in built]


def tool_action_adapters(registry: ToolRegistry, *, user_id: str = "local") -> list:
    """Back-compat: RAG search only. Verification and the graph stay unplugged."""
    return compose_action_adapters(registry=registry, use_rag=True)


def cited_answer(question: str, state: ResearchState) -> str:
    """Build a cited answer from supported claims and their passages."""
    supported = [item for item in state.verdicts if item.verdict == "supported"]
    by_id = {item.evidence_id: item for item in state.evidence}
    lines: list[str] = []
    seen: set[str] = set()
    marker = 0
    for verdict in supported:
        for evidence_id in verdict.evidence_ids:
            passage = by_id.get(evidence_id)
            if passage is None or passage.evidence_id in seen:
                continue
            seen.add(passage.evidence_id)
            marker += 1
            sentence = _first_sentence(passage.text)
            lines.append(f"{sentence} [{marker}]")
    if not lines:
        from app.security.prompts import REFUSAL_TEXT

        return REFUSAL_TEXT
    if state.contradictions:
        lines.append("The documents disagree.")
    return " ".join(lines)


def from_search(registry: ToolRegistry, action: str, request: ActionRequest) -> ActionOutcome:
    query = request.query or request.claim
    result = registry.run(
        "knowledge",
        query,
        ToolContext(user_id=request.user_id or "local", top_k=request.top_k or 4),
    )
    if not result.ok:
        return ActionOutcome(action=action, ok=False, error=result.error, cost=1)
    evidence: list[EvidenceRef] = []
    if result.search is not None:
        for passage in result.search.passages:
            evidence.append(
                EvidenceRef(
                    evidence_id=passage.source.chunk_id,
                    document_id=passage.document_id,
                    source=passage.source.filename,
                    text=passage.text,
                )
            )
    return ActionOutcome(
        action=action,
        ok=True,
        evidence=tuple(evidence),
        gaps=() if evidence else (query,),
        provenance=tuple(item.evidence_id for item in evidence),
        cost=1,
    )


def _first_sentence(text: str) -> str:
    stripped = " ".join(text.split())
    for index, char in enumerate(stripped):
        if char in ".!?" and (index + 1 == len(stripped) or stripped[index + 1] == " "):
            return stripped[: index + 1]
    return stripped[:240]
