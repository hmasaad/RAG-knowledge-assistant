"""Compose worker adapters. RAG, verification, and the graph are optional."""

from app.ingestion.metadata_store import MetadataStore
from app.observability.logging import log_event
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
        context = "\n\n".join(item.text for item in state.evidence) if state is not None else ""
        looks_like_answer = "[" in report
        if state is not None and (not report or not looks_like_answer):
            report = cited_answer(request.claim or request.query or report, state)
        log_event(
            "rag.research.stages",
            "synthesize",
            claim=request.claim or request.query,
            input_context=context,
            prompt=request.query,
            raw_response=report,
            evidence_ids=[item.evidence_id for item in (state.evidence if state else ())],
        )
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
    """Build a short cited report from the passages that best match the question."""
    from app.generation.agent import wants_section_summary
    from app.generation.answerers import _section_points, numbered_list
    from app.research.context_evidence import section_matches_question
    from app.research.synthesis_guard import preserve_supported
    from app.retrieval.text import content_tokens, focus_tokens
    from app.security.prompts import REFUSAL_TEXT

    if state.ledger:
        guarded = preserve_supported(question, state)
        if guarded != REFUSAL_TEXT:
            return guarded

    supported = [item for item in state.verdicts if item.verdict == "supported"]
    by_id = {item.evidence_id: item for item in state.evidence}
    focus = tuple(focus_tokens(question))
    scored: list[tuple[int, str]] = []
    seen: set[str] = set()
    for verdict in supported:
        for evidence_id in verdict.evidence_ids:
            passage = by_id.get(evidence_id)
            if passage is None or passage.evidence_id in seen:
                continue
            seen.add(passage.evidence_id)
            sentence = _quoted_evidence(question, passage.text)
            if not sentence:
                continue
            overlap = len(set(focus) & set(content_tokens(sentence)))
            if overlap < 1:
                continue
            scored.append((overlap, sentence))
    scored.sort(key=lambda item: (-item[0], len(item[1])))
    lines = []
    used: set[str] = set()
    limit = 16 if wants_section_summary(question) else 2
    ordered: list[tuple[str, int]] = []
    if wants_section_summary(question):
        marker = 0
        for verdict in supported:
            for evidence_id in verdict.evidence_ids:
                passage = by_id.get(evidence_id)
                if passage is None:
                    continue
                marker += 1
                inherited = " ".join(
                    part
                    for part in (passage.heading, passage.section_title)
                    if part
                )
                member = section_matches_question(passage.section_title or passage.heading, question)
                for point in _section_points(
                    passage.text,
                    focus,
                    require_topic=not member,
                    inherited=inherited,
                ):
                    key = point.casefold()
                    if key in used:
                        continue
                    used.add(key)
                    ordered.append((point, marker))
                    if len(ordered) >= limit:
                        break
                if len(ordered) >= limit:
                    break
            if len(ordered) >= limit:
                break
    if not ordered:
        for overlap, sentence in scored:
            key = sentence.casefold()
            if key in used:
                continue
            used.add(key)
            ordered.append((sentence, len(ordered) + 1))
            if len(ordered) >= limit:
                break
    if not ordered:
        marker = 0
        for verdict in supported:
            for evidence_id in verdict.evidence_ids:
                passage = by_id.get(evidence_id)
                if passage is None:
                    continue
                marker += 1
                inherited = " ".join(
                    part for part in (passage.heading, passage.section_title) if part
                )
                for point in _section_points(
                    passage.text,
                    focus,
                    require_topic=False,
                    inherited=inherited,
                ):
                    key = point.casefold()
                    if key in used:
                        continue
                    used.add(key)
                    ordered.append((point, marker))
                    if len(ordered) >= limit:
                        break
                if len(ordered) >= limit:
                    break
            if len(ordered) >= limit:
                break
    if not ordered:
        return REFUSAL_TEXT
    text = numbered_list(ordered)
    if state.contradictions:
        text = f"{text}\nThe documents disagree."
    return text


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
                    heading=passage.source.heading or "",
                    section_title=passage.source.section_title or passage.source.heading or "",
                    section_id=passage.source.section_id,
                    section_number=passage.source.section_number,
                    start_char=passage.source.start_char,
                    end_char=passage.source.end_char,
                    page=passage.source.page,
                    chunk_index=passage.source.chunk_index,
                )
            )
    passages = [] if result.search is None else list(result.search.passages)
    log_event(
        "rag.research.stages",
        "search",
        query=query,
        chunk_ids=[item.evidence_id for item in evidence],
        scores=[passage.score for passage in passages],
        token_counts=[len(item.text.split()) for item in evidence],
        texts=[item.text for item in evidence],
    )
    return ActionOutcome(
        action=action,
        ok=True,
        evidence=tuple(evidence),
        gaps=() if evidence else (query,),
        provenance=tuple(item.evidence_id for item in evidence),
        cost=1,
    )


def _quoted_evidence(question: str, text: str) -> str:
    from app.generation.answerers import topic_excerpt
    from app.retrieval.text import focus_tokens
    from app.security.content import neutralize_untrusted_text

    safe = neutralize_untrusted_text(text).text or text
    excerpt = topic_excerpt(safe, tuple(focus_tokens(question)))
    return excerpt


def _first_sentence(text: str) -> str:
    stripped = " ".join(text.split())
    for index, char in enumerate(stripped):
        if char in ".!?" and (index + 1 == len(stripped) or stripped[index + 1] == " "):
            return stripped[: index + 1]
    return stripped[:240]
