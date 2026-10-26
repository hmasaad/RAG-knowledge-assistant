"""The evidence graph is recorded only when the worker wraps adapters with it."""

from app.research.adapters import compose_action_adapters
from app.research.contracts import REVERIFY, SEARCH, ActionRequest
from app.research.graph import CLAIM, PASSAGE, SUPPORTS, EvidenceGraph
from app.research.worker import ResearchWorker


def test_worker_graph_stays_empty_until_the_recorder_is_wired() -> None:
    worker = ResearchWorker()
    worker.dispatch(ActionRequest(SEARCH, query="retry limit"))
    worker.dispatch(ActionRequest(REVERIFY, claim="the retry limit is three"))

    assert worker.graph is None


def test_graph_adapter_records_passages_and_supported_claims() -> None:
    graph = EvidenceGraph()
    worker = ResearchWorker(
        adapters=compose_action_adapters(graph=graph),
        graph=graph,
    )
    worker.dispatch(ActionRequest(SEARCH, query="retry limit"))
    worker.dispatch(ActionRequest(REVERIFY, claim="the retry limit is three attempts"))

    kinds = {node.kind for node in graph.nodes.values()}
    relations = {edge.relation for edge in graph.edges}

    assert "passage:mock-1" in graph.nodes
    assert any(node.kind == CLAIM for node in graph.nodes.values())
    assert PASSAGE in kinds
    assert SUPPORTS in relations


def test_cycle_resets_the_graph_between_runs() -> None:
    graph = EvidenceGraph()
    worker = ResearchWorker(
        adapters=compose_action_adapters(graph=graph),
        graph=graph,
    )
    first = worker.cycle("what is the retry limit")
    first_count = len(graph.nodes)
    second = worker.cycle("what is the retry limit")

    assert first.graph is graph
    assert second.graph is graph
    assert first_count > 0
    assert len(graph.nodes) == first_count
