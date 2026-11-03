"""The source trace connects claims to documents, chunks, entities, and versions."""

import hashlib
from pathlib import Path

from fastapi.testclient import TestClient

from app.research.graph import EXTRACTED_FROM, HAS_VERSION, MENTIONS, SUPPORTS
from app.research.trace import _checksum, build_asset_graph, trace_scene

ROOT = Path(__file__).resolve().parents[1]
ROOT_KINDS = {"claim", "passage", "document", "version", "source", "entity", "agent"}


def test_nvidia_graph_connects_documents_chunks_claims_and_entities() -> None:
    graph = build_asset_graph("NVDA")
    kinds = {node.kind for node in graph.nodes.values()}
    nvidia = graph.nodes["entity:nvidia"]
    mentions = [edge for edge in graph.edges if edge.relation == MENTIONS and edge.target == nvidia.id]
    extracted = [
        edge
        for edge in graph.edges
        if edge.relation == EXTRACTED_FROM and edge.target == "document:doc-nvda-news"
    ]
    versions = [
        edge
        for edge in graph.edges
        if edge.relation == HAS_VERSION and edge.source == "document:doc-nvda-news"
    ]
    history = [edge for edge in graph.edges if edge.relation == "supersedes"]

    assert ROOT_KINDS <= kinds
    assert len(mentions) >= 3
    assert extracted
    assert {edge.target for edge in versions} == {
        "version:doc-nvda-news:v1",
        "version:doc-nvda-news:v2",
    }
    assert history[0].source == "version:doc-nvda-news:v2"
    assert history[0].target == "version:doc-nvda-news:v1"


def test_aggregator_claims_keep_upstream_counts() -> None:
    scene = trace_scene("NVDA")
    by_id = {claim["id"]: claim for claim in scene["claims"]}

    assert by_id["claim:a1"]["upstream_count"] == 4
    assert by_id["claim:a2"]["upstream_count"] == 7
    assert set(by_id["claim:a3"]["upstream"]) == {
        "claim:s5",
        "claim:s6",
        "claim:s4",
        "claim:s2",
        "claim:n2",
    }
    assert "Bridgewater" in by_id["claim:a3"]["entities"]
    assert "Ray Dalio" in by_id["claim:a3"]["entities"]
    assert set(by_id["claim:s6"]["contradicts"]) == {"claim:s4"}


def test_news_claim_records_version_history_and_character_span() -> None:
    scene = trace_scene("NVDA")
    claim = next(item for item in scene["claims"] if item["id"] == "claim:n1")
    origin = claim["provenance"]

    assert origin["filename"] == "nvda-news.md"
    assert origin["version"] == 2
    assert origin["checksum"] == _checksum("nvda-news.md", 2)
    assert origin["checksum"] == hashlib.sha256(b"nvda-news.md\nv2").hexdigest()
    assert origin["start_char"] == 0
    assert origin["end_char"] > origin["start_char"]
    assert [item["version"] for item in origin["history"]] == [1, 2]
    assert origin["history"][0]["checksum"] != origin["history"][1]["checksum"]
    assert origin["history"][0]["recorded_at"] == "2026-10-01T00:00:00+00:00"
    assert origin["history"][1]["recorded_at"] == "2026-11-03T00:00:00+00:00"
    assert {"claim", "passage", "source", "document", "version"} <= {step["kind"] for step in origin["lineage"]}
    assert claim["citation"]["filename"] == "nvda-news.md"
    assert claim["citation"]["start_char"] == 0
    assert claim["citation"]["end_char"] == origin["end_char"]
    decisions = {event["decision"] for event in claim["audit"]}
    assert {"supported", "superseded"} <= decisions
    assert {agent["label"] for agent in scene["agents"]} == {"News Agent", "Social Agent", "Aggregator"}


def test_trace_api_returns_the_nvidia_scene(client: TestClient) -> None:
    page = client.get("/trace")
    home = client.get("/")
    body = client.get("/api/trace", params={"asset": "NVDA"})
    missing = client.get("/api/trace", params={"asset": "NOPE"})
    scene = body.json()

    assert page.status_code == 200
    assert "Source Trace Visualiser" in page.text
    assert "Clear Selection" in page.text
    assert 'href="/trace"' in home.text
    assert body.status_code == 200
    assert scene["asset"] == "NVDA"
    assert scene["selected_claim"] == "claim:a3"
    assert "[[3]]" in scene["answer"]
    assert scene["markers"][2] == {"n": 3, "claim_id": "claim:a3"}
    assert missing.status_code == 404


def test_ask_cites_bridgewater_and_refuses_unknown_facts(client: TestClient) -> None:
    cited = client.post(
        "/api/trace/ask",
        json={"question": "What is Bridgewater doing with NVIDIA?", "asset": "NVDA"},
    )
    refused = client.post(
        "/api/trace/ask",
        json={"question": "What is the office wifi password?", "asset": "NVDA"},
    )
    tesla = client.post("/api/trace/ask", json={"question": "What are people saying about Tesla?"})
    cited_body = cited.json()
    support = [
        edge
        for edge in cited_body["scene"]["edges"]
        if edge["relation"] == SUPPORTS
        and edge["target"] == "claim:a3"
        and edge["source"].startswith("claim:")
    ]

    assert cited.status_code == 200
    panel = [claim for claim in cited_body["scene"]["claims"] if claim["band"] == "panel"]

    assert cited_body["markers"][0]["claim_id"] == "claim:a3"
    assert "[[1]]" in cited_body["answer"]
    assert len(support) == 5
    assert cited_body["scene"]["selected_claim"] == "claim:a3"
    assert cited_body["scene"]["selected_agent"].endswith(":aggregator")
    assert len(panel) >= 4
    assert {claim["id"] for claim in panel} >= {"claim:a3", "claim:agg-s6"}
    assert "claim:s6" in {claim["id"] for claim in cited_body["scene"]["claims"]}
    bridgewater = next(claim for claim in cited_body["scene"]["claims"] if claim["id"] == "claim:a3")
    assert any(event["decision"] == "cited" for event in bridgewater["audit"])
    assert any(event["decision"] == "selected" for event in bridgewater["audit"])
    assert refused.json()["scene"]["decisions"][0]["decision"] == "refused"
    assert refused.json()["answer"] == "I don't have enough evidence to answer."
    assert tesla.json()["asset"] == "TSLA"
    assert "robotaxi" in tesla.json()["answer"].lower()


def test_document_question_uses_the_library_instead_of_the_demo(client: TestClient) -> None:
    payload = (ROOT / "samples" / "api-errors.md").read_bytes()
    uploaded = client.post(
        "/api/documents",
        files={"file": ("api-errors.md", payload, "text/markdown")},
    )
    asked = client.post(
        "/api/trace/ask",
        json={"question": "Which file implements the retry logic?", "asset": "NVDA"},
    )
    body = asked.json()
    documents = {
        claim["provenance"]["filename"]
        for claim in body["scene"]["claims"]
        if claim["provenance"]["filename"]
    }

    assert uploaded.status_code == 201
    assert asked.status_code == 200
    assert body["asset"] == "DOCS"
    assert "retry.py" in body["answer"]
    assert body["markers"]
    assert "api-errors.md" in documents
    assert body["scene"]["selected_agent"] == "agent:DOCS:aggregator"
    assert any(agent["label"] == "api-errors.md" for agent in body["scene"]["agents"])
    cited_claim = next(claim for claim in body["scene"]["claims"] if claim["citation"] and "retry" in claim["citation"]["quote"].lower())
    assert cited_claim["citation"]["filename"] == "api-errors.md"
    assert any(event["decision"] == "supported" for event in cited_claim["audit"])
    assert body["scene"]["decisions"][0]["decision"] == "cited"


def test_unreal_engine_does_not_reuse_the_nvidia_summary(client: TestClient) -> None:
    overview = client.post(
        "/api/trace/ask",
        json={"question": "What are people saying about Unreal Engine?", "asset": "NVDA"},
    )
    plain = client.post(
        "/api/trace/ask",
        json={"question": "Tell me about Unreal Engine", "asset": "NVDA"},
    )
    nvidia = client.post(
        "/api/trace/ask",
        json={"question": "What are people saying about NVIDIA?", "asset": "NVDA"},
    )

    for response in (overview, plain):
        body = response.json()
        assert response.status_code == 200
        assert body["answer"] == "I don't have enough evidence to answer."
        assert body["markers"] == []
        assert "trillion-dollar" not in body["answer"]
        assert "trillion-dollar" in body["scene"]["answer"]
    assert "trillion-dollar" in nvidia.json()["answer"]
    assert nvidia.json()["markers"]
