"""Project an evidence graph into the source-trace scene."""

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone

from app.generation.grounding import AnswerResult
from app.ingestion.metadata_store import MetadataStore
from app.research.evidence import EvidenceRecord, SourceReference
from app.research.graph import (
    CLAIM,
    CONTRADICTS,
    ENTITY,
    EXTRACTED_FROM,
    HAS_SOURCE,
    HAS_VERSION,
    MENTIONS,
    PASSAGE,
    SUPPORTS,
    VERSION,
    EvidenceGraph,
    GraphNode,
    Provenance,
    ProvenanceEngine,
)
from app.research.verification import Contradiction

AGENT = "agent"
MEMBER_OF = "member_of"
FEEDS = "feeds"
SUPERSEDES = "supersedes"

_STOP = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "are",
        "of",
        "and",
        "or",
        "to",
        "for",
        "in",
        "on",
        "with",
        "that",
        "this",
        "about",
        "what",
        "are",
        "people",
        "saying",
        "how",
        "does",
        "do",
        "did",
        "their",
        "its",
        "from",
        "into",
        "while",
        "have",
        "has",
        "been",
        "being",
        "than",
        "then",
        "them",
        "they",
        "was",
        "were",
        "will",
        "would",
        "can",
        "could",
    }
)


@dataclass(frozen=True)
class _Doc:
    document_id: str
    filename: str
    version: int


@dataclass(frozen=True)
class _Claim:
    id: str
    agent: str
    text: str
    short: str
    document_id: str
    upstream: tuple[str, ...] = ()
    marker: int | None = None
    band: str = "above"


@dataclass(frozen=True)
class _Asset:
    asset: str
    name: str
    color: str
    question: str
    answer: str
    entities: tuple[str, ...]
    documents: tuple[_Doc, ...]
    claims: tuple[_Claim, ...]
    contradictions: tuple[tuple[str, str], ...]
    selected_claim: str


def _checksum(filename: str, version: int) -> str:
    return hashlib.sha256(f"{filename}\nv{version}".encode()).hexdigest()


def _nvda() -> _Asset:
    news = "doc-nvda-news"
    social = "doc-nvda-social"
    brief = "doc-nvda-brief"
    answer = (
        "NVIDIA is generating significant discussion in the market with mixed sentiments. "
        "The company is viewed as a leader in AI with strong growth potential, positioning it "
        "among elite companies approaching trillion-dollar market capitalization by 2030 [[1]]. "
        "However, there are growing concerns about a potential AI bubble in tech stocks [[2]]. "
        "While institutional investors like Bridgewater are reducing their positions, retail "
        "investors remain bullish on NVIDIA's long-term AI prospects [[3]]. The company faces "
        "valuation debates [[4]] and is drawing particular attention due to its upcoming earnings "
        "report, though there are some concerns about access to Chinese engineering talent [[5]]. "
        "Market participants note that the stock has experienced volatility while maintaining its "
        "market leadership position [[6]]."
    )
    return _Asset(
        asset="NVDA",
        name="NVIDIA",
        color="#3dd68c",
        question="What are people saying about NVIDIA?",
        answer=answer,
        entities=("NVIDIA", "Wall Street", "Ray Dalio", "Bridgewater"),
        documents=(
            _Doc(news, "nvda-news.md", 2),
            _Doc(social, "nvda-social.md", 1),
            _Doc(brief, "nvda-brief.md", 1),
        ),
        claims=(
            _Claim(
                "n1",
                "news",
                "NVIDIA is viewed as a leader in AI with strong growth potential, placing it among companies that could approach a trillion-dollar market capitalization by 2030.",
                "AI leader",
                news,
                marker=1,
            ),
            _Claim(
                "n2",
                "news",
                "Wall Street analysts are optimistic about NVIDIA's upcoming earnings despite recent market volatility.",
                "Analyst optimism",
                news,
            ),
            _Claim(
                "n3",
                "news",
                "Data center expansion continues to drive demand for NVIDIA's accelerators.",
                "Data centers",
                news,
            ),
            _Claim(
                "n4",
                "news",
                "The AI boom is pulling forward orders across NVIDIA's data-center products.",
                "AI boom",
                news,
            ),
            _Claim(
                "n5",
                "news",
                "Blockchain infrastructure projects are adding a further source of GPU demand for NVIDIA.",
                "Blockchain GPUs",
                news,
            ),
            _Claim(
                "n10",
                "news",
                "Some coverage flags concerns about NVIDIA's access to Chinese engineering talent.",
                "Talent access",
                news,
                marker=5,
            ),
            _Claim(
                "n6",
                "news",
                "NVIDIA has posted consistent earnings alongside the infrastructure buildout.",
                "Steady earnings",
                news,
                band="below",
            ),
            _Claim(
                "n7",
                "news",
                "NVIDIA maintains its dominance in the AI infrastructure stack used by cloud providers.",
                "AI dominance",
                news,
                band="below",
            ),
            _Claim(
                "n8",
                "news",
                "Price dips in NVIDIA are being bought quickly amid strong trading activity.",
                "Dips bought",
                news,
                band="below",
            ),
            _Claim(
                "n9",
                "news",
                "NVIDIA's valuation multiples look stretched in the current high-rate environment.",
                "High multiples",
                news,
                band="below",
            ),
            _Claim(
                "s1",
                "social",
                "Traders describe NVIDIA as a core AI holding ahead of the earnings report.",
                "Core holding",
                social,
            ),
            _Claim(
                "s2",
                "social",
                "Growing commentary treats the AI rally in mega-cap tech stocks as a potential bubble.",
                "AI bubble",
                social,
                marker=2,
            ),
            _Claim(
                "s5",
                "social",
                "Institutional desks say they are cutting tech exposure after the latest rally.",
                "Cutting exposure",
                social,
            ),
            _Claim(
                "s6",
                "social",
                "Ray Dalio's Bridgewater is reducing its exposure to NVIDIA and other tech stocks.",
                "Bridgewater cuts",
                social,
            ),
            _Claim(
                "s7",
                "social",
                "Valuation debates are dominating discussion until NVIDIA reports earnings.",
                "Valuation debate",
                social,
            ),
            _Claim(
                "s4",
                "social",
                "Retail investors remain bullish on NVIDIA's long-term AI prospects.",
                "Retail bullish",
                social,
                band="below",
            ),
            _Claim(
                "s3",
                "social",
                "The stock has experienced volatility while maintaining its market leadership position.",
                "Leadership holds",
                social,
                marker=6,
                band="below",
            ),
            _Claim(
                "s8",
                "social",
                "Computational growth is the theme retail users repeat when they defend NVIDIA.",
                "Compute growth",
                social,
                band="below",
            ),
            _Claim(
                "s9",
                "social",
                "Posts tie NVIDIA demand to broader computational growth rather than one product cycle.",
                "Demand posts",
                social,
                band="below",
            ),
            _Claim(
                "a1",
                "aggregator",
                "NVIDIA's upcoming earnings report is a critical market focus, with Wall Street analysts expressing optimism despite recent market volatility. The stock continues to experience strong trading activity with price dips being quickly bought.",
                "Earnings focus",
                brief,
                upstream=("n2", "n8", "s1", "s3"),
                band="panel",
            ),
            _Claim(
                "a2",
                "aggregator",
                "Data center expansion and the AI boom continue to drive strong demand for NVIDIA's products, supported by blockchain infrastructure developments and computational growth. The company maintains its dominance in AI infrastructure while showing consistent earnings performance.",
                "AI demand",
                brief,
                upstream=("n3", "n4", "n5", "n6", "n7", "s8", "s9"),
                band="panel",
            ),
            _Claim(
                "a3",
                "aggregator",
                "Major institutional investors, including Ray Dalio's Bridgewater, are reducing their exposure to NVIDIA and other tech stocks, while retail investors maintain bullish sentiment.",
                "Bridgewater vs retail",
                brief,
                upstream=("s5", "s6", "s4", "s2", "n2"),
                marker=3,
                band="panel",
            ),
            _Claim(
                "a4",
                "aggregator",
                "Concerns persist regarding NVIDIA's high valuation multiples in the current high-rate environment, along with worries about stretched AI valuations.",
                "Valuation concerns",
                brief,
                upstream=("n9", "s7", "s2"),
                marker=4,
                band="panel",
            ),
        ),
        contradictions=(("s6", "s4"),),
        selected_claim="a3",
    )


def _tsla() -> _Asset:
    news = "doc-tsla-news"
    social = "doc-tsla-social"
    brief = "doc-tsla-brief"
    answer = (
        "Tesla is being discussed as a robotaxi story more than a car-delivery story [[1]]. "
        "Price cuts are still the pressure point on automotive margins [[2]], while retail "
        "traders treat share-price dips as entries [[3]]."
    )
    return _Asset(
        asset="TSLA",
        name="Tesla",
        color="#ef4444",
        question="What are people saying about Tesla?",
        answer=answer,
        entities=("Tesla",),
        documents=(
            _Doc(news, "tsla-news.md", 1),
            _Doc(social, "tsla-social.md", 1),
            _Doc(brief, "tsla-brief.md", 1),
        ),
        claims=(
            _Claim(
                "tn1",
                "news",
                "Tesla's robotaxi timeline is the main bull case in current coverage.",
                "Robotaxi case",
                news,
                marker=1,
            ),
            _Claim(
                "tn2",
                "news",
                "Price cuts continue to pressure Tesla's automotive margins.",
                "Margin pressure",
                news,
                marker=2,
            ),
            _Claim(
                "tn3",
                "news",
                "Delivery volatility is the near-term risk cited alongside the autonomy narrative.",
                "Delivery swings",
                news,
                band="below",
            ),
            _Claim(
                "ts1",
                "social",
                "Retail traders treat Tesla share-price dips as buying opportunities.",
                "Dip buyers",
                social,
                marker=3,
            ),
            _Claim(
                "ts2",
                "social",
                "Some holders say they are trimming Tesla after the last rally.",
                "Trimming rally",
                social,
                band="below",
            ),
            _Claim(
                "ta1",
                "aggregator",
                "Tesla's robotaxi narrative is driving retail optimism while price cuts pressure automotive margins.",
                "Robotaxi vs margins",
                brief,
                upstream=("tn1", "tn2", "ts1"),
                band="panel",
            ),
            _Claim(
                "ta2",
                "aggregator",
                "Delivery volatility and trimmed exposure after rallies are the main near-term concerns on Tesla.",
                "Near-term risk",
                brief,
                upstream=("tn3", "ts2"),
                band="panel",
            ),
        ),
        contradictions=(("ts1", "ts2"),),
        selected_claim="ta1",
    )


def _gold() -> _Asset:
    news = "doc-gold-news"
    social = "doc-gold-social"
    brief = "doc-gold-brief"
    answer = (
        "Gold is supported by official-sector buying and geopolitical hedging [[1]]. "
        "Higher real yields are the case against chasing the move [[2]]."
    )
    return _Asset(
        asset="GOLD",
        name="Gold",
        color="#e2b340",
        question="What are people saying about gold?",
        answer=answer,
        entities=("Gold",),
        documents=(
            _Doc(news, "gold-news.md", 1),
            _Doc(social, "gold-social.md", 1),
            _Doc(brief, "gold-brief.md", 1),
        ),
        claims=(
            _Claim(
                "gn1",
                "news",
                "Central banks continue to buy gold, adding a steady official-sector bid.",
                "Official buying",
                news,
                marker=1,
            ),
            _Claim(
                "gn2",
                "news",
                "Higher real yields are competing with the bull case for gold.",
                "Real yields",
                news,
                marker=2,
                band="below",
            ),
            _Claim(
                "gs1",
                "social",
                "Geopolitical hedging is the reason retail buyers give for holding gold.",
                "Hedging bid",
                social,
            ),
            _Claim(
                "gs2",
                "social",
                "Some traders say real yields make gold less attractive at these prices.",
                "Yield caution",
                social,
                band="below",
            ),
            _Claim(
                "ga1",
                "aggregator",
                "Official-sector buying and geopolitical hedging are supporting gold.",
                "Official support",
                brief,
                upstream=("gn1", "gs1"),
                band="panel",
            ),
            _Claim(
                "ga2",
                "aggregator",
                "Higher real yields are the main contradiction to the bull case for gold.",
                "Yield challenge",
                brief,
                upstream=("gn2", "gs2"),
                band="panel",
            ),
        ),
        contradictions=(("gn1", "gn2"),),
        selected_claim="ga1",
    )


_SPECS: dict[str, _Asset] = {spec.asset: spec for spec in (_nvda(), _tsla(), _gold())}

_ROLES = (
    ("news", "News Agent"),
    ("social", "Social Agent"),
    ("aggregator", "Aggregator"),
)


class _Catalog:
    def __init__(self, rows: dict[str, object]) -> None:
        self._rows = rows

    def get(self, document_id: str) -> object | None:
        return self._rows.get(document_id)


@dataclass
class _Row:
    filename: str
    content_type: str
    checksum: str
    indexed_version: int
    created_at: str = ""
    updated_at: str = ""


def _spans(spec: _Asset, document_id: str) -> dict[str, tuple[int, int]]:
    cursor = 0
    found: dict[str, tuple[int, int]] = {}
    for claim in spec.claims:
        if claim.document_id != document_id:
            continue
        if found:
            cursor += 2
        found[claim.id] = (cursor, cursor + len(claim.text))
        cursor += len(claim.text)
    return found


def build_asset_graph(asset: str) -> EvidenceGraph:
    """Connect documents, chunks, claims, entities, and source versions for one asset."""
    spec = _SPECS[asset]
    docs = {item.document_id: item for item in spec.documents}
    spans = {item.document_id: _spans(spec, item.document_id) for item in spec.documents}
    rows = {
        item.document_id: _Row(
            filename=item.filename,
            content_type="text/markdown",
            checksum=_checksum(item.filename, item.version),
            indexed_version=item.version,
            created_at="2026-11-03T00:00:00+00:00" if item.version >= 2 else "2026-10-01T00:00:00+00:00",
            updated_at="2026-11-03T00:00:00+00:00" if item.version >= 2 else "2026-10-01T00:00:00+00:00",
        )
        for item in spec.documents
    }
    records: list[EvidenceRecord] = []
    for claim in spec.claims:
        document = docs[claim.document_id]
        start, end = spans[claim.document_id][claim.id]
        records.append(
            EvidenceRecord(
                evidence_id=f"ev-{claim.id}",
                claim_id=claim.id,
                claim=claim.text,
                passage_id=f"chunk-{claim.id}",
                source=SourceReference(
                    document_id=document.document_id,
                    chunk_id=f"chunk-{claim.id}",
                    filename=document.filename,
                    content_type="text/markdown",
                    heading=claim.short,
                    page=None,
                    start_char=start,
                    end_char=end,
                ),
                score=1.0,
                created_at="2026-11-03T00:00:00+00:00",
                verdict="supported",
                support=1.0,
            )
        )
    conflicts = [
        Contradiction(
            summary="These claims disagree.",
            left_filename=docs[next(item.document_id for item in spec.claims if item.id == left)].filename,
            right_filename=docs[next(item.document_id for item in spec.claims if item.id == right)].filename,
            left_values=(),
            right_values=(),
            left_claim_id=left,
            right_claim_id=right,
        )
        for left, right in spec.contradictions
    ]
    engine = ProvenanceEngine(_Catalog(rows))
    graph = engine.from_research(records, conflicts)
    for claim in spec.claims:
        for upstream in claim.upstream:
            graph.add_edge(f"claim:{upstream}", SUPPORTS, f"claim:{claim.id}")
    for role, label in _ROLES:
        graph.add_node(GraphNode(id=f"agent:{asset}:{role}", kind=AGENT, label=label))
    graph.add_edge(f"agent:{asset}:news", FEEDS, f"agent:{asset}:aggregator")
    graph.add_edge(f"agent:{asset}:social", FEEDS, f"agent:{asset}:aggregator")
    for claim in spec.claims:
        graph.add_edge(f"claim:{claim.id}", MEMBER_OF, f"agent:{asset}:{claim.agent}")
    _prior_versions(graph, spec)
    engine.attach_entities(graph, spec.entities)
    return graph


def _prior_versions(graph: EvidenceGraph, spec: _Asset) -> None:
    for document in spec.documents:
        if document.version < 2:
            continue
        current_id = f"version:{document.document_id}:v{document.version}"
        current = graph.nodes.get(current_id)
        if current is None:
            continue
        prior = Provenance(
            document_id=document.document_id,
            version=document.version - 1,
            filename=document.filename,
            content_type="text/markdown",
            checksum=_checksum(document.filename, document.version - 1),
            recorded_at="2026-10-01T00:00:00+00:00",
        )
        prior_id = f"version:{document.document_id}:v{document.version - 1}"
        graph.add_node(
            GraphNode(
                id=prior_id,
                kind=VERSION,
                label=f"{document.filename} v{document.version - 1}",
                provenance=prior,
            )
        )
        graph.add_edge(document_node(document.document_id), HAS_VERSION, prior_id)
        if graph.add_edge(current_id, SUPERSEDES, prior_id):
            ProvenanceEngine().record_decision(
                graph,
                "superseded",
                current_id,
                (prior_id,),
                detail=f"{document.filename} v{document.version} supersedes v{document.version - 1}",
                recorded_at="2026-11-03T00:00:00+00:00",
            )


def document_node(document_id: str) -> str:
    return f"document:{document_id}"


def trace_scene(asset: str) -> dict | None:
    """Serialize one asset graph for the source-trace visualiser."""
    key = asset.strip().upper()
    spec = _SPECS.get(key)
    if spec is None:
        return None
    graph = build_asset_graph(key)
    by_spec = {item.id: item for item in spec.claims}
    claims = []
    for item in spec.claims:
        node = graph.nodes[f"claim:{item.id}"]
        upstream = _incoming_claims(graph, node.id)
        claims.append(
            {
                "id": node.id,
                "text": node.label,
                "short": item.short,
                "agent_id": f"agent:{key}:{item.agent}",
                "band": item.band,
                "upstream": upstream,
                "upstream_count": len(upstream),
                "marker": item.marker,
                "entities": _mentioned(graph, node.id),
                "contradicts": _related(graph, node.id, CONTRADICTS),
                "provenance": _provenance(graph, node),
                "citation": _citation(graph, node.id, item.marker),
                "audit": _audit_for(graph, node),
            }
        )
    return {
        "asset": key,
        "assets": [
            {"id": item.asset, "name": item.name, "color": item.color} for item in _SPECS.values()
        ],
        "question": spec.question,
        "answer": spec.answer,
        "markers": sorted(
            (
                {"n": item.marker, "claim_id": f"claim:{item.id}"}
                for item in spec.claims
                if item.marker is not None
            ),
            key=lambda item: item["n"],
        ),
        "agents": [
            {"id": f"agent:{key}:{role}", "label": label, "role": role} for role, label in _ROLES
        ],
        "entities": [
            {"id": node.id, "label": node.label}
            for node in sorted(
                (node for node in graph.nodes.values() if node.kind == ENTITY),
                key=lambda node: node.label,
            )
        ],
        "claims": claims,
        "chunks": _chunks(graph, key, by_spec),
        "edges": [
            {"source": edge.source, "relation": edge.relation, "target": edge.target}
            for edge in graph.edges
        ],
        "selected_agent": f"agent:{key}:aggregator",
        "selected_claim": f"claim:{spec.selected_claim}",
    }


LIBRARY = "DOCS"


def named_demo_asset(question: str) -> str | None:
    """The demo asset named in the question, if the question is about one."""
    text = question.lower()
    if "nvidia" in text or re.search(r"\bnvda\b", text):
        return "NVDA"
    if "tesla" in text or re.search(r"\btsla\b", text):
        return "TSLA"
    if re.search(r"\bgold\b", text):
        return "GOLD"
    return None


def library_scene(result: AnswerResult, metadata: MetadataStore | None = None) -> dict | None:
    """Turn a grounded document answer into the same trace scene as the demos."""
    if not result.sufficient_evidence or not result.citations:
        return None
    cited: list[tuple[int, object, object]] = []
    for citation in result.citations:
        number = int(str(citation.marker).strip("[]"))
        if number < 1 or number > len(result.passages):
            continue
        cited.append((number, citation, result.passages[number - 1]))
    if not cited:
        return None
    documents: list[str] = []
    for _, _, passage in cited:
        document_id = passage.source.document_id
        if document_id not in documents:
            documents.append(document_id)
    role_for = _source_roles(documents)
    records: list[EvidenceRecord] = []
    specs: dict[str, _Claim] = {}
    counts = {"news": 0, "social": 0}
    for number, citation, passage in cited:
        source = passage.source
        role = role_for.get(source.document_id, "news")
        counts[role] += 1
        claim_id = f"lib-{number}"
        quote = (citation.quote or passage.model_text() or passage.text).strip()
        records.append(
            EvidenceRecord(
                evidence_id=f"ev-{claim_id}",
                claim_id=claim_id,
                claim=quote,
                passage_id=source.chunk_id,
                source=SourceReference(
                    document_id=source.document_id,
                    chunk_id=source.chunk_id,
                    filename=source.filename,
                    content_type=source.content_type,
                    heading=source.heading,
                    page=source.page,
                    start_char=source.start_char,
                    end_char=source.end_char,
                ),
                score=float(passage.score),
                created_at="2026-11-03T00:00:00+00:00",
                verdict="supported",
                support=1.0,
            )
        )
        specs[claim_id] = _Claim(
            id=claim_id,
            agent=role,
            text=quote,
            short=_short_label(quote, source.heading),
            document_id=source.document_id,
            marker=number,
            band="above" if counts[role] <= 3 else "below",
        )
    for passage in _related_passages(result, documents, {item[2].source.chunk_id for item in cited}):
        source = passage.source
        role = role_for.get(source.document_id, "news")
        counts[role] += 1
        claim_id = f"lib-x{counts[role]}"
        quote = (passage.model_text() or passage.text).strip()
        records.append(
            EvidenceRecord(
                evidence_id=f"ev-{claim_id}",
                claim_id=claim_id,
                claim=quote,
                passage_id=source.chunk_id,
                source=SourceReference(
                    document_id=source.document_id,
                    chunk_id=source.chunk_id,
                    filename=source.filename,
                    content_type=source.content_type,
                    heading=source.heading,
                    page=source.page,
                    start_char=source.start_char,
                    end_char=source.end_char,
                ),
                score=float(passage.score),
                created_at="2026-11-03T00:00:00+00:00",
                verdict="supported",
                support=1.0,
            )
        )
        specs[claim_id] = _Claim(
            id=claim_id,
            agent=role,
            text=quote,
            short=_short_label(quote, source.heading),
            document_id=source.document_id,
            band="below",
        )
    engine = ProvenanceEngine(metadata)
    graph = engine.from_research(tuple(records))
    plain = re.sub(r"\s*\[\d+\]", "", result.answer).strip()
    first = graph.nodes[f"claim:lib-{cited[0][0]}"]
    graph.add_node(GraphNode(id="claim:lib-answer", kind=CLAIM, label=plain, provenance=first.provenance))
    upstream = []
    for number, _, _ in cited:
        claim_id = f"claim:lib-{number}"
        upstream.append(claim_id)
        graph.add_edge(claim_id, SUPPORTS, "claim:lib-answer")
    if result.conflict and len(cited) >= 2:
        left = f"claim:lib-{cited[0][0]}"
        right = f"claim:lib-{cited[1][0]}"
        graph.add_edge(left, CONTRADICTS, right)
        graph.add_edge(right, CONTRADICTS, left)
    roles = _library_agents(cited, role_for)
    for role, label in roles:
        graph.add_node(GraphNode(id=f"agent:{LIBRARY}:{role}", kind=AGENT, label=label))
    graph.add_node(GraphNode(id=f"agent:{LIBRARY}:aggregator", kind=AGENT, label="Aggregator"))
    for role, _label in roles:
        graph.add_edge(f"agent:{LIBRARY}:{role}", FEEDS, f"agent:{LIBRARY}:aggregator")
    for claim_id, spec in specs.items():
        graph.add_edge(f"claim:{claim_id}", MEMBER_OF, f"agent:{LIBRARY}:{spec.agent}")
    graph.add_edge("claim:lib-answer", MEMBER_OF, f"agent:{LIBRARY}:aggregator")
    specs["lib-answer"] = _Claim(
        id="lib-answer",
        agent="aggregator",
        text=plain,
        short="Answer",
        document_id=cited[0][2].source.document_id,
        upstream=tuple(item.removeprefix("claim:") for item in upstream),
        band="panel",
    )
    for spec in list(specs.values()):
        if spec.agent == "aggregator":
            continue
        panel_id = f"agg-{spec.id}"
        source_node = graph.nodes[f"claim:{spec.id}"]
        graph.add_node(
            GraphNode(id=f"claim:{panel_id}", kind=CLAIM, label=spec.text, provenance=source_node.provenance)
        )
        graph.add_edge(f"claim:{spec.id}", SUPPORTS, f"claim:{panel_id}")
        graph.add_edge(f"claim:{panel_id}", MEMBER_OF, f"agent:{LIBRARY}:aggregator")
        specs[panel_id] = _Claim(
            id=panel_id,
            agent="aggregator",
            text=spec.text,
            short=spec.short,
            document_id=spec.document_id,
            upstream=(spec.id,),
            band="panel",
        )
    claims = []
    for spec in specs.values():
        node = graph.nodes[f"claim:{spec.id}"]
        incoming = _incoming_claims(graph, node.id)
        claims.append(
            {
                "id": node.id,
                "text": node.label,
                "short": spec.short,
                "agent_id": f"agent:{LIBRARY}:{spec.agent}",
                "band": spec.band,
                "upstream": incoming,
                "upstream_count": len(incoming),
                "marker": spec.marker,
                "entities": _mentioned(graph, node.id),
                "contradicts": _related(graph, node.id, CONTRADICTS),
                "provenance": _provenance(graph, node),
                "citation": _citation(graph, node.id, spec.marker)
                or _citation(graph, f"claim:{spec.upstream[0]}", spec.marker)
                if spec.upstream
                else _citation(graph, node.id, spec.marker),
                "audit": _audit_for(
                    graph,
                    graph.nodes.get(f"claim:{spec.upstream[0]}", node) if spec.upstream else node,
                ),
            }
        )
    answer = re.sub(r"\[(\d+)\]", r"[[\1]]", result.answer)
    scene = {
        "asset": LIBRARY,
        "assets": [
            {"id": item.asset, "name": item.name, "color": item.color} for item in _SPECS.values()
        ],
        "question": result.query,
        "answer": answer,
        "markers": [
            {"n": spec.marker, "claim_id": f"claim:{spec.id}"}
            for spec in specs.values()
            if spec.marker is not None
        ],
        "agents": [
            {"id": f"agent:{LIBRARY}:{role}", "label": label, "role": role} for role, label in roles
        ]
        + [{"id": f"agent:{LIBRARY}:aggregator", "label": "Aggregator", "role": "aggregator"}],
        "entities": [],
        "claims": claims,
        "chunks": _library_chunks(graph, specs),
        "edges": [
            {"source": edge.source, "relation": edge.relation, "target": edge.target} for edge in graph.edges
        ],
        "selected_agent": f"agent:{LIBRARY}:aggregator",
        "selected_claim": "claim:lib-answer",
    }
    _note_decisions(scene, scene["markers"], "cited")
    return scene


def _related_passages(result: AnswerResult, documents: list[str], used_chunks: set[str]) -> list:
    """Other passages from the same documents, so the aggregator can list more than the quote."""
    found = []
    seen = set(used_chunks)
    allowed = set(documents)
    for passage in result.passages:
        source = passage.source
        if source.document_id not in allowed or source.chunk_id in seen:
            continue
        seen.add(source.chunk_id)
        found.append(passage)
        if len(found) >= 4:
            break
    return found


def _source_roles(documents: list[str]) -> dict[str, str]:
    if len(documents) <= 1:
        return {document_id: "news" for document_id in documents}
    midpoint = (len(documents) + 1) // 2
    return {
        document_id: "news" if index < midpoint else "social"
        for index, document_id in enumerate(documents)
    }


def _library_agents(cited: list[tuple[int, object, object]], role_for: dict[str, str]) -> list[tuple[str, str]]:
    files: dict[str, list[str]] = {"news": [], "social": []}
    for _, _, passage in cited:
        role = role_for.get(passage.source.document_id, "news")
        filename = passage.source.filename
        if filename not in files[role]:
            files[role].append(filename)
    agents = []
    for role in ("news", "social"):
        names = files[role]
        if not names:
            continue
        label = names[0] if len(names) == 1 else "Sources"
        agents.append((role, label))
    return agents


def _library_chunks(graph: EvidenceGraph, specs: dict[str, _Claim]) -> list[dict]:
    rows = []
    for node in graph.nodes.values():
        if node.kind != PASSAGE:
            continue
        spec = next((item for item in specs.values() if item.document_id == node.provenance.document_id and item.band != "panel"), None)
        if spec is None:
            continue
        rows.append(
            {
                "id": node.id,
                "agent_id": f"agent:{LIBRARY}:{spec.agent}",
                "filename": node.provenance.filename,
                "version": node.provenance.version,
                "label": node.provenance.heading or node.label,
                "checksum": node.provenance.checksum,
                "chunk_id": node.provenance.chunk_id,
                "document_id": node.provenance.document_id,
            }
        )
    return rows


def _short_label(text: str, heading: str | None) -> str:
    if heading and heading.strip():
        return heading.strip()[:24]
    words = text.split()
    return " ".join(words[:3])[:24] or "Passage"


def answer_from_trace(question: str, asset: str | None = None) -> dict | None:
    """Answer from claims already on the evidence graph."""
    chosen = _detect_asset(question, (asset or "NVDA").strip().upper())
    scene = trace_scene(chosen)
    if scene is None:
        return None
    if _is_asset_overview(question, _SPECS[chosen]):
        _note_decisions(scene, scene["markers"], "cited")
        return {
            "asset": chosen,
            "answer": scene["answer"],
            "markers": scene["markers"],
            "scene": scene,
        }
    picked = _match_claims(question, scene["claims"], _SPECS[chosen])
    if not picked:
        _note_decisions(scene, [], "refused")
        return {
            "asset": chosen,
            "answer": "I don't have enough evidence to answer.",
            "markers": [],
            "scene": scene,
        }
    markers = [{"n": index, "claim_id": claim["id"]} for index, claim in enumerate(picked, start=1)]
    text = " ".join(f"{claim['text']} [[{index}]]" for index, claim in enumerate(picked, start=1))
    focused = _focus_scene(scene, text, markers)
    _note_decisions(focused, markers, "cited")
    return {"asset": chosen, "answer": text, "markers": markers, "scene": focused}


def _focus_scene(scene: dict, answer: str, markers: list[dict]) -> dict:
    """Keep the claims that support this answer, each as its own aggregator card."""
    by_id = {claim["id"]: claim for claim in scene["claims"]}
    cited = [item["claim_id"] for item in markers if item["claim_id"] in by_id]
    if not cited:
        return {**scene, "answer": answer, "markers": markers}
    visible = set(cited)
    for claim_id in cited:
        visible.update(by_id[claim_id]["upstream"])
    aggregator = next(agent["id"] for agent in scene["agents"] if agent["role"] == "aggregator")
    cards = _aggregator_cards(by_id, cited, aggregator)
    claims = [by_id[claim_id] for claim_id in visible if claim_id in by_id]
    claims.extend(cards)
    claim_ids = {claim["id"] for claim in claims}
    agent_ids = {claim["agent_id"] for claim in claims}
    agents = [agent for agent in scene["agents"] if agent["id"] in agent_ids]
    filenames = {
        claim["provenance"]["filename"] for claim in claims if claim["provenance"]["filename"]
    }
    selected = next(
        (
            claim["id"]
            for claim_id in cited
            if (claim := by_id[claim_id])["band"] == "panel" and claim["agent_id"] == aggregator
        ),
        cards[0]["id"] if cards else cited[0],
    )
    return {
        **scene,
        "answer": answer,
        "markers": markers,
        "agents": agents,
        "claims": claims,
        "chunks": [
            chunk
            for chunk in scene["chunks"]
            if chunk["agent_id"] in agent_ids and chunk["filename"] in filenames
        ],
        "edges": _focused_edges(scene["edges"], claim_ids, agent_ids, cards),
        "entities": [
            entity
            for entity in scene["entities"]
            if entity["label"] in {name for claim in claims for name in claim["entities"]}
        ],
        "selected_agent": aggregator,
        "selected_claim": selected,
    }


def _aggregator_cards(by_id: dict[str, dict], cited: list[str], aggregator: str) -> list[dict]:
    """One aggregator card for every source claim behind the answer."""
    seen: set[str] = set()
    cards: list[dict] = []
    queue: list[str] = []
    for claim_id in cited:
        claim = by_id[claim_id]
        queue.append(claim_id)
        if claim["band"] == "panel":
            queue.extend(claim["upstream"])
    for claim_id in queue:
        if claim_id in seen or claim_id not in by_id or len(cards) >= 7:
            continue
        seen.add(claim_id)
        claim = by_id[claim_id]
        if claim["band"] == "panel" and claim["agent_id"] == aggregator:
            continue
        cards.append(
            {
                "id": f"claim:agg-{claim_id.removeprefix('claim:')}",
                "text": claim["text"],
                "short": claim["short"],
                "agent_id": aggregator,
                "band": "panel",
                "upstream": [claim_id],
                "upstream_count": 1,
                "marker": claim["marker"],
                "entities": claim["entities"],
                "contradicts": [],
                "provenance": claim["provenance"],
                "citation": claim.get("citation"),
                "audit": list(claim.get("audit") or []),
            }
        )
    return cards


def _focused_edges(
    edges: list[dict],
    claim_ids: set[str],
    agent_ids: set[str],
    cards: list[dict],
) -> list[dict]:
    extra: set[str] = set()
    for edge in edges:
        if edge["target"] in claim_ids and not edge["source"].startswith(("claim:", "agent:")):
            extra.add(edge["source"])
        if edge["source"] in claim_ids and edge["target"].startswith(("entity:", "document:", "version:")):
            extra.add(edge["target"])
    keep = set(claim_ids) | set(agent_ids) | extra
    kept = [edge for edge in edges if edge["source"] in keep and edge["target"] in keep]
    for card in cards:
        for claim_id in card["upstream"]:
            kept.append({"source": claim_id, "relation": SUPPORTS, "target": card["id"]})
    return kept


def _detect_asset(question: str, fallback: str) -> str:
    return named_demo_asset(question) or (fallback if fallback in _SPECS else "NVDA")


_OVERVIEW_PHRASES = (
    "what are people saying",
    "what are they saying",
    "sentiment",
    "overview",
    "summarize",
    "summary",
)
_OVERVIEW_FILLER = frozenset(
    {
        "sentiment",
        "overview",
        "summarize",
        "summary",
        "market",
        "stock",
        "stocks",
        "company",
        "latest",
        "update",
        "updates",
        "news",
        "discuss",
        "discussion",
        "think",
        "view",
        "views",
    }
)


def _overview(question: str) -> bool:
    text = question.lower()
    return any(phrase in text for phrase in _OVERVIEW_PHRASES)


def _is_asset_overview(question: str, spec: _Asset) -> bool:
    """True when the question asks about this asset and names no other subject."""
    names = _asset_names(spec)
    if _words(question) - names - _OVERVIEW_FILLER:
        return False
    return _overview(question) or bool(_words(question) & names)


def _asset_names(spec: _Asset) -> set[str]:
    return _words(spec.name) | {spec.asset.lower()}


def _words(text: str) -> set[str]:
    return {word for word in re.findall(r"[a-z0-9]+", text.lower()) if word not in _STOP and len(word) > 2}


def _match_claims(question: str, claims: list[dict], spec: _Asset) -> list[dict]:
    wanted = _words(question) - _asset_names(spec)
    if not wanted:
        return []
    ranked: list[tuple[int, int, dict]] = []
    for claim in claims:
        overlap = wanted & _words(claim["text"])
        if not overlap:
            continue
        panel_first = 0 if claim["band"] == "panel" else 1
        ranked.append((len(overlap), -panel_first, claim))
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [claim for _, _, claim in ranked[:3]]


def _incoming_claims(graph: EvidenceGraph, claim_id: str) -> list[str]:
    found = []
    for edge in graph.edges:
        if edge.target != claim_id or edge.relation != SUPPORTS:
            continue
        node = graph.nodes.get(edge.source)
        if node is not None and node.kind == CLAIM:
            found.append(node.id)
    return found


def _related(graph: EvidenceGraph, claim_id: str, relation: str) -> list[str]:
    found = []
    for edge in graph.edges:
        if edge.source == claim_id and edge.relation == relation:
            node = graph.nodes.get(edge.target)
            if node is not None and node.kind == CLAIM:
                found.append(node.id)
    return found


def _mentioned(graph: EvidenceGraph, claim_id: str) -> list[str]:
    names = []
    for edge in graph.edges:
        if edge.source != claim_id or edge.relation != MENTIONS:
            continue
        node = graph.nodes.get(edge.target)
        if node is not None and node.kind == ENTITY:
            names.append(node.label)
    return names


def _provenance(graph: EvidenceGraph, node: GraphNode) -> dict:
    origin = node.provenance
    versions = [
        item
        for item in graph.nodes.values()
        if item.kind == VERSION and item.provenance.document_id == origin.document_id
    ]
    versions.sort(key=lambda item: item.provenance.version)
    return {
        "document_id": origin.document_id,
        "filename": origin.filename,
        "version": origin.version,
        "checksum": origin.checksum,
        "chunk_id": origin.chunk_id,
        "heading": origin.heading,
        "start_char": origin.start_char,
        "end_char": origin.end_char,
        "recorded_at": origin.recorded_at,
        "lineage": _lineage(graph, node),
        "history": [
            {
                "version": item.provenance.version,
                "checksum": item.provenance.checksum,
                "label": item.label,
                "recorded_at": item.provenance.recorded_at,
            }
            for item in versions
        ],
    }


def _lineage(graph: EvidenceGraph, node: GraphNode) -> list[dict]:
    """Claim, passage, source, document, then each version in order."""
    steps: list[dict] = []
    seen: set[str] = set()

    def add(item: GraphNode | None) -> None:
        if item is None or item.id in seen:
            return
        seen.add(item.id)
        origin = item.provenance
        steps.append(
            {
                "kind": item.kind,
                "id": item.id,
                "label": item.label,
                "version": origin.version,
                "recorded_at": origin.recorded_at,
                "checksum": origin.checksum,
            }
        )

    add(node)
    passage = graph.nodes.get(f"passage:{node.provenance.chunk_id}") if node.provenance.chunk_id else None
    add(passage)
    if passage is not None:
        for relation in (HAS_SOURCE, EXTRACTED_FROM):
            for nxt in graph.neighbors(passage.id, relation):
                add(nxt)
    if node.provenance.document_id:
        add(graph.nodes.get(f"document:{node.provenance.document_id}"))
        versions = [
            item
            for item in graph.nodes.values()
            if item.kind == VERSION and item.provenance.document_id == node.provenance.document_id
        ]
        versions.sort(key=lambda item: item.provenance.version)
        for item in versions:
            add(item)
    return steps


def _citation(graph: EvidenceGraph, claim_id: str, marker: int | None) -> dict | None:
    found = next((item for item in graph.citations if item.claim_id == claim_id), None)
    if found is None:
        return None
    return {
        "marker": marker if marker is not None else found.marker,
        "quote": found.quote,
        "filename": found.filename,
        "version": found.version,
        "chunk_id": found.chunk_id,
        "start_char": found.start_char,
        "end_char": found.end_char,
        "recorded_at": found.recorded_at,
    }


def _audit_for(graph: EvidenceGraph, node: GraphNode) -> list[dict]:
    document_id = node.provenance.document_id
    events = []
    for event in graph.audit:
        if event.subject_id == node.id:
            events.append(event)
            continue
        if event.decision != "superseded" or not document_id:
            continue
        version = graph.nodes.get(event.subject_id)
        if version is not None and version.provenance.document_id == document_id:
            events.append(event)
    return [
        {
            "decision": event.decision,
            "detail": event.detail,
            "recorded_at": event.recorded_at,
            "evidence_ids": list(event.evidence_ids),
        }
        for event in events
    ]


def _note_decisions(scene: dict, markers: list[dict], decision: str) -> None:
    """Record that this answer cited claims, or that the question was refused."""
    stamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    by_id = {claim["id"]: claim for claim in scene["claims"]}
    evidence: list[str] = []
    for marker in markers:
        claim = by_id.get(marker["claim_id"])
        if claim is None:
            continue
        evidence.append(marker["claim_id"])
        claim.setdefault("audit", []).append(
            {
                "decision": decision,
                "detail": f"Citation [[{marker['n']}]] included in the answer",
                "recorded_at": stamp,
                "evidence_ids": [claim.get("provenance", {}).get("chunk_id", "")],
            }
        )
    selected = by_id.get(scene.get("selected_claim", ""))
    if decision == "cited" and selected is not None:
        selected.setdefault("audit", []).append(
            {
                "decision": "selected",
                "detail": "Chosen as the answer claim",
                "recorded_at": stamp,
                "evidence_ids": evidence,
            }
        )
    scene.setdefault("decisions", []).append(
        {
            "decision": decision,
            "detail": "Not enough evidence to answer."
            if decision == "refused"
            else f"{len(evidence)} claim citation(s) included in the answer",
            "recorded_at": stamp,
            "evidence_ids": evidence,
        }
    )


def _chunks(graph: EvidenceGraph, asset: str, claims: dict[str, _Claim]) -> list[dict]:
    rows = []
    for node in graph.nodes.values():
        if node.kind != PASSAGE:
            continue
        claim_key = node.provenance.chunk_id.removeprefix("chunk-")
        spec = claims.get(claim_key)
        if spec is None:
            continue
        rows.append(
            {
                "id": node.id,
                "agent_id": f"agent:{asset}:{spec.agent}",
                "filename": node.provenance.filename,
                "version": node.provenance.version,
                "label": node.provenance.heading or node.label,
                "checksum": node.provenance.checksum,
                "chunk_id": node.provenance.chunk_id,
                "document_id": node.provenance.document_id,
            }
        )
    return rows

