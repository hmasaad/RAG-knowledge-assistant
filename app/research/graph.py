"""A traceable graph of claims, passages, documents, versions, sources, and entities."""

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

from app.ingestion.metadata_store import MetadataStore
from app.research.evidence import EvidenceRecord
from app.research.judge import EvidenceItem, JudgeVerdict
from app.research.verification import Contradiction

CLAIM = "claim"
PASSAGE = "passage"
DOCUMENT = "document"
VERSION = "version"
SOURCE = "source"
ENTITY = "entity"
SUPPORTS = "supports"
CONTRADICTS = "contradicts"
EXTRACTED_FROM = "extracted_from"
HAS_SOURCE = "has_source"
HAS_VERSION = "has_version"
CITED_IN = "cited_in"
MENTIONS = "mentions"
SUPERSEDES = "supersedes"


@dataclass(frozen=True)
class Provenance:
    """Where a node came from in the indexed collection."""

    document_id: str = ""
    version: int = 0
    filename: str = ""
    content_type: str = ""
    checksum: str = ""
    chunk_id: str = ""
    heading: str | None = None
    page: int | None = None
    start_char: int = 0
    end_char: int = 0
    recorded_at: str = ""


@dataclass(frozen=True)
class ClaimCitation:
    """The passage a claim cites, including the character span and version."""

    claim_id: str
    quote: str
    document_id: str
    filename: str
    version: int
    chunk_id: str
    start_char: int
    end_char: int
    recorded_at: str = ""
    marker: int | None = None


@dataclass(frozen=True)
class AuditEvent:
    """One recorded decision: support, contradiction, citation, or refusal."""

    id: str
    decision: str
    subject_id: str
    evidence_ids: tuple[str, ...] = ()
    detail: str = ""
    recorded_at: str = ""


@dataclass(frozen=True)
class GraphNode:
    id: str
    kind: str
    label: str
    provenance: Provenance = field(default_factory=Provenance)


@dataclass(frozen=True)
class ProvenanceEdge:
    source: str
    relation: str
    target: str


class EvidenceGraph:
    """Directed graph of documents, chunks, claims, and the entities they name."""

    def __init__(self) -> None:
        self.nodes: dict[str, GraphNode] = {}
        self.edges: list[ProvenanceEdge] = []
        self.citations: list[ClaimCitation] = []
        self.audit: list[AuditEvent] = []
        self._seen: set[tuple[str, str, str]] = set()

    def add_node(self, node: GraphNode) -> GraphNode:
        existing = self.nodes.get(node.id)
        if existing is None:
            self.nodes[node.id] = node
            return node
        return existing

    def add_edge(self, source: str, relation: str, target: str) -> bool:
        key = (source, relation, target)
        if key in self._seen or source not in self.nodes or target not in self.nodes:
            return False
        self._seen.add(key)
        self.edges.append(ProvenanceEdge(source=source, relation=relation, target=target))
        return True

    def neighbors(self, node_id: str, relation: str | None = None) -> list[GraphNode]:
        found: list[GraphNode] = []
        for edge in self.edges:
            if edge.source != node_id:
                continue
            if relation is not None and edge.relation != relation:
                continue
            node = self.nodes.get(edge.target)
            if node is not None:
                found.append(node)
        return found

    def clear(self) -> None:
        self.nodes.clear()
        self.edges.clear()
        self.citations.clear()
        self.audit.clear()
        self._seen.clear()

    def claims(self) -> list[GraphNode]:
        return [node for node in self.nodes.values() if node.kind == CLAIM]

    def to_payload(self) -> dict[str, object]:
        return {
            "nodes": [
                {
                    "id": node.id,
                    "kind": node.kind,
                    "label": node.label,
                    "document_id": node.provenance.document_id,
                    "version": node.provenance.version,
                    "filename": node.provenance.filename,
                    "content_type": node.provenance.content_type,
                    "checksum": node.provenance.checksum,
                    "chunk_id": node.provenance.chunk_id,
                    "heading": node.provenance.heading,
                    "page": node.provenance.page,
                    "start_char": node.provenance.start_char,
                    "end_char": node.provenance.end_char,
                    "recorded_at": node.provenance.recorded_at,
                }
                for node in self.nodes.values()
            ],
            "edges": [
                {"source": edge.source, "relation": edge.relation, "target": edge.target}
                for edge in self.edges
            ],
            "citations": [
                {
                    "claim_id": item.claim_id,
                    "quote": item.quote,
                    "document_id": item.document_id,
                    "filename": item.filename,
                    "version": item.version,
                    "chunk_id": item.chunk_id,
                    "start_char": item.start_char,
                    "end_char": item.end_char,
                    "recorded_at": item.recorded_at,
                    "marker": item.marker,
                }
                for item in self.citations
            ],
            "audit": [
                {
                    "id": event.id,
                    "decision": event.decision,
                    "subject_id": event.subject_id,
                    "evidence_ids": list(event.evidence_ids),
                    "detail": event.detail,
                    "recorded_at": event.recorded_at,
                }
                for event in self.audit
            ],
        }


class GraphBoundAdapter:
    """Record one action's outcome on an evidence graph. The inner adapter does the work."""

    def __init__(self, inner, graph: EvidenceGraph, engine: "ProvenanceEngine") -> None:
        self.action = inner.action
        self._inner = inner
        self._graph = graph
        self._engine = engine

    def run(self, request, state=None):
        try:
            outcome = self._inner.run(request, state)
        except TypeError:
            outcome = self._inner.run(request)
        pool = list(outcome.evidence)
        if state is not None:
            seen = {item.evidence_id for item in pool}
            pool.extend(item for item in state.evidence if item.evidence_id not in seen)
        self._engine.record_outcome(self._graph, outcome, evidence=pool)
        return outcome


class ProvenanceEngine:
    """Connect claims and evidence to the document versions they came from."""

    def __init__(self, metadata: MetadataStore | None = None) -> None:
        self._metadata = metadata

    def record_outcome(self, graph: EvidenceGraph, outcome, *, evidence=()) -> None:
        """Attach passages and claims from one worker action. Does not search or judge."""
        by_id = {item.evidence_id: item for item in evidence}
        by_id.update({item.evidence_id: item for item in getattr(outcome, "evidence", ())})
        for item in by_id.values():
            origin = self._origin(
                item.document_id,
                item.source,
                "",
                chunk_id=item.evidence_id,
            )
            passage = GraphNode(
                id=f"passage:{item.evidence_id}",
                kind=PASSAGE,
                label=item.source,
                provenance=origin,
            )
            graph.add_node(passage)
            self._attach_document(graph, origin)
        for verdict in getattr(outcome, "verdicts", ()):
            claim_id = f"claim:{_claim_key(verdict.claim)}"
            graph.add_node(GraphNode(id=claim_id, kind=CLAIM, label=verdict.claim))
            decision = verdict.verdict or "recorded"
            self.record_decision(
                graph,
                decision,
                claim_id,
                tuple(verdict.evidence_ids),
                detail=verdict.claim,
            )
            relation = CONTRADICTS if verdict.verdict == "contradicted" else SUPPORTS
            if verdict.verdict not in {"supported", "contradicted"}:
                continue
            for evidence_id in verdict.evidence_ids:
                item = by_id.get(evidence_id)
                if item is None:
                    continue
                graph.add_edge(f"passage:{evidence_id}", relation, claim_id)
                if item.document_id:
                    graph.add_edge(claim_id, CITED_IN, f"document:{item.document_id}")

    def from_research(
        self,
        records: tuple[EvidenceRecord, ...] | list[EvidenceRecord],
        conflicts: tuple[Contradiction, ...] | list[Contradiction] = (),
    ) -> EvidenceGraph:
        graph = EvidenceGraph()
        for record in records:
            self._add_record(graph, record, SUPPORTS)
        for conflict in conflicts:
            left = f"claim:{conflict.left_claim_id}"
            right = f"claim:{conflict.right_claim_id}"
            if conflict.left_claim_id and conflict.right_claim_id:
                graph.add_edge(left, CONTRADICTS, right)
                graph.add_edge(right, CONTRADICTS, left)
                recorded_at = records[-1].created_at if records else ""
                self.record_decision(
                    graph,
                    "contradicted",
                    left,
                    (right,),
                    detail=conflict.summary,
                    recorded_at=recorded_at,
                )
                self.record_decision(
                    graph,
                    "contradicted",
                    right,
                    (left,),
                    detail=conflict.summary,
                    recorded_at=recorded_at,
                )
        return graph

    def from_verdict(self, verdict: JudgeVerdict) -> EvidenceGraph:
        graph = EvidenceGraph()
        claim_id = "claim:judge"
        claim_node = GraphNode(id=claim_id, kind=CLAIM, label=verdict.claim)
        graph.add_node(claim_node)
        by_id = {item.evidence_id: item for item in verdict.evidence}
        for evidence_id in verdict.support_ids:
            item = by_id.get(evidence_id)
            if item is not None:
                self._add_item(graph, claim_id, item, SUPPORTS)
        for evidence_id in verdict.oppose_ids:
            item = by_id.get(evidence_id)
            if item is not None:
                self._add_item(graph, claim_id, item, CONTRADICTS)
        return graph

    def _add_record(self, graph: EvidenceGraph, record: EvidenceRecord, relation: str) -> None:
        source = record.source
        origin = self._origin(
            source.document_id,
            source.filename,
            source.content_type,
            chunk_id=source.chunk_id,
            heading=source.heading,
            page=source.page,
            start_char=source.start_char,
            end_char=source.end_char,
        )
        claim = GraphNode(
            id=f"claim:{record.claim_id}",
            kind=CLAIM,
            label=record.claim,
            provenance=origin,
        )
        passage = GraphNode(
            id=f"passage:{source.chunk_id}",
            kind=PASSAGE,
            label=source.filename,
            provenance=origin,
        )
        graph.add_node(claim)
        graph.add_node(passage)
        self._attach_document(graph, origin)
        graph.add_edge(f"passage:{source.chunk_id}", relation, claim.id)
        graph.add_edge(claim.id, CITED_IN, f"document:{source.document_id}")
        self._cite(graph, claim.id, record.claim, origin, record.created_at)
        self.record_decision(
            graph,
            record.verdict or "supported",
            claim.id,
            (record.evidence_id,),
            detail=f"Cited {origin.filename} v{origin.version} chars {origin.start_char}–{origin.end_char}",
            recorded_at=record.created_at,
        )

    def _add_item(self, graph: EvidenceGraph, claim_id: str, item: EvidenceItem, relation: str) -> None:
        origin = self._origin(
            item.document_id,
            item.filename,
            "",
            chunk_id=item.evidence_id,
            heading=item.heading,
            page=item.page,
            start_char=item.start_char,
            end_char=item.end_char,
        )
        passage = GraphNode(
            id=f"passage:{item.evidence_id}",
            kind=PASSAGE,
            label=item.filename,
            provenance=origin,
        )
        graph.add_node(passage)
        self._attach_document(graph, origin)
        graph.add_edge(passage.id, relation, claim_id)
        graph.add_edge(claim_id, CITED_IN, f"document:{item.document_id}")
        self._cite(graph, claim_id, item.text if hasattr(item, "text") else item.filename, origin)
        self.record_decision(
            graph,
            "supported" if relation == SUPPORTS else "contradicted",
            claim_id,
            (item.evidence_id,),
            detail=f"Cited {origin.filename} v{origin.version} chars {origin.start_char}–{origin.end_char}",
            recorded_at=origin.recorded_at,
        )

    def _attach_document(self, graph: EvidenceGraph, origin: Provenance) -> None:
        if not origin.document_id:
            return
        document = GraphNode(
            id=f"document:{origin.document_id}",
            kind=DOCUMENT,
            label=origin.filename,
            provenance=origin,
        )
        source = GraphNode(
            id=f"source:{origin.chunk_id or origin.document_id}",
            kind=SOURCE,
            label=origin.filename,
            provenance=origin,
        )
        graph.add_node(document)
        graph.add_node(source)
        if origin.chunk_id:
            graph.add_edge(f"passage:{origin.chunk_id}", EXTRACTED_FROM, document.id)
            graph.add_edge(f"passage:{origin.chunk_id}", HAS_SOURCE, source.id)
        versions = self._document_versions(origin)
        for item in versions:
            version_id = f"version:{origin.document_id}:v{item.version}"
            graph.add_node(GraphNode(id=version_id, kind=VERSION, label=f"{item.filename} v{item.version}", provenance=item))
            graph.add_edge(document.id, HAS_VERSION, version_id)
        ordered = sorted(versions, key=lambda item: item.version)
        for older, newer in zip(ordered, ordered[1:]):
            newer_id = f"version:{origin.document_id}:v{newer.version}"
            older_id = f"version:{origin.document_id}:v{older.version}"
            if graph.add_edge(newer_id, SUPERSEDES, older_id):
                self.record_decision(
                    graph,
                    "superseded",
                    newer_id,
                    (older_id,),
                    detail=f"{newer.filename} v{newer.version} supersedes v{older.version}",
                    recorded_at=newer.recorded_at,
                )

    def record_decision(
        self,
        graph: EvidenceGraph,
        decision: str,
        subject_id: str,
        evidence_ids: tuple[str, ...] = (),
        *,
        detail: str = "",
        recorded_at: str = "",
    ) -> AuditEvent:
        """Append one decision to the audit trail."""
        event = AuditEvent(
            id=f"audit-{len(graph.audit) + 1}",
            decision=decision,
            subject_id=subject_id,
            evidence_ids=tuple(evidence_ids),
            detail=detail,
            recorded_at=recorded_at or datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        )
        graph.audit.append(event)
        return event

    def _cite(
        self,
        graph: EvidenceGraph,
        claim_id: str,
        quote: str,
        origin: Provenance,
        recorded_at: str = "",
    ) -> None:
        if any(item.claim_id == claim_id for item in graph.citations):
            return
        graph.citations.append(
            ClaimCitation(
                claim_id=claim_id,
                quote=quote,
                document_id=origin.document_id,
                filename=origin.filename,
                version=origin.version,
                chunk_id=origin.chunk_id,
                start_char=origin.start_char,
                end_char=origin.end_char,
                recorded_at=recorded_at or origin.recorded_at,
            )
        )

    def _document_versions(self, origin: Provenance) -> list[Provenance]:
        listed = []
        list_versions = getattr(self._metadata, "list_versions", None)
        if callable(list_versions):
            try:
                listed = list(list_versions(origin.document_id))
            except Exception:
                listed = []
        if listed:
            return [
                Provenance(
                    document_id=origin.document_id,
                    version=item.version,
                    filename=item.filename or origin.filename,
                    content_type=item.content_type or origin.content_type,
                    checksum=item.checksum,
                    recorded_at=item.created_at,
                )
                for item in listed
            ]
        return [
            Provenance(
                document_id=origin.document_id,
                version=origin.version,
                filename=origin.filename,
                content_type=origin.content_type,
                checksum=origin.checksum,
                recorded_at=origin.recorded_at,
            )
        ]

    def attach_entities(self, graph: EvidenceGraph, names: list[str] | tuple[str, ...]) -> None:
        """Connect claims that name an entity. Does not search or judge."""
        for name in names:
            cleaned = name.strip()
            if not cleaned:
                continue
            slug = re.sub(r"[^a-z0-9]+", "-", cleaned.lower()).strip("-")
            entity_id = f"entity:{slug}"
            graph.add_node(GraphNode(id=entity_id, kind=ENTITY, label=cleaned))
            needle = cleaned.lower()
            for claim in graph.claims():
                if needle in claim.label.lower():
                    graph.add_edge(claim.id, MENTIONS, entity_id)

    def _origin(
        self,
        document_id: str,
        filename: str,
        content_type: str,
        *,
        chunk_id: str = "",
        heading: str | None = None,
        page: int | None = None,
        start_char: int = 0,
        end_char: int = 0,
    ) -> Provenance:
        record = None if self._metadata is None else self._metadata.get(document_id)
        version = 0 if record is None else record.indexed_version
        checksum = "" if record is None else record.checksum
        stored_type = content_type
        stored_name = filename
        if record is not None:
            stored_name = record.filename or filename
            stored_type = record.content_type or content_type
            recorded_at = getattr(record, "updated_at", "") or getattr(record, "created_at", "")
        else:
            recorded_at = ""
        return Provenance(
            document_id=document_id,
            version=version,
            filename=stored_name,
            content_type=stored_type,
            checksum=checksum,
            chunk_id=chunk_id,
            heading=heading,
            page=page,
            start_char=start_char,
            end_char=end_char,
            recorded_at=recorded_at,
        )


def _claim_key(claim: str) -> str:
    return hashlib.sha256(claim.strip().encode()).hexdigest()[:12]
