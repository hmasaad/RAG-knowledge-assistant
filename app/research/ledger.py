"""Append-only record of verification decisions and source spans."""

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class SourceSpan:
    document_id: str
    evidence_id: str
    section_id: str
    heading: str
    page: int | None
    start_char: int
    end_char: int


@dataclass(frozen=True)
class LedgerEntry:
    """One claim scored against one source span."""

    claim: str
    evidence_id: str
    verdict: str
    relevance: float | None
    contextual_meaning: float | None
    support: float | None
    contradicted: bool
    inherited: str
    span: SourceSpan

    def to_payload(self) -> dict:
        payload = asdict(self)
        return payload


@dataclass
class EvidenceLedger:
    """Every verification decision for a research session."""

    entries: list[LedgerEntry] = field(default_factory=list)

    def record(self, entry: LedgerEntry) -> LedgerEntry:
        self.entries.append(entry)
        return entry

    def extend(self, entries: list[LedgerEntry] | tuple[LedgerEntry, ...]) -> None:
        self.entries.extend(entries)

    def supported_ids(self, claim: str = "") -> tuple[str, ...]:
        wanted = claim.strip().casefold()
        ids: list[str] = []
        for entry in self.entries:
            if entry.verdict != "supported":
                continue
            if wanted and entry.claim.strip().casefold() != wanted:
                continue
            if entry.evidence_id not in ids:
                ids.append(entry.evidence_id)
        return tuple(ids)

    def to_payload(self) -> list[dict]:
        return [entry.to_payload() for entry in self.entries]

    @classmethod
    def from_payload(cls, rows: list[dict] | None) -> "EvidenceLedger":
        ledger = cls()
        for row in rows or []:
            span = row.get("span") or {}
            ledger.record(
                LedgerEntry(
                    claim=str(row.get("claim", "")),
                    evidence_id=str(row.get("evidence_id", "")),
                    verdict=str(row.get("verdict", "")),
                    relevance=row.get("relevance"),
                    contextual_meaning=row.get("contextual_meaning"),
                    support=row.get("support"),
                    contradicted=bool(row.get("contradicted", False)),
                    inherited=str(row.get("inherited", "")),
                    span=SourceSpan(
                        document_id=str(span.get("document_id", "")),
                        evidence_id=str(span.get("evidence_id", "") or row.get("evidence_id", "")),
                        section_id=str(span.get("section_id", "")),
                        heading=str(span.get("heading", "")),
                        page=span.get("page"),
                        start_char=int(span.get("start_char", 0) or 0),
                        end_char=int(span.get("end_char", 0) or 0),
                    ),
                )
            )
        return ledger
