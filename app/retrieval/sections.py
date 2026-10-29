"""Pull the rest of a section once one of its chunks matches."""

from app.ingestion.vector_store import VectorHit
from app.retrieval.hybrid import RankedChunk, _coverage
from app.retrieval.text import content_tokens
from app.security.content import neutralize_untrusted_text


def heading_overlap(heading: str, query_tokens: list[str]) -> int:
    return len(set(content_tokens(heading or "")) & set(query_tokens))


def heading_seeds(catalog: list[VectorHit], query_tokens: list[str], *, limit: int = 6) -> list[VectorHit]:
    """Chunks whose stored heading shares topic words, anywhere in the file."""
    scored: list[tuple[int, int, VectorHit]] = []
    seen_headings: set[str] = set()
    for hit in catalog:
        heading = str(hit.metadata.get("heading") or "").strip()
        if not heading:
            continue
        overlap = heading_overlap(heading, query_tokens)
        if overlap < 1:
            continue
        key = heading.casefold()
        if key in seen_headings:
            continue
        seen_headings.add(key)
        scored.append((overlap, -int(hit.metadata.get("chunk_index", 0)), hit))
    scored.sort(reverse=True)
    return [hit for _, __, hit in scored[:limit]]


def promote_heading_matches(
    ranked: list[RankedChunk],
    catalog: list[VectorHit],
    query_tokens: list[str],
    *,
    limit: int = 8,
) -> list[RankedChunk]:
    """Move matching sections to the front even when they sit late in a long file."""
    seeds = heading_seeds(catalog, query_tokens)
    if not seeds:
        return expand_section(ranked, catalog, query_tokens, limit=limit) if ranked else ranked
    promoted: list[RankedChunk] = []
    seen: set[str] = set()
    for hit in seeds:
        seed = _rank_hit(hit, query_tokens, 2.0)
        for chunk in expand_section([seed], catalog, query_tokens, limit=limit):
            if chunk.chunk_id in seen:
                continue
            seen.add(chunk.chunk_id)
            promoted.append(chunk)
        if len(promoted) >= limit:
            break
    rest = [chunk for chunk in ranked if chunk.chunk_id not in seen]
    return promoted[:limit] + rest


def expand_section(
    ranked: list[RankedChunk],
    catalog: list[VectorHit],
    query_tokens: list[str],
    *,
    limit: int = 8,
) -> list[RankedChunk]:
    """Add sibling chunks that share the seed heading, page, or nearby index."""
    if not ranked or not catalog:
        return ranked
    seed = max(
        ranked[:8],
        key=lambda chunk: (
            heading_overlap(str(chunk.metadata.get("heading") or ""), query_tokens),
            chunk.coverage,
            -ranked.index(chunk),
        ),
    )
    siblings = _siblings(seed, catalog)
    if not siblings:
        return ranked
    known = {chunk.chunk_id: chunk for chunk in ranked}
    extra: list[RankedChunk] = []
    for hit in siblings:
        current = known.get(hit.chunk_id)
        extra.append(current if current is not None else _rank_hit(hit, query_tokens, seed.score))
    extra.sort(key=lambda chunk: int(chunk.metadata.get("chunk_index", 0)))
    leading = [chunk for chunk in extra if chunk.chunk_id == seed.chunk_id]
    rest = [chunk for chunk in extra if chunk.chunk_id != seed.chunk_id]
    merged = (leading + rest)[:limit]
    leftover = [chunk for chunk in ranked if chunk.chunk_id not in {item.chunk_id for item in merged}]
    return merged + leftover


def _siblings(seed: RankedChunk, catalog: list[VectorHit]) -> list[VectorHit]:
    document_id = str(seed.metadata.get("document_id") or "")
    heading = str(seed.metadata.get("heading") or "").strip()
    index = int(seed.metadata.get("chunk_index", 0))
    page = seed.metadata.get("page")
    found: list[VectorHit] = []
    for hit in catalog:
        if str(hit.metadata.get("document_id") or "") != document_id:
            continue
        other_heading = str(hit.metadata.get("heading") or "").strip()
        other_index = int(hit.metadata.get("chunk_index", -99))
        same_heading = bool(heading) and other_heading == heading
        nearby = abs(other_index - index) <= 2
        same_page = page not in {None, "", -1} and hit.metadata.get("page") == page
        if same_heading or (not heading and (nearby or same_page)):
            found.append(hit)
    return found or [hit for hit in catalog if hit.chunk_id == seed.chunk_id]


def _rank_hit(hit: VectorHit, query_tokens: list[str], score: float) -> RankedChunk:
    safe = neutralize_untrusted_text(hit.text)
    coverage, matched = _coverage(query_tokens, content_tokens(safe.text))
    return RankedChunk(
        chunk_id=hit.chunk_id,
        text=hit.text,
        metadata=hit.metadata,
        vector_score=0.0,
        lexical_score=0.0,
        score=score,
        rerank_score=coverage,
        coverage=coverage,
        matched_terms=matched,
        safe_text=safe.text,
        instruction_detected=safe.instruction_detected,
    )
