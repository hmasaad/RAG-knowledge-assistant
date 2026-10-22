"""Fuse vector and lexical rankings, then rerank by term coverage."""

from dataclasses import dataclass
from typing import Any

from app.ingestion.vector_store import VectorHit
from app.retrieval.lexical import bm25_scores
from app.retrieval.text import content_tokens
from app.security.content import neutralize_untrusted_text
from app.security.poison import is_keyword_stuffed


@dataclass(frozen=True)
class RankedChunk:
    chunk_id: str
    text: str
    metadata: dict[str, Any]
    vector_score: float
    lexical_score: float
    score: float
    rerank_score: float
    coverage: float
    matched_terms: tuple[str, ...]
    safe_text: str
    instruction_detected: bool


def reciprocal_rank_fusion(rankings: list[list[str]], *, k: int) -> dict[str, float]:
    """Combine ranked id lists. A hit near the top of more than one list scores higher."""
    fused: dict[str, float] = {}
    for ranking in rankings:
        for index, chunk_id in enumerate(ranking, start=1):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (k + index)
    return fused


def rank_chunks(
    *,
    query_tokens: list[str],
    vector_hits: list[VectorHit],
    chunks: list[VectorHit],
    candidate_k: int,
    rrf_k: int,
    coverage_weight: float,
) -> list[RankedChunk]:
    """Return hybrid candidates, best rerank score first."""
    catalog: dict[str, VectorHit] = {}
    for hit in chunks:
        catalog[hit.chunk_id] = hit
    for hit in vector_hits:
        catalog.setdefault(hit.chunk_id, hit)
    if not catalog:
        return []

    identifiers = [
        chunk_id for chunk_id, hit in catalog.items() if not is_keyword_stuffed(hit.text)
    ]
    if not identifiers:
        return []
    sanitized = {
        chunk_id: neutralize_untrusted_text(catalog[chunk_id].text) for chunk_id in identifiers
    }
    tokenized = [content_tokens(sanitized[chunk_id].text) for chunk_id in identifiers]
    lexical = bm25_scores(query_tokens, tokenized)
    peak_lexical = max(lexical) if lexical else 0.0
    lexical_score = {
        identifiers[index]: (lexical[index] / peak_lexical if peak_lexical else 0.0)
        for index in range(len(identifiers))
    }
    lexical_order = sorted(
        range(len(identifiers)),
        key=lambda index: lexical[index],
        reverse=True,
    )
    lexical_ids = [
        identifiers[index] for index in lexical_order if lexical[index] > 0
    ][:candidate_k]
    vector_ids = [hit.chunk_id for hit in vector_hits if hit.chunk_id in sanitized]
    vector_score = {hit.chunk_id: hit.score for hit in vector_hits}

    fused = reciprocal_rank_fusion([vector_ids, lexical_ids], k=rrf_k)
    peak_fused = max(fused.values()) if fused else 0.0
    ranked: list[RankedChunk] = []
    for chunk_id, fused_score in fused.items():
        hit = catalog[chunk_id]
        safe = sanitized[chunk_id]
        coverage, matched = _coverage(query_tokens, content_tokens(safe.text))
        normalized = fused_score / peak_fused if peak_fused else 0.0
        score = (1 - coverage_weight) * normalized + coverage_weight * coverage
        ranked.append(
            RankedChunk(
                chunk_id=chunk_id,
                text=hit.text,
                metadata=hit.metadata,
                vector_score=vector_score.get(chunk_id, 0.0),
                lexical_score=lexical_score.get(chunk_id, 0.0),
                score=score,
                rerank_score=coverage,
                coverage=coverage,
                matched_terms=matched,
                safe_text=safe.text,
                instruction_detected=safe.instruction_detected,
            )
        )
    ranked.sort(key=lambda item: (-item.score, -item.vector_score, item.chunk_id))
    return ranked


def merge_query_rankings(
    groups: list[list[RankedChunk]],
    *,
    query_tokens: list[str],
    rrf_k: int,
) -> list[RankedChunk]:
    """Fuse one ranking per query. Coverage is scored against ``query_tokens``.

    A single ranking is returned unchanged, so one search keeps its score.
    """
    if len(groups) <= 1:
        return groups[0] if groups else []
    fused = reciprocal_rank_fusion(
        [[chunk.chunk_id for chunk in group] for group in groups],
        k=rrf_k,
    )
    if not fused:
        return []
    chosen: dict[str, RankedChunk] = {}
    for group in groups:
        for chunk in group:
            current = chosen.get(chunk.chunk_id)
            if current is None or chunk.score > current.score:
                chosen[chunk.chunk_id] = chunk
    peak = max(fused.values())
    merged: list[RankedChunk] = []
    for chunk_id, fused_score in fused.items():
        chunk = chosen[chunk_id]
        coverage, matched = _coverage(query_tokens, content_tokens(chunk.safe_text))
        merged.append(
            RankedChunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                metadata=chunk.metadata,
                vector_score=chunk.vector_score,
                lexical_score=chunk.lexical_score,
                score=fused_score / peak if peak else 0.0,
                rerank_score=chunk.rerank_score,
                coverage=coverage,
                matched_terms=matched,
                safe_text=chunk.safe_text,
                instruction_detected=chunk.instruction_detected,
            )
        )
    merged.sort(key=lambda item: (-item.score, -item.vector_score, item.chunk_id))
    return merged


def _coverage(query_tokens: list[str], passage_tokens: list[str]) -> tuple[float, tuple[str, ...]]:
    unique_query = list(dict.fromkeys(query_tokens))
    if not unique_query:
        return 0.0, ()
    present = set(passage_tokens)
    matched = tuple(term for term in unique_query if term in present)
    return len(matched) / len(unique_query), matched
