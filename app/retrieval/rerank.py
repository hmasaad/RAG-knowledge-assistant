"""Rescore hybrid candidates with a cross-encoder or a reranking API.

Vector search and BM25 each rank passages alone. A cross-encoder scores the
query and a passage together. ``coverage`` leaves the hybrid score in place.
``cross_encoder`` is the offline pairwise model. ``api`` posts the same
candidates to a Cohere-compatible rerank endpoint.
"""

import json
from dataclasses import replace
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.ingestion.errors import KnowledgeError
from app.retrieval.hybrid import RankedChunk
from app.retrieval.text import content_tokens


class Reranker(Protocol):
    model: str

    def score(self, query: str, texts: list[str]) -> list[float]:
        """Return one relevance score per text, in the same order."""


class LocalCrossEncoder:
    """Score a query and a passage from their joint token positions.

    Coverage asks which question terms appear. Proximity asks whether those
    terms sit together, and order asks whether they follow the question.
    A passage that merely mentions the words far apart scores lower.
    """

    def __init__(self, model: str = "local-cross-encoder") -> None:
        self.model = model

    def score(self, query: str, texts: list[str]) -> list[float]:
        query_tokens = content_tokens(query)
        return [_pair_score(query_tokens, content_tokens(text)) for text in texts]


class ApiReranker:
    """Call a reranking API that accepts the Cohere v2 request shape."""

    def __init__(
        self,
        *,
        url: str,
        api_key: str,
        model: str = "rerank-v3.5",
        timeout: float = 15.0,
        transport=None,
    ) -> None:
        self.model = model
        self._url = url.strip()
        self._api_key = api_key
        self._timeout = timeout
        self._transport = transport

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        if not self._url:
            raise KnowledgeError("Set RERANK_API_URL to use the reranking API.", status_code=503)
        if not self._api_key:
            raise KnowledgeError("Set RERANK_API_KEY to use the reranking API.", status_code=503)
        payload = json.dumps(
            {
                "model": self.model,
                "query": query,
                "documents": texts,
                "top_n": len(texts),
            }
        ).encode()
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        try:
            raw = self._post(payload, headers)
            parsed = json.loads(raw)
        except KnowledgeError:
            raise
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            raise KnowledgeError("Reranking failed.", status_code=503) from exc
        results = parsed.get("results") if isinstance(parsed, dict) else None
        if not isinstance(results, list):
            raise KnowledgeError("Reranking failed.", status_code=503)
        scores = [0.0] * len(texts)
        for item in results:
            if not isinstance(item, dict) or not isinstance(item.get("index"), int):
                continue
            index = item["index"]
            if index < 0 or index >= len(texts):
                continue
            value = item.get("relevance_score", item.get("score", 0.0))
            try:
                scores[index] = float(value)
            except (TypeError, ValueError):
                continue
        return scores

    def _post(self, payload: bytes, headers: dict[str, str]) -> bytes:
        if self._transport is not None:
            return self._transport(self._url, payload, headers, self._timeout)
        request = Request(self._url, data=payload, headers=headers, method="POST")
        with urlopen(request, timeout=self._timeout) as response:
            return response.read()


def build_reranker(
    *,
    provider: str,
    model: str = "",
    api_url: str = "",
    api_key: str = "",
    timeout: float = 15.0,
) -> Reranker | None:
    """Return the configured reranker. Coverage keeps the hybrid score."""
    if provider == "coverage":
        return None
    if provider == "cross_encoder":
        return LocalCrossEncoder(model=model or "local-cross-encoder")
    if provider == "api":
        return ApiReranker(
            url=api_url,
            api_key=api_key,
            model=model or "rerank-v3.5",
            timeout=timeout,
        )
    raise ValueError(f"Unknown rerank provider: {provider}.")


def apply_rerank(query: str, ranked: list[RankedChunk], reranker: Reranker) -> list[RankedChunk]:
    """Replace the hybrid score with the reranker score and sort again.

    The passage text sent to the reranker is the neutralized text, so an
    instruction hidden in a document cannot steer the ranking.
    """
    if not ranked:
        return []
    scores = reranker.score(query, [chunk.safe_text for chunk in ranked])
    if len(scores) != len(ranked):
        raise KnowledgeError("Reranking failed.", status_code=503)
    updated = [
        replace(chunk, score=float(score), rerank_score=float(score))
        for chunk, score in zip(ranked, scores, strict=True)
    ]
    updated.sort(key=lambda item: (-item.rerank_score, -item.vector_score, item.chunk_id))
    return updated


def _pair_score(query_tokens: list[str], passage_tokens: list[str]) -> float:
    unique = list(dict.fromkeys(query_tokens))
    if not unique:
        return 0.0
    positions = [passage_tokens.index(term) for term in unique if term in passage_tokens]
    coverage = len(positions) / len(unique)
    if len(positions) < 2:
        return coverage
    ordered = sorted(positions)
    gaps = [ordered[index + 1] - ordered[index] for index in range(len(ordered) - 1)]
    mean_gap = sum(gaps) / len(gaps)
    proximity = 1.0 / (1.0 + max(0.0, mean_gap - 1.0))
    in_order = sum(
        1 for index in range(len(positions) - 1) if positions[index] < positions[index + 1]
    )
    order = in_order / (len(positions) - 1)
    return (0.45 * coverage) + (0.35 * proximity) + (0.20 * order)
