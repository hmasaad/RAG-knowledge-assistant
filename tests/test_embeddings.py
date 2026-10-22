import math

import pytest

from app.ingestion.embeddings import LocalEmbedder, OpenAIEmbedder, build_embedder
from app.ingestion.errors import IngestionError


def _cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


class _Item:
    def __init__(self, index: int, embedding: list[float]) -> None:
        self.index = index
        self.embedding = embedding


class _Response:
    def __init__(self, data: list[_Item]) -> None:
        self.data = data


class _EmbeddingsApi:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def create(self, model: str, input: list[str]) -> _Response:
        self.calls.append(list(input))
        items = [_Item(index, [float(ord(text[0]))]) for index, text in enumerate(input)]
        items.reverse()
        return _Response(items)


class _Client:
    def __init__(self) -> None:
        self.embeddings = _EmbeddingsApi()


def test_local_embedder_ranks_overlapping_words_higher() -> None:
    embedder = LocalEmbedder(dimension=64)
    query = embedder.embed_query("which file implements the retry logic")
    related, unrelated = embedder.embed_documents(
        [
            "The file that implements retry logic is app/http/retry.py.",
            "Schema changes ship as SQL files under db/migrations.",
        ]
    )

    assert math.isclose(sum(value * value for value in query), 1.0, rel_tol=1e-6)
    assert _cosine(query, related) > _cosine(query, unrelated)


def test_openai_embedder_sorts_batches_by_index() -> None:
    client = _Client()
    embedder = OpenAIEmbedder(api_key="test", client=client, batch_size=2)

    vectors = embedder.embed_documents(["a", "b", "c", "d", "e"])

    assert client.embeddings.calls == [["a", "b"], ["c", "d"], ["e"]]
    assert vectors == [[97.0], [98.0], [99.0], [100.0], [101.0]]


def test_openai_embedder_rejects_empty_text() -> None:
    embedder = OpenAIEmbedder(api_key="test", client=_Client(), batch_size=2)
    with pytest.raises(IngestionError, match="empty"):
        embedder.embed_documents(["ok", "   "])


def test_openai_provider_requires_an_api_key() -> None:
    with pytest.raises(IngestionError, match="OPENAI_API_KEY") as error:
        build_embedder(provider="openai", api_key="", model="text-embedding-3-small", dimension=8)
    assert error.value.status_code == 503
