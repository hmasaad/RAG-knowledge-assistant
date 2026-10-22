"""Pairs that a useful embedding model should rank in the right order."""

from dataclasses import dataclass


@dataclass(frozen=True)
class EmbeddingPair:
    id: str
    query: str
    positive: str
    negative: str


EMBEDDING_PAIRS = (
    EmbeddingPair(
        id="retry-file",
        query="which file implements the retry logic",
        positive="The file that implements retry logic is app/http/retry.py.",
        negative="Schema changes ship as SQL files under db/migrations.",
    ),
    EmbeddingPair(
        id="timeouts",
        query="where are connect and read timeouts configured",
        positive="Connect and read timeouts are configured in app/http/client.py.",
        negative="The runner stores applied versions in the schema_migrations table.",
    ),
    EmbeddingPair(
        id="retry-limit",
        query="how many retry attempts are allowed",
        positive="Callers are allowed three retry attempts.",
        negative="Failed SQL stops the runner so later versions do not apply.",
    ),
)
