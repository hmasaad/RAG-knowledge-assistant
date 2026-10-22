"""Compare embedding vectors.

The vector collection uses cosine distance. For normalized vectors that distance
is ``1 - cosine_similarity``, and the retrieval score converts it back.
"""


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Dot product of two equal-length vectors. Normalized vectors lie in [-1, 1]."""
    if not left or len(left) != len(right):
        raise ValueError("Vectors must be the same non-zero length.")
    return sum(a * b for a, b in zip(left, right, strict=True))


def cosine_distance(left: list[float], right: list[float]) -> float:
    """Cosine distance. Identical vectors are 0. Opposite vectors are 2."""
    return 1.0 - cosine_similarity(left, right)


def embedding_matches(embedder: object, identity: tuple[str, int] | None) -> bool:
    """True when this embedder can query vectors stored under ``identity``.

    A missing stored identity is compatible. Dimension is checked when the
    embedder reports one, so a same-named model at a new width is not.
    """
    if identity is None:
        return True
    model, dimension = identity
    if getattr(embedder, "model", None) != model:
        return False
    reported = getattr(embedder, "dimension", None)
    if reported is None:
        return True
    return int(reported) == int(dimension)


def mismatch_message(stored: tuple[str, int], embedder: object, *, action: str) -> str:
    active_model = str(getattr(embedder, "model", "unknown"))
    active_dimension = getattr(embedder, "dimension", None)
    if active_dimension is None:
        active = active_model
    else:
        active = f"{active_model} ({int(active_dimension)} dimensions)"
    model, dimension = stored
    return (
        f"The collection was embedded with {model} ({dimension} dimensions). "
        f"The active model is {active}. Migrate embeddings before {action}."
    )
