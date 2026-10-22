"""Expand a matching child chunk to the parent text stored beside it."""

from app.ingestion.vector_store import VectorHit


def expand_parents(hits: list[VectorHit]) -> list[VectorHit]:
    """Collapse child hits that share a parent, keeping the stronger score.

    Hits without parent metadata are unchanged. The returned id is the parent
    id, and the text is the parent text retrieval should show.
    """
    if not any(hit.metadata.get("parent_id") and hit.metadata.get("parent_text") for hit in hits):
        return hits
    best: dict[str, VectorHit] = {}
    order: list[str] = []
    for hit in hits:
        parent_id = str(hit.metadata.get("parent_id") or "")
        parent_text = str(hit.metadata.get("parent_text") or "")
        if not parent_id or not parent_text:
            key = hit.chunk_id
            text = hit.text
            metadata = hit.metadata
        else:
            key = parent_id
            text = parent_text
            metadata = dict(hit.metadata)
            metadata["chunk_index"] = int(metadata.get("parent_index", metadata.get("chunk_index", 0)))
            if metadata.get("parent_start") is not None:
                metadata["start_char"] = int(metadata["parent_start"])
            if metadata.get("parent_end") is not None:
                metadata["end_char"] = int(metadata["parent_end"])
        current = best.get(key)
        if current is None:
            order.append(key)
            best[key] = VectorHit(chunk_id=key, text=text, score=hit.score, metadata=metadata)
        elif hit.score > current.score:
            best[key] = VectorHit(chunk_id=key, text=text, score=hit.score, metadata=metadata)
    return [best[key] for key in order]
