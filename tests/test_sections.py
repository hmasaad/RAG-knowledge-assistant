from app.ingestion.vector_store import VectorHit
from app.retrieval.hybrid import RankedChunk
from app.retrieval.sections import expand_section, heading_seeds, promote_heading_matches


def test_expand_section_adds_sibling_chunks_with_the_same_heading() -> None:
    seed = _chunk("c1", "Sign up and sign in.", heading="Authentication & Profile", index=0)
    sibling = _hit("c2", "Edit name, photo, and saved salons.", heading="Authentication & Profile", index=1)
    other = _hit("c3", "Customers pick a salon.", heading="Booking", index=2)

    expanded = expand_section([seed], [ _hit_from(seed), sibling, other], ["authentication", "profile"])

    assert [item.chunk_id for item in expanded[:2]] == ["c1", "c2"]
    assert "c3" not in [item.chunk_id for item in expanded[:2]]


def test_heading_match_is_found_late_in_a_long_file() -> None:
    catalog = [_hit(f"p{index}", f"Page banner {index} salon booking app.", heading="Cover", index=index) for index in range(40)]
    target = _hit(
        "auth",
        "Sign up and sign in with email or phone.",
        heading="Authentication & Profile",
        index=40,
    )
    catalog.append(target)
    noise = [_chunk(f"p{index}", f"Page banner {index} salon booking app.", heading="Cover", index=index) for index in range(5)]

    seeds = heading_seeds(catalog, ["authentication", "profile"])
    promoted = promote_heading_matches(noise, catalog, ["authentication", "profile"])

    assert seeds[0].chunk_id == "auth"
    assert promoted[0].chunk_id == "auth"


def _chunk(chunk_id: str, text: str, *, heading: str, index: int) -> RankedChunk:
    return RankedChunk(
        chunk_id=chunk_id,
        text=text,
        metadata={"document_id": "doc", "heading": heading, "chunk_index": index},
        vector_score=1.0,
        lexical_score=1.0,
        score=1.0,
        rerank_score=1.0,
        coverage=1.0,
        matched_terms=("authentication",),
        safe_text=text,
        instruction_detected=False,
    )


def _hit(chunk_id: str, text: str, *, heading: str, index: int) -> VectorHit:
    return VectorHit(
        chunk_id=chunk_id,
        text=text,
        score=0.5,
        metadata={"document_id": "doc", "heading": heading, "chunk_index": index},
    )


def _hit_from(chunk: RankedChunk) -> VectorHit:
    return VectorHit(chunk_id=chunk.chunk_id, text=chunk.text, score=chunk.score, metadata=chunk.metadata)
