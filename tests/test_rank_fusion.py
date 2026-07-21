from app.knowledge_management.domain.models import Document
from app.knowledge_management.infrastructure.persistence.rank_fusion import (
    fuse_documents,
    reciprocal_rank_fusion,
)


def test_rrf_promotes_key_present_in_both_rankings():
    # "b" jest 2. w obu rankingach; "a" i "c" tylko 1. w jednym.
    # Suma 1/(60+2)+1/(60+2) dla "b" bije pojedyncze 1/(60+1).
    fused = reciprocal_rank_fusion([["a", "b"], ["c", "b"]])
    assert fused[0] == "b"


def test_rrf_single_ranking_preserves_order():
    assert reciprocal_rank_fusion([["x", "y", "z"]]) == ["x", "y", "z"]


def test_rrf_smoothing_constant_controls_top_weight():
    # Przy małej stałej wygładzającej wysokie pozycje ważą mocniej: "a" (1. w jednym)
    # przebija "b" (2. w obu).
    fused = reciprocal_rank_fusion([["a", "b"], ["c", "b"]], smoothing_constant=0)
    assert fused[0] == "a"


def test_rrf_empty_input_is_empty():
    assert reciprocal_rank_fusion([]) == []


def _chunk(doc_id: str, chunk_index: int, content: str = "...") -> Document:
    return Document(
        id=doc_id,
        content=content,
        metadata={"doc_id": doc_id, "chunk_index": chunk_index},
    )


def _key(document: Document) -> str:
    return f"{document.metadata['doc_id']}::{document.metadata['chunk_index']}"


def test_fuse_documents_deduplicates_same_chunk_across_lists():
    shared = _chunk("doc", 1, "wspólny fragment")
    vector_hits = [shared, _chunk("doc", 2)]
    keyword_hits = [shared, _chunk("doc", 3)]

    fused = fuse_documents([vector_hits, keyword_hits], top_k=10, key_of=_key)

    keys = [_key(document) for document in fused]
    assert keys.count("doc::1") == 1  # wspólny fragment nie dubluje się
    assert fused[0].content == "wspólny fragment"  # i awansuje na czoło
    assert len(fused) == 3


def test_fuse_documents_respects_top_k():
    vector_hits = [_chunk("doc", index) for index in range(5)]
    keyword_hits = [_chunk("doc", index) for index in range(5, 10)]

    fused = fuse_documents([vector_hits, keyword_hits], top_k=3, key_of=_key)

    assert len(fused) == 3