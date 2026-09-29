from app.retrieval.pipeline import RetrievalPipeline
from app.retrieval.ports import RerankerService
from app.shared.kernel.document import Document
from tests.fakes import StubVectorStoreRepo


class FakeVectorStoreRepo(StubVectorStoreRepo):
    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents
        self.search_top_k: int | None = None
        self.search_owner_id: str | None = None

    async def search(self, query: str, owner_id: str, top_k: int = 4) -> list[Document]:
        self.search_top_k = top_k
        self.search_owner_id = owner_id
        return self.documents


class FakeReranker(RerankerService):
    def __init__(self) -> None:
        self.rerank_top_k: int | None = None
        self.received: list[Document] | None = None

    async def rerank(self, query: str, documents: list[Document], top_k: int = 4) -> list[Document]:
        self.rerank_top_k = top_k
        self.received = documents
        return documents[:top_k]


def _build_documents(count: int) -> list[Document]:
    return [
        Document(id=f"doc::{index}", content=f"excerpt {index}", metadata={})
        for index in range(count)
    ]


async def test_retrieve_fetches_candidates_and_reranks_to_top_k():
    documents = _build_documents(5)
    vector_repo = FakeVectorStoreRepo(documents)
    reranker = FakeReranker()
    pipeline = RetrievalPipeline(vector_repo, reranker, candidate_count=20, top_k=2)

    sources = await pipeline.retrieve("How does quicksort work?", owner_id="owner1")

    # search retrieves a large set of candidates; rerank narrows it down to top_k.
    assert vector_repo.search_top_k == 20
    assert vector_repo.search_owner_id == "owner1"
    assert reranker.rerank_top_k == 2
    assert sources == documents[:2]


def _scored(name: str, score: float | None) -> Document:
    metadata = {} if score is None else {"score": score}
    return Document(id=f"doc::{name}", content=name, metadata=metadata)


async def test_retrieve_drops_candidates_below_min_score_and_keeps_the_threshold():
    above, at, below = _scored("above", 0.9), _scored("at", 0.7), _scored("below", 0.69)
    pipeline = RetrievalPipeline(
        FakeVectorStoreRepo([above, at, below]), FakeReranker(), top_k=10, min_score=0.7
    )

    sources = await pipeline.retrieve("question", owner_id="owner1")

    assert sources == [above, at]


async def test_retrieve_keeps_candidates_without_a_score():
    # Keyword-only hits (hybrid search) and graph facts carry no similarity to compare.
    keyword_hit = _scored("keyword", None)
    pipeline = RetrievalPipeline(
        FakeVectorStoreRepo([keyword_hit]), FakeReranker(), top_k=10, min_score=0.7
    )

    sources = await pipeline.retrieve("question", owner_id="owner1")

    assert sources == [keyword_hit]


async def test_retrieve_cuts_before_reranking_so_weak_hits_take_no_top_k_slot():
    weak, strong = _scored("weak", 0.1), _scored("strong", 0.9)
    reranker = FakeReranker()
    pipeline = RetrievalPipeline(
        FakeVectorStoreRepo([weak, strong]), reranker, top_k=1, min_score=0.5
    )

    sources = await pipeline.retrieve("question", owner_id="owner1")

    assert reranker.received == [strong]
    assert sources == [strong]
