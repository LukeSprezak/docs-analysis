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

    async def rerank(self, query: str, documents: list[Document], top_k: int = 4) -> list[Document]:
        self.rerank_top_k = top_k
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
