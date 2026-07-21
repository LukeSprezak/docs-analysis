from app.knowledge_management.application.use_cases.ask_question import AskQuestionUseCase
from app.knowledge_management.domain.models import Document
from app.knowledge_management.domain.repositories import RerankerService
from tests.fakes import StubRAGService, StubVectorStoreRepo


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


class FakeRAGService(StubRAGService):
    def __init__(self) -> None:
        self.received_documents: list[Document] | None = None

    async def answer_question(
        self,
        question: str,
        context: list[Document],
        history: list[dict[str, str]] | None = None,
    ) -> str:
        self.received_documents = context
        return "Answer based on the context"


def _build_documents(count: int) -> list[Document]:
    return [
        Document(id=f"doc::{index}", content=f"excerpt {index}", metadata={})
        for index in range(count)
    ]


async def test_execute_fetches_candidates_reranks_and_answers():
    documents = _build_documents(5)
    vector_repo = FakeVectorStoreRepo(documents)
    reranker = FakeReranker()
    rag_service = FakeRAGService()
    use_case = AskQuestionUseCase(
        vector_repo,
        rag_service,
        reranker,
        candidate_count=20,
        top_k=2,
    )

    answer = await use_case.execute("How does quicksort work?", owner_id="owner1")

    # search retrieves a large set of candidates; rerank narrows it down to top_k.
    assert vector_repo.search_top_k == 20
    assert vector_repo.search_owner_id == "owner1"
    assert reranker.rerank_top_k == 2
    assert len(answer.sources) == 2
    # The RAG receives exactly the reranked fragments.
    assert rag_service.received_documents == documents[:2]
    assert answer.text == "Answer based on the context"
