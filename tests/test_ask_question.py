from app.knowledge_management.application.use_cases.ask_question import AskQuestionUseCase
from app.knowledge_management.domain.models import Document


class FakeVectorStoreRepo:
    def __init__(self, documents):
        self.documents = documents
        self.search_top_k = None
        self.search_owner_id = None

    async def search(self, owner_id, top_k=4):
        self.search_top_k = top_k
        self.search_owner_id = owner_id
        return self.documents

    async def add_documents(self, documents, owner_id):  # pragma: no cover - not needed in the test
        raise NotImplementedError

    async def delete_by_document_id(self, document_id, owner_id):  # pragma: no cover
        raise NotImplementedError


class FakeReranker:
    def __init__(self):
        self.rerank_top_k = None

    async def rerank(self, documents, top_k):
        self.rerank_top_k = top_k
        return documents[:top_k]


class FakeRAGService:
    def __init__(self):
        self.received_documents = None

    async def answer_question(self, question, documents, history=None):
        self.received_documents = documents
        return "Answer based on the context"


def _build_documents(count):
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
