"""A true integration test of the RAG path — unlike `test_integration.py`
(which mocks the use cases and only exercises FastAPI routing), here REAL components run
end to end: `upload_document` → a real `FaissVectorStoreRepo`
(chunking + embedding) → `RetrievalPipeline` → a real reranker + a real `LangChainRAGService`.

No network: deterministic embeddings (`DeterministicFakeEmbedding`) and an LLM mocked at the
library level (`GenericFakeChatModel`) — no use case or repo is mocked.
E2E against a real Postgres (testcontainers / docker pgvector) is left as a separate task
(it needs a live database) — FAISS covers the full retrieval logic offline.
"""

from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from app.documents.service import upload_document
from app.retrieval.chunker import TextChunker
from app.retrieval.indexing import IndexDocumentUseCase
from app.retrieval.pipeline import RetrievalPipeline
from app.retrieval.rag_service import LangChainRAGService
from app.retrieval.reranker import NoOpReranker
from app.retrieval.vector_store.faiss import FaissVectorStoreRepo
from app.shared.kernel.document import Document
from tests.fakes import StubDocumentRepo


class InMemoryDocRepo(StubDocumentRepo):
    def __init__(self) -> None:
        self.documents: dict[str, tuple[Document, str]] = {}

    async def save(self, document: Document, owner_id: str) -> None:
        self.documents[document.id] = (document, owner_id)


async def test_upload_then_ask_flows_through_real_components():
    embeddings = DeterministicFakeEmbedding(size=32)
    vector_repo = FaissVectorStoreRepo(
        embeddings=embeddings, chunker=TextChunker(chunk_size=200, chunk_overlap=20)
    )
    doc_repo = InMemoryDocRepo()

    await upload_document(
        doc_repo,
        IndexDocumentUseCase(vector_repo),
        doc_id="algo.txt",
        content="Quicksort has O(n log n) complexity in the average case. " * 10,
        metadata={"filename": "algo.txt"},
        owner_id="o1",
    )

    # The upload stored the (namespaced) document in the document repo AND its fragments in the vectors.
    assert "o1::algo.txt" in doc_repo.documents

    fake_llm = GenericFakeChatModel(messages=iter([AIMessage(content="Quicksort: O(n log n).")]))
    pipeline = RetrievalPipeline(vector_repo, NoOpReranker(), candidate_count=20, top_k=4)
    question = "What is the complexity of quicksort?"

    sources = await pipeline.retrieve(question, owner_id="o1")
    answer = await LangChainRAGService(llm=fake_llm).answer_question(question, sources)

    assert answer == "Quicksort: O(n log n)."
    # Retrieval really did return fragments of the uploaded document.
    assert len(sources) > 0
    assert any("Quicksort" in source.content for source in sources)


async def test_retrieval_is_isolated_per_owner_end_to_end():
    embeddings = DeterministicFakeEmbedding(size=32)
    vector_repo = FaissVectorStoreRepo(
        embeddings=embeddings, chunker=TextChunker(chunk_size=10_000)
    )
    doc_repo = InMemoryDocRepo()
    await upload_document(
        doc_repo,
        IndexDocumentUseCase(vector_repo),
        doc_id="secret.txt",
        content="Confidential ACME company data.",
        metadata={"filename": "secret.txt"},
        owner_id="owner",
    )

    pipeline = RetrievalPipeline(vector_repo, NoOpReranker(), candidate_count=20, top_k=4)

    # Another user cannot search someone else's document (owner_id isolation in retrieval).
    sources = await pipeline.retrieve("Confidential ACME data?", owner_id="intruder")

    assert sources == []
