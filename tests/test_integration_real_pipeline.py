"""A true integration test of the RAG path — unlike `test_integration.py`
(which mocks the use cases and only exercises FastAPI routing), here REAL components run
end to end: `upload_document` → a real `FaissVectorStoreRepo`
(chunking + embedding) → `AskQuestionUseCase` → a real reranker + a real `LangChainRAGService`.

No network: deterministic embeddings (`DeterministicFakeEmbedding`) and an LLM mocked at the
library level (`GenericFakeChatModel`) — no use case or repo is mocked.
E2E against a real Postgres (testcontainers / docker pgvector) is left as a separate task
(it needs a live database) — FAISS covers the full retrieval logic offline.
"""

from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from app.documents.service import upload_document
from app.retrieval.application.ask_question import AskQuestionUseCase
from app.retrieval.application.index_document import IndexDocumentUseCase
from app.retrieval.application.retrieval_pipeline import RetrievalPipeline
from app.retrieval.infrastructure.faiss_vectorstore_repo import FaissVectorStoreRepo
from app.retrieval.infrastructure.langchain_rag_service import LangChainRAGService
from app.retrieval.infrastructure.reranker import NoOpReranker
from app.retrieval.infrastructure.text_chunker import TextChunker
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
    ask = AskQuestionUseCase(
        RetrievalPipeline(vector_repo, NoOpReranker(), candidate_count=20, top_k=4),
        LangChainRAGService(llm=fake_llm),
    )

    answer = await ask.execute("What is the complexity of quicksort?", owner_id="o1")

    assert answer.text == "Quicksort: O(n log n)."
    # Retrieval really did return fragments of the uploaded document.
    assert len(answer.sources) > 0
    assert any("Quicksort" in source.content for source in answer.sources)


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

    fake_llm = GenericFakeChatModel(messages=iter([AIMessage(content="no context")]))
    ask = AskQuestionUseCase(
        RetrievalPipeline(vector_repo, NoOpReranker(), candidate_count=20, top_k=4),
        LangChainRAGService(llm=fake_llm),
    )

    # Another user cannot search someone else's document (owner_id isolation in retrieval).
    answer = await ask.execute("Confidential ACME data?", owner_id="intruder")

    assert answer.sources == []
