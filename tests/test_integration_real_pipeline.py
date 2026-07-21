"""Prawdziwy test integracyjny ścieżki RAG — w odróżnieniu od `test_integration.py`
(który mockuje use case'y i testuje tylko routing FastAPI), tutaj przez całą ścieżkę
przechodzą REALNE komponenty: `UploadDocumentUseCase` → realny `FaissVectorStoreRepo`
(chunking + embedding) → `AskQuestionUseCase` → realny reranker + realny `LangChainRAGService`.

Bez sieci: embeddingi deterministyczne (`DeterministicFakeEmbedding`), LLM zamockowany na
poziomie biblioteki (`GenericFakeChatModel`) — nie mockujemy żadnego use case'a ani repo.
E2E na realnym Postgresie (testcontainers / docker pgvector) zostaje jako osobne zadanie
(wymaga żywej bazy) — FAISS pokrywa pełną logikę retrievalu offline.
"""

from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from app.knowledge_management.application.use_cases.ask_question import AskQuestionUseCase
from app.knowledge_management.application.use_cases.upload_document import UploadDocumentUseCase
from app.knowledge_management.infrastructure.llm.langchain_rag_service import LangChainRAGService
from app.knowledge_management.infrastructure.llm.reranker import NoOpReranker
from app.knowledge_management.infrastructure.persistence.faiss_vectorstore_repo import (
    FaissVectorStoreRepo,
)
from app.knowledge_management.infrastructure.text.text_chunker import TextChunker


class InMemoryDocRepo:
    def __init__(self):
        self.documents = {}

    async def save(self, document, owner_id):
        self.documents[document.id] = (document, owner_id)


async def test_upload_then_ask_flows_through_real_components():
    embeddings = DeterministicFakeEmbedding(size=32)
    vector_repo = FaissVectorStoreRepo(
        embeddings=embeddings, chunker=TextChunker(chunk_size=200, chunk_overlap=20)
    )
    doc_repo = InMemoryDocRepo()

    await UploadDocumentUseCase(doc_repo, vector_repo).execute(
        doc_id="algo.txt",
        content="Quicksort ma złożoność O(n log n) w średnim przypadku. " * 10,
        metadata={"filename": "algo.txt"},
        owner_id="o1",
    )

    # Upload zapisał dokument (namespaced) w repo dokumentów ORAZ fragmenty w wektorach.
    assert "o1::algo.txt" in doc_repo.documents

    fake_llm = GenericFakeChatModel(messages=iter([AIMessage(content="Quicksort: O(n log n).")]))
    ask = AskQuestionUseCase(
        vector_repo,
        LangChainRAGService(llm=fake_llm),
        NoOpReranker(),
        candidate_count=20,
        top_k=4,
    )

    answer = await ask.execute("Jaka jest złożoność quicksort?", owner_id="o1")

    assert answer.text == "Quicksort: O(n log n)."
    # Retrieval realnie zwrócił fragmenty wgranego dokumentu.
    assert len(answer.sources) > 0
    assert any("Quicksort" in source.content for source in answer.sources)


async def test_retrieval_is_isolated_per_owner_end_to_end():
    embeddings = DeterministicFakeEmbedding(size=32)
    vector_repo = FaissVectorStoreRepo(
        embeddings=embeddings, chunker=TextChunker(chunk_size=10_000)
    )
    doc_repo = InMemoryDocRepo()
    upload = UploadDocumentUseCase(doc_repo, vector_repo)

    await upload.execute(
        doc_id="tajne.txt",
        content="Poufne dane firmy ACME.",
        metadata={"filename": "tajne.txt"},
        owner_id="wlasciciel",
    )

    fake_llm = GenericFakeChatModel(messages=iter([AIMessage(content="brak kontekstu")]))
    ask = AskQuestionUseCase(
        vector_repo, LangChainRAGService(llm=fake_llm), NoOpReranker(), candidate_count=20, top_k=4
    )

    # Inny użytkownik nie może wyszukać cudzego dokumentu (izolacja owner_id w retrievalu).
    answer = await ask.execute("Poufne dane ACME?", owner_id="intruz")

    assert answer.sources == []
