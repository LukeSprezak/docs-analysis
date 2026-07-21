"""Typed test doubles for the domain contracts.

The contracts in `domain/repositories.py` are ABCs, so a stand-in has to implement every
abstract method to be a valid argument for a use case. These bases implement all of them
as `NotImplementedError`; a test subclasses one and overrides only what it exercises.
"""

from collections.abc import AsyncIterator

from app.knowledge_management.domain.models import (
    Conversation,
    Document,
    Summary,
)
from app.knowledge_management.domain.repositories import (
    AnswerJudge,
    ConversationRepo,
    DocumentRepo,
    RAGService,
    RerankerService,
    SummaryRepo,
    VectorStoreRepo,
)


class StubDocumentRepo(DocumentRepo):
    async def save(self, document: Document, owner_id: str) -> None:
        raise NotImplementedError

    async def get_by_id(self, doc_id: str, owner_id: str) -> Document | None:
        raise NotImplementedError

    async def list_all(self, owner_id: str, limit: int = 50, offset: int = 0) -> list[Document]:
        raise NotImplementedError

    async def delete(self, doc_id: str, owner_id: str) -> None:
        raise NotImplementedError


class StubVectorStoreRepo(VectorStoreRepo):
    async def add_documents(self, documents: list[Document], owner_id: str) -> None:
        raise NotImplementedError

    async def search(self, query: str, owner_id: str, top_k: int = 4) -> list[Document]:
        raise NotImplementedError

    async def delete_by_document_id(self, doc_id: str, owner_id: str) -> None:
        raise NotImplementedError


class StubSummaryRepo(SummaryRepo):
    async def save(self, summary: Summary, owner_id: str) -> str:
        raise NotImplementedError

    async def get_by_id(self, summary_id: str, owner_id: str) -> Summary | None:
        raise NotImplementedError

    async def list_all(self, owner_id: str, limit: int = 50, offset: int = 0) -> list[Summary]:
        raise NotImplementedError

    async def delete(self, summary_id: str, owner_id: str) -> None:
        raise NotImplementedError


class StubConversationRepo(ConversationRepo):
    async def save(self, conversation: Conversation, owner_id: str) -> None:
        raise NotImplementedError

    async def get_by_id(self, conversation_id: str, owner_id: str) -> Conversation | None:
        raise NotImplementedError

    async def list_all(self, owner_id: str, limit: int = 50, offset: int = 0) -> list[Conversation]:
        raise NotImplementedError

    async def delete(self, conversation_id: str, owner_id: str) -> None:
        raise NotImplementedError


class StubRAGService(RAGService):
    async def answer_question(
        self,
        question: str,
        context: list[Document],
        history: list[dict[str, str]] | None = None,
    ) -> str:
        raise NotImplementedError

    async def condense_question(self, question: str, history: list[dict[str, str]]) -> str:
        return question

    def astream_answer(
        self,
        question: str,
        context: list[Document],
        history: list[dict[str, str]] | None = None,
    ) -> AsyncIterator[str]:
        raise NotImplementedError


class StubAnswerJudge(AnswerJudge):
    async def score_faithfulness(self, answer: str, context: list[Document]) -> float:
        raise NotImplementedError

    async def score_answer_relevance(self, question: str, answer: str) -> float:
        raise NotImplementedError


class PassthroughReranker(RerankerService):
    """Keeps the order coming out of the vector search and only trims to top_k."""

    async def rerank(self, query: str, documents: list[Document], top_k: int = 4) -> list[Document]:
        return documents[:top_k]
