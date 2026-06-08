from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from .models import Conversation, Document, Summary


class DocumentRepo(ABC):
    @abstractmethod
    async def save(self, document: Document, owner_id: str) -> None:
        pass

    @abstractmethod
    async def get_by_id(self, doc_id: str, owner_id: str) -> Document | None:
        pass

    @abstractmethod
    async def list_all(
        self, owner_id: str, limit: int = 50, offset: int = 0
    ) -> list[Document]:
        pass

    @abstractmethod
    async def delete(self, doc_id: str, owner_id: str) -> None:
        pass


class VectorStoreRepo(ABC):
    @abstractmethod
    async def add_documents(self, documents: list[Document], owner_id: str) -> None:
        pass

    @abstractmethod
    async def search(self, query: str, owner_id: str, top_k: int = 4) -> list[Document]:
        pass

    @abstractmethod
    async def delete_by_document_id(self, doc_id: str, owner_id: str) -> None:
        pass


class RAGService(ABC):
    @abstractmethod
    async def answer_question(
        self,
        question: str,
        context: list[Document],
        history: list[dict[str, str]] | None = None,
    ) -> str:
        pass

    @abstractmethod
    async def condense_question(self, question: str, history: list[dict[str, str]]) -> str:
        pass

    @abstractmethod
    def astream_answer(
        self,
        question: str,
        context: list[Document],
        history: list[dict[str, str]] | None = None,
    ) -> AsyncIterator[str]:
        ...


class RerankerService(ABC):
    @abstractmethod
    async def rerank(
        self, query: str, documents: list[Document], top_k: int = 4
    ) -> list[Document]:
        pass


class AnswerJudge(ABC):
    @abstractmethod
    async def score_faithfulness(self, answer: str, context: list[Document]) -> float:
        pass

    @abstractmethod
    async def score_answer_relevance(self, question: str, answer: str) -> float:
        pass


class SummarizerService(ABC):
    @abstractmethod
    async def summarize(self, documents: list[Document]) -> str:
        pass


class SummaryRepo(ABC):
    @abstractmethod
    async def save(self, summary: Summary, owner_id: str) -> str:
        pass

    @abstractmethod
    async def get_by_id(self, summary_id: str, owner_id: str) -> Summary | None:
        pass

    @abstractmethod
    async def list_all(
        self, owner_id: str, limit: int = 50, offset: int = 0
    ) -> list[Summary]:
        pass

    @abstractmethod
    async def delete(self, summary_id: str, owner_id: str) -> None:
        pass


class ConversationRepo(ABC):
    @abstractmethod
    async def save(self, conversation: Conversation, owner_id: str) -> None:
        pass

    @abstractmethod
    async def get_by_id(self, conversation_id: str, owner_id: str) -> Conversation | None:
        pass

    @abstractmethod
    async def list_all(
        self, owner_id: str, limit: int = 50, offset: int = 0
    ) -> list[Conversation]:
        pass

    @abstractmethod
    async def delete(self, conversation_id: str, owner_id: str) -> None:
        pass
