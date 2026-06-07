from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from .models import Conversation, Document, KnowledgeBase, Summary, User


class DocumentRepo(ABC):
    @abstractmethod
    def save(self, document: Document, owner_id: str) -> None:
        pass

    @abstractmethod
    def get_by_id(self, doc_id: str, owner_id: str) -> Document | None:
        pass

    @abstractmethod
    def list_all(self, owner_id: str) -> list[Document]:
        pass

    @abstractmethod
    def delete(self, doc_id: str, owner_id: str) -> None:
        pass


class KnowledgeBaseRepository(ABC):
    @abstractmethod
    def get_kb_by_id(self, kb_id: str) -> KnowledgeBase | None:
        pass

    @abstractmethod
    def list_all_kbs(self) -> list[KnowledgeBase]:
        pass

    @abstractmethod
    def save_kb(self, kb: KnowledgeBase) -> None:
        pass


class VectorStoreRepo(ABC):
    @abstractmethod
    def add_documents(self, documents: list[Document], owner_id: str) -> None:
        pass

    @abstractmethod
    def search(self, query: str, owner_id: str, top_k: int = 4) -> list[Document]:
        pass

    @abstractmethod
    def delete_by_document_id(self, doc_id: str, owner_id: str) -> None:
        pass


class RAGService(ABC):
    @abstractmethod
    def answer_question(
        self,
        question: str,
        context: list[Document],
        history: list[dict[str, str]] | None = None,
    ) -> str:
        pass

    @abstractmethod
    def condense_question(self, question: str, history: list[dict[str, str]]) -> str:
        """Przeformułowuje pytanie zależne od kontekstu rozmowy na samodzielne
        (do retrievalu). Np. 'a co z tym?' → 'jaka jest złożoność quicksort?'."""
        pass

    @abstractmethod
    def astream_answer(
        self,
        question: str,
        context: list[Document],
        history: list[dict[str, str]] | None = None,
    ) -> AsyncIterator[str]:
        """Strumieniuje odpowiedź token po tokenie (do UX czatu na żywo)."""
        ...


class RerankerService(ABC):
    @abstractmethod
    def rerank(
        self, query: str, documents: list[Document], top_k: int = 4
    ) -> list[Document]:
        """Sortuje kandydatów wg trafności do zapytania i zwraca najlepsze top_k."""
        pass


class AnswerJudge(ABC):
    """Sędzia jakości odpowiedzi (LLM-as-judge) na potrzeby ewaluacji RAG.

    Metryki w stylu RAGAS, ale liczone własnym sędzią (bez ciężkiej zależności
    `ragas`). Implementacja jest opcjonalna i włączana flagą — patrz factory.
    """

    @abstractmethod
    def score_faithfulness(self, answer: str, context: list[Document]) -> float:
        """Na ile odpowiedź jest ugruntowana w dostarczonym kontekście (0.0–1.0).
        Niska wartość = halucynacja / treść spoza kontekstu."""
        pass

    @abstractmethod
    def score_answer_relevance(self, question: str, answer: str) -> float:
        """Na ile odpowiedź faktycznie adresuje zadane pytanie (0.0–1.0)."""
        pass


class SummarizerService(ABC):
    @abstractmethod
    def summarize(self, documents: list[Document]) -> str:
        pass


class SummaryRepo(ABC):
    @abstractmethod
    def save(self, summary: Summary, owner_id: str) -> str:
        pass

    @abstractmethod
    def get_by_id(self, summary_id: str, owner_id: str) -> Summary | None:
        pass

    @abstractmethod
    def list_all(self, owner_id: str) -> list[Summary]:
        pass

    @abstractmethod
    def delete(self, summary_id: str, owner_id: str) -> None:
        pass


class UserRepo(ABC):
    @abstractmethod
    def get_by_email(self, email: str) -> User | None:
        pass

    @abstractmethod
    def get_by_id(self, user_id: str) -> User | None:
        pass

    @abstractmethod
    def save(self, user: User) -> None:
        pass


class ConversationRepo(ABC):
    @abstractmethod
    def save(self, conversation: Conversation, owner_id: str) -> None:
        pass

    @abstractmethod
    def get_by_id(self, conversation_id: str, owner_id: str) -> Conversation | None:
        pass

    @abstractmethod
    def list_all(self, owner_id: str) -> list[Conversation]:
        pass

    @abstractmethod
    def delete(self, conversation_id: str, owner_id: str) -> None:
        pass
