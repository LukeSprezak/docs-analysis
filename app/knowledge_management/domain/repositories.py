from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from .models import Conversation, Document, GraphFragment, Summary


class DocumentRepo(ABC):
    @abstractmethod
    async def save(self, document: Document, owner_id: str) -> None:
        pass

    @abstractmethod
    async def get_by_id(self, doc_id: str, owner_id: str) -> Document | None:
        pass

    @abstractmethod
    async def list_all(self, owner_id: str, limit: int = 50, offset: int = 0) -> list[Document]:
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

    async def close(self) -> None:  # noqa: B027 — the empty default is the point, see below
        """Releases whatever connection the adapter opened for itself.

        Called from the application lifespan on shutdown. Deliberately not abstract: an
        in-memory store has nothing to release, and the Postgres one borrows the shared pool
        that `dispose_engine()` already closes — only an adapter owning its own driver (Neo4j)
        overrides this. Keeping it on the port means shutdown code never has to ask which
        adapter it is holding."""


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
        """Rephrases a question that depends on conversation context into a standalone one
        (for retrieval). E.g. 'and what about that?' → 'what is quicksort's complexity?'."""
        pass

    @abstractmethod
    def astream_answer(
        self,
        question: str,
        context: list[Document],
        history: list[dict[str, str]] | None = None,
    ) -> AsyncIterator[str]:
        """Streams the answer token by token (for live chat UX)."""
        ...


class RerankerService(ABC):
    @abstractmethod
    async def rerank(self, query: str, documents: list[Document], top_k: int = 4) -> list[Document]:
        """Orders candidates by relevance to the query and returns the best top_k."""
        pass


class AnswerJudge(ABC):
    """Answer quality judge (LLM-as-judge) used by the RAG evaluation.

    RAGAS-style metrics, but computed with our own judge (without the heavy `ragas`
    dependency). The implementation is optional and enabled by a flag — see the factory.
    """

    @abstractmethod
    async def score_faithfulness(self, answer: str, context: list[Document]) -> float:
        """How well the answer is grounded in the supplied context (0.0-1.0).
        A low value means hallucination / content from outside the context."""
        pass

    @abstractmethod
    async def score_answer_relevance(self, question: str, answer: str) -> float:
        """How well the answer actually addresses the question asked (0.0-1.0)."""
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
    async def list_all(self, owner_id: str, limit: int = 50, offset: int = 0) -> list[Summary]:
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
    async def list_all(self, owner_id: str, limit: int = 50, offset: int = 0) -> list[Conversation]:
        pass

    @abstractmethod
    async def delete(self, conversation_id: str, owner_id: str) -> None:
        pass


class KnowledgeGraphRepo(ABC):
    """The knowledge graph — a retrieval source *alongside* the vector store, not instead of it.

    Vector search finds passages that read like the question. The graph answers a different
    shape of question: what a thing is connected to, and how. Facts extracted from separate
    documents join on entity name, so the graph can surface a connection no single passage
    states.

    `search_related` returns `Document`s so the results drop straight into the same context
    list the RAG service already consumes — the retrieval pipeline fuses the two rankings and
    stays unaware of where each candidate came from.
    """

    @abstractmethod
    async def add_fragment(self, fragment: GraphFragment, owner_id: str) -> None:
        """Merges one document's facts into the graph, replacing that document's previous ones."""

    @abstractmethod
    async def search_related(self, query: str, owner_id: str, top_k: int = 4) -> list[Document]:
        """Facts connected to the entities the query mentions, as readable statements."""

    @abstractmethod
    async def delete_by_document_id(self, doc_id: str, owner_id: str) -> None:
        """Retracts the facts this document asserted, leaving other documents' facts intact."""

    async def close(self) -> None:  # noqa: B027 — see VectorStoreRepo.close
        """Releases whatever connection the adapter opened for itself."""


class EntityExtractor(ABC):
    """Turns document text into entities and relations (typically an LLM call)."""

    @abstractmethod
    async def extract(self, document: Document) -> GraphFragment:
        pass
