from collections.abc import AsyncIterator
from typing import Protocol

from app.retrieval.models import GraphFragment
from app.shared.kernel.document import Document


class VectorStoreRepo(Protocol):
    async def add_documents(self, documents: list[Document], owner_id: str) -> None: ...

    async def search(self, query: str, owner_id: str, top_k: int = 4) -> list[Document]:
        """Chunks found by vector similarity carry their relevance (0-1, higher is better) in
        `metadata["score"]`; chunks found only by keyword search (hybrid mode) have none."""

    async def delete_by_document_id(self, doc_id: str, owner_id: str) -> None: ...

    async def count(self) -> int:
        """Chunks stored across all owners."""

    async def close(self) -> None:
        """Releases whatever connection the adapter opened for itself.

        Called from the application lifespan on shutdown. Deliberately not abstract: an
        in-memory store has nothing to release, and the Postgres one borrows the shared pool
        that `dispose_engine()` already closes — only an adapter owning its own driver (Neo4j)
        overrides this. Keeping it on the port means shutdown code never has to ask which
        adapter it is holding. Adapters subclass the port explicitly to inherit this default."""
        return None


class RAGService(Protocol):
    async def answer_question(
        self,
        question: str,
        context: list[Document],
        history: list[dict[str, str]] | None = None,
    ) -> str: ...

    async def condense_question(self, question: str, history: list[dict[str, str]]) -> str:
        """Rephrases a question that depends on conversation context into a standalone one
        (for retrieval). E.g. 'and what about that?' → 'what is quicksort's complexity?'."""
        ...

    def astream_answer(
        self,
        question: str,
        context: list[Document],
        history: list[dict[str, str]] | None = None,
    ) -> AsyncIterator[str]:
        """Streams the answer token by token (for live chat UX)."""
        ...


class RerankerService(Protocol):
    async def rerank(self, query: str, documents: list[Document], top_k: int = 4) -> list[Document]:
        """Orders candidates by relevance to the query and returns the best top_k."""
        ...


class AnswerJudge(Protocol):
    """Answer quality judge (LLM-as-judge) used by the RAG evaluation.

    RAGAS-style metrics, but computed with our own judge (without the heavy `ragas`
    dependency). The implementation is optional and enabled by a flag — see the factory.
    """

    async def score_faithfulness(self, answer: str, context: list[Document]) -> float | None:
        """How well the answer is grounded in the supplied context (0.0-1.0).
        A low value means hallucination / content from outside the context; `None` means the
        judge failed to produce a score, which is not the same as scoring zero."""
        ...

    async def score_answer_relevance(self, question: str, answer: str) -> float | None:
        """How well the answer actually addresses the question asked (0.0-1.0), or `None` when
        the judge failed to produce a score."""
        ...


class KnowledgeGraphRepo(Protocol):
    """The knowledge graph — a retrieval source *alongside* the vector store, not instead of it.

    Vector search finds passages that read like the question. The graph answers a different
    shape of question: what a thing is connected to, and how. Facts extracted from separate
    documents join on entity name, so the graph can surface a connection no single passage
    states.

    `search_related` returns `Document`s so the results drop straight into the same context
    list the RAG service already consumes — the retrieval pipeline fuses the two rankings and
    stays unaware of where each candidate came from.
    """

    async def add_fragment(self, fragment: GraphFragment, owner_id: str) -> None:
        """Merges one document's facts into the graph, replacing that document's previous ones."""

    async def search_related(self, query: str, owner_id: str, top_k: int = 4) -> list[Document]:
        """Facts connected to the entities the query mentions, as readable statements."""

    async def delete_by_document_id(self, doc_id: str, owner_id: str) -> None:
        """Retracts the facts this document asserted, leaving other documents' facts intact."""

    async def close(self) -> None:
        """Releases whatever connection the adapter opened for itself."""
        return None


class EntityExtractor(Protocol):
    """Turns document text into entities and relations (typically an LLM call)."""

    async def extract(self, document: Document) -> GraphFragment: ...
