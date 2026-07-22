"""The graph turned off, as an object rather than a flag.

With `KNOWLEDGE_GRAPH_PROVIDER=none` the use cases still call `search_related` and
`add_fragment` — they just get nothing back. That keeps a single code path in the retrieval
pipeline instead of an `if self.graph_enabled` at every call site, and it makes "graph off"
behave identically to "graph on but empty" — a state the system has to handle correctly
anyway.

It lives in the domain rather than in `infrastructure/persistence/` on purpose: it touches no
database, driver or file, so it is the one implementation the application layer can default
to without reaching outward through the layers.
"""

from .models import Document, GraphFragment
from .repositories import KnowledgeGraphRepo


class NullKnowledgeGraphRepo(KnowledgeGraphRepo):
    async def add_fragment(self, fragment: GraphFragment, owner_id: str) -> None:
        return None

    async def search_related(self, query: str, owner_id: str, top_k: int = 4) -> list[Document]:
        return []

    async def delete_by_document_id(self, doc_id: str, owner_id: str) -> None:
        return None
