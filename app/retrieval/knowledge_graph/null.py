"""The graph turned off, as objects rather than a flag.

With `KNOWLEDGE_GRAPH_PROVIDER=none` the use cases still call `search_related` and
`add_fragment` — they just get nothing back. That keeps a single code path in the retrieval
pipeline instead of an `if self.graph_enabled` at every call site, and it makes "graph off"
behave identically to "graph on but empty" — a state the system has to handle correctly
anyway.

`NullEntityExtractor` is its counterpart on the write side: with no graph an upload still
"extracts", it just produces nothing — and so never pays for the LLM call.
"""

from app.retrieval.models import GraphFragment
from app.retrieval.ports import EntityExtractor, KnowledgeGraphRepo
from app.shared.kernel.document import Document


class NullKnowledgeGraphRepo(KnowledgeGraphRepo):
    async def add_fragment(self, fragment: GraphFragment, owner_id: str) -> None:
        return None

    async def search_related(self, query: str, owner_id: str, top_k: int = 4) -> list[Document]:
        return []

    async def delete_by_document_id(self, doc_id: str, owner_id: str) -> None:
        return None


class NullEntityExtractor(EntityExtractor):
    async def extract(self, document: Document) -> GraphFragment:
        return GraphFragment(doc_id=document.id, entities=[], relations=[])
