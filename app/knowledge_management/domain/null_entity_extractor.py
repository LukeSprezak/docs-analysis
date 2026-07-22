"""Extraction turned off, as an object rather than a flag.

The counterpart of [NullKnowledgeGraphRepo][]: with no knowledge graph configured an upload
still "extracts", it just produces nothing — and so never pays for the LLM call. Lives in the
domain for the same reason: it performs no I/O, so the application layer may default to it.
"""

from .models import Document, GraphFragment
from .repositories import EntityExtractor


class NullEntityExtractor(EntityExtractor):
    async def extract(self, document: Document) -> GraphFragment:
        return GraphFragment(doc_id=document.id, entities=[], relations=[])
