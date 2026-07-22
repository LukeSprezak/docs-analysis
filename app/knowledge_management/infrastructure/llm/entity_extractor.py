"""Entity/relation extraction — the step that turns prose into graph facts.

`LLMGraphTransformer` (shipped with langchain-neo4j) prompts the configured LLM to return
nodes and relationships for a passage. It is the expensive part of an upload: one LLM call
per document, on top of embedding. That cost is why the graph is opt-in per deployment
rather than always on.

The extractor is a port so the upload use case never branches on whether a graph is
configured; `KNOWLEDGE_GRAPH_PROVIDER=none` wires in the domain's `NullEntityExtractor`
instead of this one.
"""

import logging

from langchain_core.documents import Document as LCDocument
from langchain_core.language_models import BaseChatModel
from langchain_neo4j import LLMGraphTransformer

from ...domain.models import Document, Entity, GraphFragment, Relation
from ...domain.repositories import EntityExtractor

logger = logging.getLogger(__name__)


class LLMEntityExtractor(EntityExtractor):
    def __init__(self, llm: BaseChatModel, allowed_entity_types: list[str] | None = None):
        # An empty allowlist lets the model choose its own types. Constraining them produces a
        # tidier graph but silently drops anything outside the list, so it stays opt-in.
        self._transformer = LLMGraphTransformer(llm=llm, allowed_nodes=allowed_entity_types or [])

    async def extract(self, document: Document) -> GraphFragment:
        graph_documents = await self._transformer.aconvert_to_graph_documents(
            [LCDocument(page_content=document.content, metadata=document.metadata or {})]
        )
        if not graph_documents:
            return GraphFragment(doc_id=document.id, entities=[], relations=[])

        graph_document = graph_documents[0]
        entities = [Entity(name=str(node.id), type=node.type) for node in graph_document.nodes]
        relations = [
            Relation(
                source=Entity(name=str(rel.source.id), type=rel.source.type),
                target=Entity(name=str(rel.target.id), type=rel.target.type),
                type=rel.type,
            )
            for rel in graph_document.relationships
        ]
        logger.info(
            "Extracted %d entities and %d relations from %s",
            len(entities),
            len(relations),
            document.id,
        )
        return GraphFragment(doc_id=document.id, entities=entities, relations=relations)
