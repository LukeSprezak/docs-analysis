"""Knowledge graph stored in Neo4j.

Shape: `(:GraphEntity {name, type, owner_id, doc_ids})-[:RELATES {type, doc_ids}]->(:GraphEntity)`.

Two properties carry the weight here:

* **`owner_id` is part of entity identity.** Entities merge on `(owner_id, name)`, never on
  name alone, so two users writing about "Kubernetes" get two separate nodes. Merging them
  would silently join one user's facts to another's — the graph equivalent of handing over
  someone else's documents.
* **`doc_ids` is a list, not a single value.** The same fact usually appears in several
  documents. Deleting one document must retract only that document's claim, so a delete
  removes the id from the list and drops the node or relationship once the list empties.
  Storing one id would make the last delete wipe facts other documents still assert.

Entity names come from an LLM, so the same thing arrives spelled differently ("Postgres" vs
"PostgreSQL") and lands as two nodes. Resolving that is a separate problem (entity
resolution) and deliberately out of scope here.
"""

import logging
from typing import Any

from langchain_core.runnables.config import run_in_executor
from langchain_neo4j import Neo4jGraph

from ...domain.models import Document, GraphFragment
from ...domain.repositories import KnowledgeGraphRepo

logger = logging.getLogger(__name__)


class Neo4jKnowledgeGraphRepo(KnowledgeGraphRepo):
    def __init__(
        self,
        url: str,
        username: str,
        password: str,
        database: str | None = None,
        node_label: str = "GraphEntity",
        relationship_type: str = "RELATES",
        index_name: str = "graph_entity_names",
    ) -> None:
        self.node_label = node_label
        self.relationship_type = relationship_type
        self.index_name = index_name
        self.graph = Neo4jGraph(
            url=url,
            username=username,
            password=password,
            database=database,
            refresh_schema=False,  # we own the schema; introspecting it on every boot is waste
        )
        self._ensure_indexes()

    def _ensure_indexes(self) -> None:
        """Full-text index for entity lookup, plus a lookup index for the merge key.

        Both are `IF NOT EXISTS`, so this is safe to run on every startup. A composite
        *uniqueness* constraint on `(owner_id, name)` would be the stronger guarantee, but
        `MERGE` already enforces it in practice and a plain index keeps this working on the
        community edition.
        """
        self.graph.query(
            f"CREATE FULLTEXT INDEX {self.index_name} IF NOT EXISTS "
            f"FOR (entity:`{self.node_label}`) ON EACH [entity.name]"
        )
        self.graph.query(
            f"CREATE INDEX {self.index_name}_owner IF NOT EXISTS "
            f"FOR (entity:`{self.node_label}`) ON (entity.owner_id, entity.name)"
        )

    async def add_fragment(self, fragment: GraphFragment, owner_id: str) -> None:
        # Replace semantics: retract what this document said before, so a re-upload does not
        # leave behind facts the new version no longer contains.
        await self.delete_by_document_id(fragment.doc_id, owner_id)
        if fragment.is_empty():
            return

        await self._query(
            f"UNWIND $entities AS entity "
            f"MERGE (node:`{self.node_label}` {{owner_id: $owner_id, name: entity.name}}) "
            f"SET node.type = entity.type, "
            f"    node.doc_ids = coalesce(node.doc_ids, []) + "
            f"        CASE WHEN $doc_id IN coalesce(node.doc_ids, []) THEN [] ELSE [$doc_id] END",
            {
                "entities": [{"name": e.name, "type": e.type} for e in fragment.entities],
                "owner_id": owner_id,
                "doc_id": fragment.doc_id,
            },
        )

        if not fragment.relations:
            return
        await self._query(
            f"UNWIND $relations AS relation "
            f"MERGE (source:`{self.node_label}` "
            f"       {{owner_id: $owner_id, name: relation.source}}) "
            f"MERGE (target:`{self.node_label}` "
            f"       {{owner_id: $owner_id, name: relation.target}}) "
            f"MERGE (source)-[rel:`{self.relationship_type}` {{type: relation.type}}]->(target) "
            f"SET rel.doc_ids = coalesce(rel.doc_ids, []) + "
            f"    CASE WHEN $doc_id IN coalesce(rel.doc_ids, []) THEN [] ELSE [$doc_id] END",
            {
                "relations": [
                    {"source": r.source.name, "target": r.target.name, "type": r.type}
                    for r in fragment.relations
                ],
                "owner_id": owner_id,
                "doc_id": fragment.doc_id,
            },
        )

    async def search_related(self, query: str, owner_id: str, top_k: int = 4) -> list[Document]:
        """Finds the entities the query names, then returns the facts hanging off them.

        The full-text hit is only the entry point — the value is the neighbourhood, which is
        what a vector search over passages cannot reconstruct.
        """
        rows = await self._query(
            "CALL db.index.fulltext.queryNodes($index_name, $query, {limit: $candidates}) "
            "YIELD node, score "
            "WHERE node.owner_id = $owner_id "
            "WITH node, score ORDER BY score DESC LIMIT $seed_limit "
            f"MATCH (source:`{self.node_label}`)-[rel:`{self.relationship_type}`]->"
            f"      (target:`{self.node_label}`) "
            "WHERE (source = node OR target = node) "
            "  AND source.owner_id = $owner_id AND target.owner_id = $owner_id "
            "RETURN DISTINCT source.name AS source, rel.type AS type, target.name AS target, "
            "       rel.doc_ids AS doc_ids, score "
            "ORDER BY score DESC LIMIT $top_k",
            {
                "index_name": self.index_name,
                "query": _escape_lucene(query),
                "owner_id": owner_id,
                "candidates": max(top_k * 5, 50),
                "seed_limit": max(top_k, 4),
                "top_k": top_k,
            },
        )
        return [self._to_document(row) for row in rows]

    async def delete_by_document_id(self, doc_id: str, owner_id: str) -> None:
        # Drop this document's claim, then remove whatever no document asserts any more.
        await self._query(
            f"MATCH (source:`{self.node_label}` {{owner_id: $owner_id}})"
            f"-[rel:`{self.relationship_type}`]->(:`{self.node_label}`) "
            "WHERE $doc_id IN rel.doc_ids "
            "SET rel.doc_ids = [d IN rel.doc_ids WHERE d <> $doc_id] "
            "WITH rel WHERE size(rel.doc_ids) = 0 "
            "DELETE rel",
            {"doc_id": doc_id, "owner_id": owner_id},
        )
        await self._query(
            f"MATCH (node:`{self.node_label}` {{owner_id: $owner_id}}) "
            "WHERE $doc_id IN node.doc_ids "
            "SET node.doc_ids = [d IN node.doc_ids WHERE d <> $doc_id] "
            "WITH node WHERE size(node.doc_ids) = 0 "
            "DETACH DELETE node",
            {"doc_id": doc_id, "owner_id": owner_id},
        )

    async def close(self) -> None:
        await run_in_executor(None, self.graph.close)

    async def _query(self, cypher: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        """Runs Cypher off the event loop (Neo4jGraph wraps the synchronous driver)."""
        return await run_in_executor(None, lambda: self.graph.query(cypher, params=params))

    @staticmethod
    def _to_document(row: dict[str, Any]) -> Document:
        """Renders a triple as a sentence the LLM can read as context."""
        doc_ids = row.get("doc_ids") or []
        return Document(
            id=str(doc_ids[0]) if doc_ids else "knowledge-graph",
            content=f"{row['source']} {row['type']} {row['target']}",
            metadata={
                # Marks the provenance so an answer can say the fact came from the graph
                # rather than from a quoted passage.
                "source": "knowledge_graph",
                "doc_ids": list(doc_ids),
            },
        )


def _escape_lucene(query: str) -> str:
    """Neutralizes Lucene syntax in user input — see the note in `neo4j_vectorstore_repo`."""
    special_characters = r'+-&|!(){}[]^"~*?:\/'
    return "".join(f"\\{char}" if char in special_characters else char for char in query)
