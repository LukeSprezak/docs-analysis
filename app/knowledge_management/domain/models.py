from dataclasses import dataclass
from typing import Any


@dataclass
class Document:
    id: str
    content: str
    metadata: dict[str, Any]


@dataclass
class Answer:
    text: str
    sources: list[Document]


@dataclass(frozen=True)
class Entity:
    """A thing mentioned in a document — a person, system, concept.

    `name` is the identity: two mentions with the same name are the same entity, which is
    what lets facts from different documents join up into one graph.
    """

    name: str
    type: str


@dataclass(frozen=True)
class Relation:
    """A directed fact: `source` --type--> `target`."""

    source: Entity
    target: Entity
    type: str


@dataclass
class GraphFragment:
    """Everything one document contributed to the knowledge graph.

    Kept per document so a delete can retract exactly that document's contribution without
    disturbing facts other documents asserted about the same entities.
    """

    doc_id: str
    entities: list[Entity]
    relations: list[Relation]

    def is_empty(self) -> bool:
        return not self.entities and not self.relations


@dataclass
class Summary:
    text: str
    document_ids: list[str]
    id: str | None = None
    created_at: str | None = None


@dataclass
class ChatMessage:
    role: str
    content: str
    timestamp: str | None = None


@dataclass
class Conversation:
    id: str
    title: str
    messages: list[ChatMessage]
    created_at: str | None = None
