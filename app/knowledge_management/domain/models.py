from dataclasses import dataclass
from typing import Any


@dataclass
class Document:
    id: str
    content: str
    metadata: dict[str, Any]


@dataclass
class KnowledgeBase:
    id: str
    name: str
    documents: list[Document]


@dataclass
class Question:
    text: str


@dataclass
class Answer:
    text: str
    sources: list[Document]
    question: Question | None = None


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


@dataclass
class User:
    id: str
    email: str
    hashed_password: str
    created_at: str | None = None
