from dataclasses import dataclass


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
