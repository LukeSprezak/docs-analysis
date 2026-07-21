import json
from typing import Any

from app.shared.postgres_repo import BasePostgresRepo

from ...domain.models import ChatMessage, Conversation
from ...domain.repositories import ConversationRepo


class PostgresConversationRepo(BasePostgresRepo, ConversationRepo):
    # Schema zarządzana przez Alembic (migrations/); połączenia ze współdzielonej puli
    # async przez BasePostgresRepo (nie `psycopg.connect` per-wywołanie).

    async def save(self, conversation: Conversation, owner_id: str) -> None:
        messages_json = json.dumps(
            [
                {"role": m.role, "content": m.content, "timestamp": m.timestamp}
                for m in conversation.messages
            ]
        )
        await self._execute_statement(
            """
            INSERT INTO conversations (id, title, messages, owner_id)
            VALUES (:id, :title, :messages, :owner_id)
            ON CONFLICT (id) DO UPDATE
            SET title = EXCLUDED.title,
                messages = EXCLUDED.messages,
                owner_id = EXCLUDED.owner_id
            """,
            {
                "id": conversation.id,
                "title": conversation.title,
                "messages": messages_json,
                "owner_id": owner_id,
            },
        )

    async def get_by_id(self, conversation_id: str, owner_id: str) -> Conversation | None:
        row = await self._fetch_one_row(
            "SELECT id, title, messages, created_at FROM conversations "
            "WHERE id = :id AND owner_id = :owner_id",
            {"id": conversation_id, "owner_id": owner_id},
        )
        return self._row_to_conversation(row) if row else None

    async def list_all(
        self, owner_id: str, limit: int = 50, offset: int = 0
    ) -> list[Conversation]:
        rows = await self._fetch_all_rows(
            "SELECT id, title, messages, created_at FROM conversations "
            "WHERE owner_id = :owner_id ORDER BY created_at DESC "
            "LIMIT :limit OFFSET :offset",
            {"owner_id": owner_id, "limit": limit, "offset": offset},
        )
        return [self._row_to_conversation(row) for row in rows]

    async def delete(self, conversation_id: str, owner_id: str) -> None:
        await self._execute_statement(
            "DELETE FROM conversations WHERE id = :id AND owner_id = :owner_id",
            {"id": conversation_id, "owner_id": owner_id},
        )

    def _row_to_conversation(self, row: Any) -> Conversation:
        messages_raw = self._deserialize_json_column(row[2])
        messages = [
            ChatMessage(
                role=m["role"], content=m["content"], timestamp=m.get("timestamp")
            )
            for m in messages_raw
        ]
        return Conversation(
            id=str(row[0]),
            title=row[1],
            messages=messages,
            created_at=row[3].isoformat(),
        )