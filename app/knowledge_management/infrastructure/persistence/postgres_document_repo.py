import json
from typing import Any

from app.shared.postgres_repo import BasePostgresRepo

from ...domain.models import Document
from ...domain.repositories import DocumentRepo


class PostgresDocumentRepo(BasePostgresRepo, DocumentRepo):
    # Schema zarządzana przez Alembic (migrations/); połączenia ze współdzielonej puli
    # async przez BasePostgresRepo (nie `psycopg.connect` per-wywołanie).

    async def save(self, document: Document, owner_id: str) -> None:
        await self._execute_statement(
            """
            INSERT INTO documents (id, content, metadata, owner_id)
            VALUES (:id, :content, :metadata, :owner_id)
            ON CONFLICT (id) DO UPDATE
            SET content = EXCLUDED.content,
                metadata = EXCLUDED.metadata,
                owner_id = EXCLUDED.owner_id
            """,
            {
                "id": document.id,
                "content": document.content,
                "metadata": json.dumps(document.metadata),
                "owner_id": owner_id,
            },
        )

    async def get_by_id(self, doc_id: str, owner_id: str) -> Document | None:
        row = await self._fetch_one_row(
            "SELECT id, content, metadata FROM documents "
            "WHERE id = :id AND owner_id = :owner_id",
            {"id": doc_id, "owner_id": owner_id},
        )
        return self._row_to_document(row) if row else None

    async def list_all(
        self, owner_id: str, limit: int = 50, offset: int = 0
    ) -> list[Document]:
        rows = await self._fetch_all_rows(
            "SELECT id, content, metadata FROM documents "
            "WHERE owner_id = :owner_id ORDER BY created_at DESC "
            "LIMIT :limit OFFSET :offset",
            {"owner_id": owner_id, "limit": limit, "offset": offset},
        )
        return [self._row_to_document(row) for row in rows]

    async def delete(self, doc_id: str, owner_id: str) -> None:
        await self._execute_statement(
            "DELETE FROM documents WHERE id = :id AND owner_id = :owner_id",
            {"id": doc_id, "owner_id": owner_id},
        )

    def _row_to_document(self, row: Any) -> Document:
        return Document(
            id=row[0],
            content=row[1],
            metadata=self._deserialize_json_column(row[2]),
        )