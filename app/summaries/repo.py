import json
from typing import Any, Protocol

from app.shared.postgres_repo import (
    BasePostgresRepo,
    _execute_statement,
    _fetch_all_rows,
    _fetch_one_row,
)
from app.summaries.models import Summary


class SummaryRepo(Protocol):
    async def save(self, text: str, document_ids: list[str], owner_id: str) -> Summary:
        """Stores a new summary and returns it, with the id and timestamp the store assigned.

        Takes the content rather than a `Summary`: the identity is the store's to mint, so
        there is no half-built summary for a caller to hold on to."""
        ...

    async def get_by_id(self, summary_id: str, owner_id: str) -> Summary | None: ...

    async def list_all(self, owner_id: str, limit: int = 50, offset: int = 0) -> list[Summary]: ...

    async def delete(self, summary_id: str, owner_id: str) -> None: ...


class PostgresSummaryRepo(BasePostgresRepo):
    # Schema managed by Alembic (migrations/); connections come from the shared async pool
    # via BasePostgresRepo (not a per-call `psycopg.connect`).

    async def save(self, text: str, document_ids: list[str], owner_id: str) -> Summary:
        row = await _fetch_one_row(
            "INSERT INTO summaries (text, document_ids, owner_id) "
            "VALUES (:text, :document_ids, :owner_id) RETURNING id, created_at",
            {
                "text": text,
                "document_ids": json.dumps(document_ids),
                "owner_id": owner_id,
            },
        )
        if row is None:
            raise RuntimeError("INSERT ... RETURNING returned no row")
        return Summary(
            text=text,
            document_ids=document_ids,
            id=str(row[0]),
            created_at=row[1].isoformat(),
        )

    async def get_by_id(self, summary_id: str, owner_id: str) -> Summary | None:
        row = await _fetch_one_row(
            "SELECT text, document_ids, id, created_at FROM summaries "
            "WHERE id = :id AND owner_id = :owner_id",
            {"id": summary_id, "owner_id": owner_id},
        )
        return self._row_to_summary(row) if row else None

    async def list_all(self, owner_id: str, limit: int = 50, offset: int = 0) -> list[Summary]:
        rows = await _fetch_all_rows(
            "SELECT text, document_ids, id, created_at FROM summaries "
            "WHERE owner_id = :owner_id ORDER BY created_at DESC "
            "LIMIT :limit OFFSET :offset",
            {"owner_id": owner_id, "limit": limit, "offset": offset},
        )
        return [self._row_to_summary(row) for row in rows]

    async def delete(self, summary_id: str, owner_id: str) -> None:
        await _execute_statement(
            "DELETE FROM summaries WHERE id = :id AND owner_id = :owner_id",
            {"id": summary_id, "owner_id": owner_id},
        )

    def _row_to_summary(self, row: Any) -> Summary:
        return Summary(
            text=row[0],
            document_ids=self._deserialize_json_column(row[1]),
            id=str(row[2]),
            created_at=row[3].isoformat(),
        )
