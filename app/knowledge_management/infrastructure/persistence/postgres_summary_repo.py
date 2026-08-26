import json
from typing import Any

from app.shared.postgres_repo import BasePostgresRepo

from ...domain.models import Summary
from ...domain.repositories import SummaryRepo


class PostgresSummaryRepo(BasePostgresRepo, SummaryRepo):
    # Schema managed by Alembic (migrations/); connections come from the shared async pool
    # via BasePostgresRepo (not a per-call `psycopg.connect`).

    async def save(self, summary: Summary, owner_id: str) -> str:
        row = await _fetch_one_row(
            "INSERT INTO summaries (text, document_ids, owner_id) "
            "VALUES (:text, :document_ids, :owner_id) RETURNING id, created_at",
            {
                "text": summary.text,
                "document_ids": json.dumps(summary.document_ids),
                "owner_id": owner_id,
            },
        )
        if row is None:
            raise RuntimeError("INSERT ... RETURNING returned no row")
        summary_id = str(row[0])
        summary.id = summary_id
        summary.created_at = row[1].isoformat()
        return summary_id

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
