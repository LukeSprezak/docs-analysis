"""Common base for Postgres repositories — removes the repeated boilerplate.

Every repo (documents, summaries, conversations, users) duplicated the same pattern: take
a connection from the pool, run the query, fetch the row(s). The base exposes three
higher-level operations (`_execute_statement`, `_fetch_one_row`, `_fetch_all_rows`) plus a
helper for deserializing JSONB columns.

The base lives in `app.shared` because both contexts (`knowledge_management` and
`identity`) use the pool (`app.shared.database`) — a shared parent does not couple them
to each other.
"""

import json
from collections.abc import Mapping, Sequence
from typing import Any

from sqlalchemy import Row, text

from app.shared.database import db_connection


class BasePostgresRepo:
    async def _execute_statement(self, statement: str, parameters: Mapping[str, Any]) -> None:
        """Runs a statement that returns no result (INSERT/UPDATE/DELETE)."""
        async with db_connection() as connection:
            await connection.execute(text(statement), parameters)

    async def _fetch_one_row(
        self, statement: str, parameters: Mapping[str, Any]
    ) -> Row[Any] | None:
        """Returns the first row of the result, or None when the query returned nothing."""
        async with db_connection() as connection:
            result = await connection.execute(text(statement), parameters)
            return result.fetchone()

    async def _fetch_all_rows(
        self, statement: str, parameters: Mapping[str, Any]
    ) -> Sequence[Row[Any]]:
        """Returns every row of the result."""
        async with db_connection() as connection:
            result = await connection.execute(text(statement), parameters)
            return result.fetchall()

    @staticmethod
    def _deserialize_json_column(value: Any) -> Any:
        """Normalizes a JSONB column into a Python object.

        JSONB comes back either already deserialized (dict/list) or as raw JSON text,
        depending on the column type and the driver. We only deserialize while the value
        is still a string.
        """
        if isinstance(value, str):
            return json.loads(value)
        return value
