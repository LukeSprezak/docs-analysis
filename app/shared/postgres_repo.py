import json
from collections.abc import Mapping, Sequence
from typing import Any

from sqlalchemy import Row, text

from app.shared.database import db_connection


class BasePostgresRepo:
    async def _execute_statement(
        self, statement: str, parameters: Mapping[str, Any]
    ) -> None:
        async with db_connection() as connection:
            await connection.execute(text(statement), parameters)

    async def _fetch_one_row(
        self, statement: str, parameters: Mapping[str, Any]
    ) -> Row[Any] | None:
        async with db_connection() as connection:
            result = await connection.execute(text(statement), parameters)
            return result.fetchone()

    async def _fetch_all_rows(
        self, statement: str, parameters: Mapping[str, Any]
    ) -> Sequence[Row[Any]]:
        async with db_connection() as connection:
            result = await connection.execute(text(statement), parameters)
            return result.fetchall()

    @staticmethod
    def _deserialize_json_column(value: Any) -> Any:
        if isinstance(value, str):
            return json.loads(value)
        return value
