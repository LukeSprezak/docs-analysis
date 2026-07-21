"""Wspólna baza dla repozytoriów Postgresowych — usuwa powtarzany boilerplate.

Każde repo (dokumenty, streszczenia, konwersacje, użytkownicy) powielało ten sam
wzorzec: otwórz połączenie z puli, wykonaj zapytanie, pobierz wiersz(e). Baza
udostępnia trzy operacje wyższego rzędu (`_execute_statement`, `_fetch_one_row`,
`_fetch_all_rows`) plus pomocnik na deserializację kolumn JSONB.

Baza żyje w `app.shared`, bo z puli (`app.shared.database`) korzystają oba konteksty
(`knowledge_management` i `identity`) — wspólny rodzic nie sprzęga ich ze sobą.
"""

import json
from collections.abc import Mapping, Sequence
from typing import Any

from sqlalchemy import Row, text

from app.shared.database import db_connection


class BasePostgresRepo:
    async def _execute_statement(
        self, statement: str, parameters: Mapping[str, Any]
    ) -> None:
        """Wykonuje zapytanie bez zwracania wyniku (INSERT/UPDATE/DELETE)."""
        async with db_connection() as connection:
            await connection.execute(text(statement), parameters)

    async def _fetch_one_row(
        self, statement: str, parameters: Mapping[str, Any]
    ) -> Row[Any] | None:
        """Zwraca pierwszy wiersz wyniku albo None, gdy zapytanie nic nie zwróciło."""
        async with db_connection() as connection:
            result = await connection.execute(text(statement), parameters)
            return result.fetchone()

    async def _fetch_all_rows(
        self, statement: str, parameters: Mapping[str, Any]
    ) -> Sequence[Row[Any]]:
        """Zwraca wszystkie wiersze wyniku."""
        async with db_connection() as connection:
            result = await connection.execute(text(statement), parameters)
            return result.fetchall()

    @staticmethod
    def _deserialize_json_column(value: Any) -> Any:
        """Normalizuje kolumnę JSONB do obiektu Pythona.

        JSONB bywa zwracany już zdeserializowany (dict/list), a bywa jako surowy tekst
        JSON — zależnie od typu kolumny i sterownika. Deserializujemy tylko wtedy, gdy
        wartość jest jeszcze stringiem.
        """
        if isinstance(value, str):
            return json.loads(value)
        return value