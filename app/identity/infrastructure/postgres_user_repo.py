from typing import Any

from app.identity.domain.models import User
from app.identity.domain.repositories import UserRepo
from app.shared.postgres_repo import BasePostgresRepo


class PostgresUserRepo(BasePostgresRepo, UserRepo):
    async def get_by_email(self, email: str) -> User | None:
        row = await self._fetch_one_row(
            "SELECT id, email, hashed_password, created_at "
            "FROM users WHERE email = :email",
            {"email": email},
        )
        return self._row_to_user(row)

    async def get_by_id(self, user_id: str) -> User | None:
        row = await self._fetch_one_row(
            "SELECT id, email, hashed_password, created_at "
            "FROM users WHERE id = :user_id",
            {"user_id": user_id},
        )
        return self._row_to_user(row)

    async def save(self, user: User) -> None:
        await self._execute_statement(
            """
            INSERT INTO users (id, email, hashed_password)
            VALUES (:id, :email, :hashed_password)
            ON CONFLICT (id) DO UPDATE
            SET email = EXCLUDED.email,
                hashed_password = EXCLUDED.hashed_password
            """,
            {
                "id": user.id,
                "email": user.email,
                "hashed_password": user.hashed_password,
            },
        )

    @staticmethod
    def _row_to_user(row: Any) -> User | None:
        if row is None:
            return None
        return User(
            id=str(row[0]),
            email=row[1],
            hashed_password=row[2],
            created_at=row[3].isoformat() if row[3] else None,
        )
