"""initial schema

Revision ID: 0001_initial_schema
Revises:
Create Date: 2026-06-07 22:32:10.066136

Schemat aplikacji przeniesiony z `_ensure_table_exists()` w repozytoriach do migracji.
Obejmuje cztery tabele należące do aplikacji: users, documents, summaries, conversations
(`owner_id` skonsolidowany do definicji tabeli zamiast osobnego ALTER-a).

GRANICA: tabele wektorowe `langchain_pg_collection` / `langchain_pg_embedding` są
zarządzane przez bibliotekę `langchain_postgres` (PGVector tworzy je automatycznie),
więc świadomie NIE są częścią tej migracji. Rozszerzenie `vector` jest tu tworzone, bo
jest wymagane przez pgvector i lepiej, żeby należało do migracji niż do kodu aplikacji.
"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0001_initial_schema"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            email TEXT NOT NULL UNIQUE,
            hashed_password TEXT NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            metadata JSONB NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            owner_id TEXT
        )
        """
    )

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS summaries (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            text TEXT NOT NULL,
            document_ids JSONB NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            owner_id TEXT
        )
        """
    )

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS conversations (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            title TEXT NOT NULL,
            messages JSONB NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            owner_id TEXT
        )
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS conversations")
    op.execute("DROP TABLE IF EXISTS summaries")
    op.execute("DROP TABLE IF EXISTS documents")
    op.execute("DROP TABLE IF EXISTS users")
    # Rozszerzenie `vector` zostawiamy — może być używane przez tabele PGVector
    # poza zarządzaniem tej migracji.
