"""initial schema

Revision ID: 0001_initial_schema
Revises:
Create Date: 2026-06-07 22:32:10.066136

The application schema, moved out of `_ensure_table_exists()` in the repositories and into a
migration. It covers the four tables the application owns: users, documents, summaries and
conversations (with `owner_id` folded into the table definition instead of a separate ALTER).

BOUNDARY: the vector tables `langchain_pg_collection` / `langchain_pg_embedding` are managed
by `langchain_postgres` (PGVector creates them itself), so they are deliberately NOT part of
this migration. The `vector` extension is created here because pgvector requires it, and it
belongs to a migration rather than to application code.
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
    # The `vector` extension stays: PGVector tables outside this migration's control may
    # still be using it.
