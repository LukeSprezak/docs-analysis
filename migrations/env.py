from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from app.shared.config import settings

# Alembic Config object — dostęp do wartości z alembic.ini.
config = context.config

# URL do bazy bierzemy z konfiguracji aplikacji (jedno źródło prawdy: zmienne POSTGRES_*),
# zamiast duplikować go w alembic.ini. database_url używa sterownika psycopg3
# (postgresql+psycopg://), tego samego, który ma aplikacja.
config.set_main_option("sqlalchemy.url", settings.database_url)

# Logowanie wg pliku ini (jeśli podany).
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Projekt nie używa modeli ORM (raw psycopg + dataclasses) — migracje piszemy ręcznie,
# więc nie ma metadanych do autogenerate.
target_metadata = None


def run_migrations_offline() -> None:
    """Tryb offline — generuje SQL bez połączenia z bazą (np. do code review migracji)."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Tryb online — łączy się z bazą i wykonuje migracje."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
