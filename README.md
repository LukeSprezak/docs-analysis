# docs-analysis

A RAG service over your own documents: upload PDFs and text files, then ask questions,
chat with follow-ups, or summarize. Retrieval combines vector search, full-text search and
— optionally — a knowledge graph built from the documents themselves.

Every document, conversation and summary belongs to a user, and every query is filtered by
that owner. The backing stores are swappable behind ports, and the contract suites in
`tests/contracts/` are what keep a new adapter honest.

## Stack

| Area | Choice |
|---|---|
| API | FastAPI, Python 3.12, `uv` |
| Records | Postgres (documents, summaries, conversations, users) via SQLAlchemy async + psycopg3 |
| Vectors | pgvector, FAISS or Neo4j |
| Knowledge graph | Neo4j, or `none` |
| LLM | OpenAI, Anthropic, Google or Ollama (LangChain) |
| Reranking | none, LLM, Cohere, or a local BGE cross-encoder |
| Frontend | React 19 + Vite + TypeScript ([`client/CLAUDE.md`](client/CLAUDE.md)) |
| Migrations | Alembic |

## Quick start

Requires Docker and [`uv`](https://docs.astral.sh/uv/). Node/corepack only if you work on
the frontend outside its container.

```bash
cp .env.example .env      # then fill in at least the LLM key and the passwords
./scripts/setup.sh        # deps, containers, migrations, lint, tests
```

The script is idempotent — it reuses existing images unless you pass `--build`. When it
finishes:

- API and OpenAPI docs → http://localhost:8001/docs
- Frontend → http://localhost:3000

Every setting is required at startup: a missing variable is a startup error, not a silent
default. The exception is what is genuinely optional (provider keys you are not using).

## Everyday commands

```bash
docker compose up -d                       # start everything
docker compose up -d --build api           # rebuild after changing dependencies
docker compose logs -f api
docker compose exec api uv run alembic upgrade head   # migrations (also run by the entrypoint)

uv run pytest                              # unit + use-case tests (integration excluded)
./scripts/lint.sh                          # ruff check + ruff format --check + mypy
./scripts/lint-fix.sh                      # the same, applying the fixes it can
```

After `uv add` on the host, recreate the container — `.venv` is a named volume, so a host
install never reaches it:

```bash
docker compose up -d --force-recreate api
```

## Tests

```bash
uv run pytest                              # the default run; `integration` is deselected
POSTGRES_HOST=localhost POSTGRES_PORT=5433 \
  NEO4J_URI=bolt://localhost:7687 \
  uv run pytest -m integration              # everything that needs a live server
cd client && yarn test                     # Vitest, the data layer in src/core
```

`tests/contracts/` holds one suite per port (documents, summaries, conversations, users,
vector store, knowledge graph). Every adapter runs the *same* assertions, so a second
backing store is only finished once it is added to that suite's `ADAPTERS` and passes
unchanged. The server-backed parameters are marked `integration` and stay out of the
default run.

Ports from the host: Postgres on `5433`, Neo4j Bolt on `7687` and its browser UI on
`7474` (see `docker-compose.override.yml`). From inside the compose network the hostnames
are `db` and `graph` instead.

## Layout

Two bounded contexts, each in domain / application / infrastructure / ui layers:

```
app/
  identity/              users, registration, login, JWT
  knowledge_management/  documents, retrieval, chat, summaries, the knowledge graph
  shared/                config, database pool, exceptions, storage, rate limiting
client/                  React SPA
migrations/              Alembic
eval/                    golden set for the evaluation harness
tests/  contracts/       one suite per port, parametrized over adapters
```

The domain layer holds the rules that outlive any adapter — how a document id is built and
taken apart (`document_identity.py`), how an entity name is normalized
(`entity_normalization.py`) — and depends on nothing below it.

## Knowledge graph

Off by default (`KNOWLEDGE_GRAPH_PROVIDER=none`), and that is a real choice rather than an
unconfigured one: it wires in a null repository and a null extractor, so uploads skip the
entity-extraction LLM call and retrieval stays vector-only, with no branch in the use cases.

Turning it on costs an LLM call per upload, so it is worth measuring first — the harness
below is there for exactly that. **Set the provider before loading the corpus:** facts are
extracted during upload, so documents uploaded with the graph off have no entities in it.

## Evaluation harness

Retrieval metrics are offline and deterministic; generation metrics (faithfulness,
answer relevance) need an LLM judge and are enabled separately.

```bash
docker compose exec api uv run python -m \
  app.knowledge_management.application.evaluation.run_evaluation \
  --dataset eval/golden_set.json --owner-id <user_id> \
  --compare-graph --json eval/last-run.json
```

`--compare-graph` runs each question twice, with and without the graph, so the two rankings
can be compared directly. `--fail-on-regression` turns the run into a gate; hold off until
the golden set has enough questions per category to be stable (see `BACKLOG.md` §5).

Only the template is committed (`eval/golden_set.template.json`) — a golden set is corpus
specific.

## Production

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
```

The prod stage builds the frontend and serves `dist/` through Caddy, which also handles the
SPA fallback and proxies `/api`. The API entrypoint applies migrations before starting
gunicorn.

## Where the rest is written down

- [`BACKLOG.md`](BACKLOG.md) — open work, with the reasoning behind each item
- [`AUDYT.md`](AUDYT.md) — the technical audit and the state of its findings
- [`client/CLAUDE.md`](client/CLAUDE.md) — frontend stack, commands and conventions
