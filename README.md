# docs-analysis

RAG over your own documents: upload PDFs and text files, then ask questions, chat or
summarize. Retrieval combines vector search, full-text search and an optional knowledge
graph. All data is scoped per user.

## Stack

| Area | Choice |
|---|---|
| API | FastAPI, Python 3.12, `uv` |
| Records | Postgres (SQLAlchemy async + psycopg3), Alembic |
| Vectors | pgvector, FAISS or Neo4j |
| Knowledge graph | Neo4j or `none` |
| LLM | OpenAI, Anthropic, Google or Ollama (LangChain) |
| Reranking | none, LLM, Cohere or local BGE |
| Frontend | React 19 + Vite + TypeScript ([`client/CLAUDE.md`](client/CLAUDE.md)) |

## Quick start

Requires Docker and [`uv`](https://docs.astral.sh/uv/).

```bash
cp .env.example .env      # fill in the LLM key and passwords
./scripts/setup.sh        # deps, containers, migrations, lint, tests
```

- API docs → http://localhost:8001/docs
- Frontend → http://localhost:3000

All settings are required at startup, except keys for providers you don't use.

## Commands

```bash
docker compose up -d
docker compose up -d --force-recreate api  # after `uv add` (.venv is a named volume)
uv run pytest                              # unit tests
uv run pytest -m integration               # needs POSTGRES_HOST=localhost POSTGRES_PORT=5433 NEO4J_URI=bolt://localhost:7687
./scripts/lint.sh                          # ruff + mypy
cd client && yarn test                     # frontend
```

## Layout

Bounded contexts (domain / application / infrastructure / ui); cross-context imports only via `api.py`.

```
app/
  identity/        users, auth, JWT
  documents/       upload, listing, deletion
  retrieval/       vector store, knowledge graph, rerank, answers, evaluation
  conversations/   chat history
  summaries/       document summaries
  shared/          config, database, storage, llm
tests/contracts/   one suite per port, run against every adapter
```

## Knowledge graph

Off by default (`KNOWLEDGE_GRAPH_PROVIDER=none`). Enabling it adds an LLM call per upload;
set it **before** loading documents, since entities are extracted at upload time.

## Evaluation

```bash
docker compose exec api uv run python -m \
  app.retrieval.application.evaluation.run_evaluation \
  --dataset eval/golden_set.json --owner-id <user_id> --compare-graph
```

Template: `eval/golden_set.template.json`.

## Production

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
```
