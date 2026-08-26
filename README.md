# docs-analysis

A document analysis (RAG) application: upload PDFs and the system chunks them, indexes the
chunks in a vector store, and lets you ask questions in natural language, chat with your
documents, and generate summaries and translations. The backend is FastAPI (Python) with
PostgreSQL + pgvector and an optional Neo4j knowledge graph; the frontend is React + Vite.

## Getting started

```bash
./scripts/setup.sh          # --build forces an image rebuild
```

The script creates `.env` from `.env.example` (fill in the API keys and passwords), installs
backend (`uv`) and frontend (`yarn`) dependencies, starts the containers, applies the Alembic
migrations, and runs the linters and tests.

## Default ports

| Service | Host port | Notes |
|---|---|---|
| Frontend (Vite / Caddy) | `3000` | http://localhost:3000, proxies `/api` to the API |
| API (FastAPI) | `8001` | `8000` inside the container; docs at `/docs` |
| PostgreSQL (pgvector) | `127.0.0.1:5433` | `5432` inside the container; dev only (override), not exposed in prod |
| Neo4j – Bolt | `127.0.0.1:7687` | dev only (override) |
| Neo4j – Browser | `127.0.0.1:7474` | dev only (override) |
