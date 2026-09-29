# docs-analysis — backend guide

Read this at the start of every session and stick to it. If something here is unclear or
contradicts the code — ask instead of guessing. Frontend rules live in
[`client/CLAUDE.md`](client/CLAUDE.md).

## Goal

Question answering over the user's own documents. The system finds the most relevant
fragments and answers **only from those fragments**, citing sources. A question the documents
do not cover gets an explicit "not found in the documents". All data is scoped per user
(`owner_id`).

## Architecture

Two paths joined by the vector store (and optionally the knowledge graph).

**Indexing** — on upload (`POST /documents/upload`):

```
router → loader (PDF: PyMuPDF, one Document per page; text: UTF-8) → documents repo (Postgres)
       → IndexDocumentUseCase → chunker → embeddings → vector store
                              → entity extractor → knowledge graph   (only if enabled)
```

**Query** — `POST /qa/ask`, `POST /chat` (+ `/chat/stream`):

```
question → RetrievalPipeline (vector search + graph facts → RRF fusion → reranker → top_k)
         → RAGService (LLM) → answer + sources
```

**Critical rule:** documents and questions must be embedded with the same model. The
embedding model follows `LLM_PROVIDER` (`retrieval/embedder.py`); changing it makes the
existing index incompatible — re-upload the documents.

## Layout

```
app/
  identity/        models, repo, security, service, router, dependencies
  documents/       repo, loader, service, router, dependencies, api
  conversations/   models, repo, chat, router, dependencies
  summaries/       models, repo, summarizer, service, router, dependencies
  retrieval/       models, ports, factory, pipeline, indexing, rag_service, reranker,
                   embedder, chunker, rank_fusion, answer_judge, router, dependencies, api
    vector_store/    postgres, faiss, neo4j
    knowledge_graph/ neo4j, null, entity_extractor, normalization
    evaluation/      metrics, dataset, run_evaluation (CLI)
  shared/          config, database, storage, logging, llm, kernel (Document, ids)
tests/             test_<topic>.py, fakes.py, conftest.py
tests/contracts/   one suite per port, run against every adapter
```

## Structure rules

Keep it flat; no layer for its own sake.

- **One package per bounded context, flat modules inside.** No `domain/application/
  infrastructure/ui` subpackages. A module over ~300 lines becomes a package
  (as `vector_store/`, `knowledge_graph/`).
- **`Protocol` only with ≥ 2 implementations** — test fakes in `tests/fakes.py` count.
  Otherwise use the concrete class. Retrieval adapters subclass their port explicitly to
  inherit the default `close()`.
- **Provider choice only where alternatives really exist:** `retrieval/factory.py`,
  `create_reranker`, `create_embeddings`, `create_answer_judge`, `LLMFactory`. No factory for
  a single implementation.
- **No use case that only forwards one call** — the router calls the repo directly. Logic
  goes into a function in `service.py`. A class only when it holds collaborators across
  several entry points (`ChatWithDocsUseCase`, `IndexDocumentUseCase`, `RetrievalPipeline`).
- **Postgres repos are created per request** (`get_*_repo()` returns a new instance; they
  borrow the shared pool). Process singletons with `init()/shutdown()` in the lifespan only
  for adapters that own a driver (`retrieval/dependencies.py`).
- **Cross-context imports only via `api.py`** (`documents`, `retrieval`) plus the auth guard
  `identity.dependencies.get_current_user` and `identity.models.User`.

## Conventions

- Python 3.12, `uv`, type hints everywhere; `ruff` + `mypy --strict` (`./scripts/lint.sh`).
- Code, comments, docstrings and README in English. LLM prompts are in Polish.
- Settings only from `.env` via `app/shared/config.py`; all required at startup except keys
  for unused providers. Secrets never in code or committed files.
- Logging through `logging` (`app/shared/logging.py`), never `print` in library code
  (CLI output in `run_evaluation` is the exception).
- Tests: `pytest`, no network, no real API keys. Stubs from `tests/fakes.py`, dependency
  overrides via the `override_dependency` fixture. Tests needing a live store are marked
  `integration` and excluded by default.
- Do not add dependencies without asking.

## Commands

```bash
./scripts/setup.sh                 # deps, containers, migrations, lint, tests
uv run pytest                      # unit tests
uv run pytest -m integration       # contracts; needs `docker compose up -d` and
                                   # POSTGRES_HOST=localhost POSTGRES_PORT=5433 NEO4J_URI=bolt://localhost:7687
./scripts/lint.sh                  # ruff check + ruff format --check + mypy
uv run python -m app.retrieval.evaluation.run_evaluation --dataset eval/golden_set.json --owner-id <id>
```

## Working rules

1. Work in **phases**, one at a time.
2. Before coding, present a short plan of the phase. After it, run `uv run pytest` and
   `./scripts/lint.sh`, then summarize: what was done, how to run it, what is uncertain.
   **Stop and wait for approval.**
3. After each phase update the README and the "Status" section below.
4. Never commit `.env`, `storage/` or logs.

## Status

Flattening the bounded contexts:

- [x] Phase 1 — `identity`, `summaries`
- [x] Phase 2 — `documents`, `conversations`, drop `PersistenceProvider`
- [x] Phase 3 — `retrieval`
- [x] Phase 4 — README layout, this file

From the original RAG spec — decide before starting the open ones:

- [x] `score` (vector similarity, `null` for keyword-only hits and graph facts) and `snippet`
  per source in `/qa/ask` responses; `/chat` still returns `list[str]` labels
- [x] `/health` with `chunks_indexed` (all owners, `VectorStoreRepo.count()`)
- [x] `min_score` cutoff on retrieval (`RETRIEVAL_MIN_SCORE`, applied before reranking;
  hits without a score pass) — value still to be tuned with the evaluation
- [ ] store the embedding model with the index and fail clearly on a mismatch at query time
