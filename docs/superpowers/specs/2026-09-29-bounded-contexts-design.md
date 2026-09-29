# Split `knowledge_management` into four bounded contexts

## Goal

`app/knowledge_management/` holds four separate concerns in one context. Split it into
`documents`, `retrieval`, `conversations` and `summaries`, each keeping the
domain / application / infrastructure / ui layers. `identity` stays as it is.

Success: `uv run pytest` and `./scripts/lint.sh` pass, and no module imports another
context's internals — only its `api.py`.

## Non-goals

- No change to HTTP endpoints, request/response shapes, DB schema, migrations, config or
  the frontend.
- No CQRS, command/query bus or domain events. Contexts call each other synchronously.
- No boundary-enforcement tooling (e.g. `import-linter`).
- No change to test assertions — only imports and use-case constructors.

## Dependency direction

```
documents     → retrieval
conversations → retrieval
summaries     → documents
retrieval     → (nothing but shared)
```

Acyclic. Every context may import `app.shared` and `app.identity` (for `get_current_user`,
`User`), as routers do today.

## Layout

```
app/
  identity/                     unchanged
  shared/
    kernel/document.py          Document, namespaced_document_id, chunk_id, citation_label
                                (today: domain/models.Document + domain/document_identity.py)
    llm/                        llm_factory, api_keys, spotlighting
    translations_router.py      today: knowledge_management/ui/api/routers/translations.py
    (dependencies.py removed — split per context)
  documents/
    domain/repositories.py      DocumentRepo
    application/                upload_document, delete_document
    infrastructure/             postgres_document_repo, pymupdf_loader, factory
    ui/router.py                today: routers/documents.py
    dependencies.py
    api.py
  retrieval/
    domain/                     models (Answer, Entity, Relation, GraphFragment),
                                repositories (VectorStoreRepo, KnowledgeGraphRepo,
                                EntityExtractor, RAGService, RerankerService, AnswerJudge),
                                entity_normalization, evaluation, null_entity_extractor,
                                null_knowledge_graph_repo
    application/                ask_question, index_document, remove_from_index,
                                context_retriever, candidate_retrieval, rank_fusion,
                                evaluation/
    infrastructure/             vector stores (postgres, faiss, neo4j), neo4j graph, lucene,
                                text_chunker, embeddings_factory, langchain_rag_service,
                                reranker + factory, entity_extractor,
                                answer_judge + factory, factory
    ui/                         router.py (today: routers/qa.py), sources.py
    dependencies.py
    api.py
  conversations/
    domain/                     Conversation, ChatMessage, ConversationRepo
    application/                chat_with_docs, manage_conversations
    infrastructure/             postgres_conversation_repo, factory
    ui/router.py                today: routers/chat.py
    dependencies.py
  summaries/
    domain/                     Summary, SummaryRepo, SummarizerService
    application/                summarize_docs, delete_summary
    infrastructure/             postgres_summary_repo, langchain_summarizer, factory
    ui/router.py                today: routers/summarize.py
    dependencies.py
```

`app/knowledge_management/` is removed. Today's single
`infrastructure/persistence/factory.py` is split into one `factory.py` per context holding
that context's `create_*` functions; `_neo4j_credentials` and `_unsupported` go to
`retrieval/infrastructure/factory.py` (only Neo4j and provider switches live there). If
`_unsupported` is also needed by another factory, it moves to `app/shared/`.

`Document` goes to the shared kernel because `documents` produces it and `retrieval`
consumes it; owning it in either would create a cycle, since `documents` already depends
on `retrieval`.

## Public API of each context

`api.py` only re-exports existing names — no forwarding wrappers.

`retrieval/api.py`:
- `IndexDocumentUseCase`, `get_index_document_use_case`
- `RemoveFromIndexUseCase`, `get_remove_from_index_use_case`
- `ContextRetriever`, `get_context_retriever`
- `RAGService`, `get_rag_service`
- `Answer`

`documents/api.py`:
- `DocumentRepo`, `get_doc_repo`

`conversations` and `summaries` have no `api.py` — nothing depends on them.

## New application classes in `retrieval`

Extracted from existing code, behavior unchanged:

- `IndexDocumentUseCase(vector_repo, graph_repo, entity_extractor)`
  `.execute(document, owner_id, pages) -> None` — builds the per-page vector
  documents (or the whole document), adds them to the vector store, extracts the graph
  fragment from the whole document and adds it. Today: the second half of
  `UploadDocumentUseCase.execute`.
- `RemoveFromIndexUseCase(vector_repo, graph_repo)`
  `.execute(doc_id, owner_id) -> None` — deletes vectors, then graph facts. Today: the
  middle of `DeleteDocumentUseCase.execute`.
- `ContextRetriever(candidate_retriever, reranker, candidate_count, top_k)`
  `.find(query, owner_id) -> list[Document]` — candidate retrieval then rerank to `top_k`.
  Today duplicated in `AskQuestionUseCase.execute` and `ChatWithDocsUseCase._prepare_context`.
  `CandidateRetriever` stays as is; the evaluation pipeline keeps using it directly.

## Flows after the change

- **Upload** — `UploadDocumentUseCase(doc_repo, index_document)`: mint the namespaced id,
  save the record, `index_document.execute(...)`. Same order and error propagation as today.
- **Delete** — `DeleteDocumentUseCase(doc_repo, remove_from_index)`: look up (owner-filtered),
  remove the file if within storage, `remove_from_index.execute(...)`, delete the record.
  Same order as today.
- **Ask** — `AskQuestionUseCase(context_retriever, rag_service)`: `find`, then
  `answer_question`.
- **Chat** — `ChatWithDocsUseCase(conversation_repo, context_retriever, rag_service)`:
  load/create conversation, condense the question when there is history, `find`, answer or
  stream, persist the turn. Condensing stays here: it is about conversation history.
- **Summarize** — `SummarizeDocsUseCase(doc_repo, summarizer, summary_repo)` with `doc_repo`
  from `documents.api`.

## Wiring

- Each context's `dependencies.py` holds its repository singletons (same lock pattern as
  today) and its FastAPI providers, plus `init()` and `async shutdown()` for its own
  singletons.
- `main.py` lifespan calls `init()` of `documents`, `retrieval`, `conversations`,
  `summaries`, and on shutdown `shutdown()` of each, then `dispose_engine()` as today.
  Only `retrieval.shutdown()` closes drivers (vector store, graph).
- `main.py` includes the four context routers, `auth.router` and the translations router,
  with the same prefixes.

## Tests

- Tests stay flat in `tests/`; `tests/contracts/` stays as is.
- Imports updated everywhere, including `tests/fakes.py` and dependency overrides in
  `conftest.py` / API tests (overrides now target the providers in the per-context
  `dependencies.py`).
- `test_upload_document`, `test_delete_document`, `test_ask_question`,
  `test_chat_with_docs`: construct the use cases through `IndexDocumentUseCase`,
  `RemoveFromIndexUseCase` and `ContextRetriever` built from the same fakes. Assertions
  unchanged.
- `test_lifespan`: updated to the per-context `init` / `shutdown`.

## Docs

README: the Layout section and the evaluation command
(`python -m app.retrieval.application.evaluation.run_evaluation`).

## Done when

1. `uv run pytest` passes.
2. `./scripts/lint.sh` passes (ruff + mypy).
3. For each of the four contexts `X`, no file outside `app/X/` imports `app.X.` other than
   `app.X.api` — except `main.py`, which imports routers and `dependencies` for wiring.
