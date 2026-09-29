# Bounded Contexts Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split `app/knowledge_management/` into four bounded contexts — `documents`, `retrieval`, `conversations`, `summaries` — that talk to each other only through each context's `api.py`.

**Architecture:** Task 1 moves every module to its new home with a throwaway script that rewrites imports (behavior unchanged). Tasks 2–3 extract the cross-context seams (`IndexDocumentUseCase`, `RemoveFromIndexUseCase`, reuse of `RetrievalPipeline`). Task 4 splits dependency injection per context and removes `app/shared/dependencies.py`. Task 5 updates the README and verifies the boundaries.

**Tech Stack:** Python 3.12, FastAPI, pytest (asyncio auto), ruff, mypy strict, `uv`.

**Spec:** `docs/superpowers/specs/2026-09-29-bounded-contexts-design.md`

## Global Constraints

- No change to HTTP endpoints, request/response shapes, DB schema, migrations, config or the frontend.
- No CQRS, command/query bus or domain events; contexts call each other synchronously.
- No boundary-enforcement tooling (`import-linter` etc.).
- Test assertions do not change — only imports and use-case constructors.
- Dependency direction: `documents → retrieval`, `conversations → retrieval`, `summaries → documents`; `retrieval` depends on nothing but `shared`.
- Outside `app/X/`, code imports `app.X.*` only via `app.X.api` — except `app/main.py` (routers + `dependencies` for wiring). Tests may import internals.
- After every task: `uv run pytest` and `./scripts/lint.sh` pass.
- Do not touch the user's uncommitted change in `README.md` except where Task 5 edits it; stage files explicitly, never `git add -A` at repo root.

## Review Focus

1. Upload with the knowledge graph on: the entity extractor must receive the **whole** document, not the per-page split → Task 2 `test_index_extracts_graph_facts_from_the_whole_document_not_the_pages`.
2. Delete with the knowledge graph on: graph facts are retracted after vectors → Task 2 `test_remove_from_index_retracts_vectors_then_graph_facts`.
3. App shutdown with Neo4j: the vector store and graph drivers are still closed and singletons cleared → Task 4 updated `test_lifespan.py`.
4. Import cycles at startup (`documents.api → retrieval.api`, `summaries → documents.api`) → Task 4 step `python -c "import app.main"`.
5. Evaluation CLI at its new module path still starts → Task 5 step running `--help`.

---

## File Map

| Today (`app/knowledge_management/…`) | After |
|---|---|
| `domain/models.py` | split: `Document` → `app/shared/kernel/document.py`; `Answer, Entity, Relation, GraphFragment` → `app/retrieval/domain/models.py`; `Summary` → `app/summaries/domain/models.py`; `ChatMessage, Conversation` → `app/conversations/domain/models.py` |
| `domain/repositories.py` | split: `DocumentRepo` → `app/documents/domain/repositories.py`; `VectorStoreRepo, RAGService, RerankerService, AnswerJudge, KnowledgeGraphRepo, EntityExtractor` → `app/retrieval/domain/repositories.py`; `SummarizerService, SummaryRepo` → `app/summaries/domain/repositories.py`; `ConversationRepo` → `app/conversations/domain/repositories.py` |
| `domain/document_identity.py` | `app/shared/kernel/document_identity.py` |
| `domain/{entity_normalization,evaluation,null_entity_extractor,null_knowledge_graph_repo}.py` | `app/retrieval/domain/` |
| `application/evaluation/` (all but `retrieval_pipeline.py`) | `app/retrieval/application/evaluation/` |
| `application/evaluation/retrieval_pipeline.py` | `app/retrieval/application/retrieval_pipeline.py` |
| `application/retrieval/{candidate_retrieval,rank_fusion}.py` | `app/retrieval/application/` |
| `application/use_cases/ask_question.py` | `app/retrieval/application/` |
| `application/use_cases/{upload_document,delete_document}.py` | `app/documents/application/` |
| `application/use_cases/{chat_with_docs,manage_conversations}.py` | `app/conversations/application/` |
| `application/use_cases/{summarize_docs,delete_summary}.py` | `app/summaries/application/` |
| `infrastructure/llm/{llm_factory,api_keys,spotlighting}.py` | `app/shared/llm/` |
| `infrastructure/llm/langchain_summarizer.py` | `app/summaries/infrastructure/` |
| `infrastructure/llm/*` (rest) | `app/retrieval/infrastructure/` |
| `infrastructure/pdf/pymupdf_loader.py` | `app/documents/infrastructure/` |
| `infrastructure/persistence/postgres_{document,conversation,summary}_repo.py` | each to its context's `infrastructure/` |
| `infrastructure/persistence/*` (vector stores, graph, lucene) | `app/retrieval/infrastructure/` |
| `infrastructure/persistence/factory.py` | `app/retrieval/infrastructure/factory.py` + new small `factory.py` in documents/conversations/summaries |
| `infrastructure/text/text_chunker.py` | `app/retrieval/infrastructure/` |
| `ui/api/routers/{documents,qa,chat,summarize}.py` | `app/{documents,retrieval,conversations,summaries}/ui/router.py` |
| `ui/api/routers/translations.py` | `app/shared/translations_router.py` |
| `ui/api/sources.py` | `app/retrieval/ui/sources.py` |
| `app/shared/dependencies.py` | removed in Task 4 → `app/{documents,retrieval,conversations,summaries}/dependencies.py` |
| — | new `app/retrieval/api.py`, `app/documents/api.py` |

---

### Task 1: Move modules into the four contexts (no behavior change)

**Files:**
- Create (scratch, not committed): `$SCRATCH/relayout.py` where `$SCRATCH` is `/private/tmp/claude-501/-Users-lsprezak-Repo-python-docs-analysis/9ce2b938-e623-4185-adc3-af1413edf65e/scratchpad`
- Move/split: everything in the File Map except `app/shared/dependencies.py`
- Create: `app/documents/infrastructure/factory.py`, `app/conversations/infrastructure/factory.py`, `app/summaries/infrastructure/factory.py`
- Modify: `app/retrieval/infrastructure/factory.py` (after move), `app/shared/dependencies.py`, `app/main.py`
- Test: whole existing suite (imports rewritten by the script)

**Interfaces:**
- Produces: every module path in the File Map "After" column. Later tasks import these exact paths.

- [ ] **Step 1: Write the relayout script**

Save as `$SCRATCH/relayout.py`:

```python
"""Throwaway: rewrites imports to the new layout and (phase `km`) moves the files.

Run from the repository root: `python relayout.py km` or `python relayout.py deps`.
Every import is rewritten to an absolute path; ruff sorts them afterwards.
"""

import ast
import shutil
import subprocess
import sys
from pathlib import Path

KM = "app.knowledge_management"

KM_MODULES = {
    f"{KM}.domain.document_identity": "app.shared.kernel.document_identity",
    f"{KM}.domain.entity_normalization": "app.retrieval.domain.entity_normalization",
    f"{KM}.domain.evaluation": "app.retrieval.domain.evaluation",
    f"{KM}.domain.null_entity_extractor": "app.retrieval.domain.null_entity_extractor",
    f"{KM}.domain.null_knowledge_graph_repo": "app.retrieval.domain.null_knowledge_graph_repo",
    f"{KM}.application.evaluation": "app.retrieval.application.evaluation",
    f"{KM}.application.evaluation.dataset": "app.retrieval.application.evaluation.dataset",
    f"{KM}.application.evaluation.evaluate_generation": "app.retrieval.application.evaluation.evaluate_generation",
    f"{KM}.application.evaluation.evaluate_retrieval": "app.retrieval.application.evaluation.evaluate_retrieval",
    f"{KM}.application.evaluation.retrieval_metrics": "app.retrieval.application.evaluation.retrieval_metrics",
    f"{KM}.application.evaluation.run_evaluation": "app.retrieval.application.evaluation.run_evaluation",
    f"{KM}.application.evaluation.retrieval_pipeline": "app.retrieval.application.retrieval_pipeline",
    f"{KM}.application.retrieval.candidate_retrieval": "app.retrieval.application.candidate_retrieval",
    f"{KM}.application.retrieval.rank_fusion": "app.retrieval.application.rank_fusion",
    f"{KM}.application.use_cases.ask_question": "app.retrieval.application.ask_question",
    f"{KM}.application.use_cases.upload_document": "app.documents.application.upload_document",
    f"{KM}.application.use_cases.delete_document": "app.documents.application.delete_document",
    f"{KM}.application.use_cases.chat_with_docs": "app.conversations.application.chat_with_docs",
    f"{KM}.application.use_cases.manage_conversations": "app.conversations.application.manage_conversations",
    f"{KM}.application.use_cases.summarize_docs": "app.summaries.application.summarize_docs",
    f"{KM}.application.use_cases.delete_summary": "app.summaries.application.delete_summary",
    f"{KM}.infrastructure.llm.llm_factory": "app.shared.llm.llm_factory",
    f"{KM}.infrastructure.llm.api_keys": "app.shared.llm.api_keys",
    f"{KM}.infrastructure.llm.spotlighting": "app.shared.llm.spotlighting",
    f"{KM}.infrastructure.llm.langchain_summarizer": "app.summaries.infrastructure.langchain_summarizer",
    f"{KM}.infrastructure.llm.answer_judge": "app.retrieval.infrastructure.answer_judge",
    f"{KM}.infrastructure.llm.answer_judge_factory": "app.retrieval.infrastructure.answer_judge_factory",
    f"{KM}.infrastructure.llm.embeddings_factory": "app.retrieval.infrastructure.embeddings_factory",
    f"{KM}.infrastructure.llm.entity_extractor": "app.retrieval.infrastructure.entity_extractor",
    f"{KM}.infrastructure.llm.langchain_rag_service": "app.retrieval.infrastructure.langchain_rag_service",
    f"{KM}.infrastructure.llm.reranker": "app.retrieval.infrastructure.reranker",
    f"{KM}.infrastructure.llm.reranker_factory": "app.retrieval.infrastructure.reranker_factory",
    f"{KM}.infrastructure.pdf.pymupdf_loader": "app.documents.infrastructure.pymupdf_loader",
    f"{KM}.infrastructure.persistence.factory": "app.retrieval.infrastructure.factory",
    f"{KM}.infrastructure.persistence.faiss_vectorstore_repo": "app.retrieval.infrastructure.faiss_vectorstore_repo",
    f"{KM}.infrastructure.persistence.lucene": "app.retrieval.infrastructure.lucene",
    f"{KM}.infrastructure.persistence.neo4j_knowledge_graph_repo": "app.retrieval.infrastructure.neo4j_knowledge_graph_repo",
    f"{KM}.infrastructure.persistence.neo4j_vectorstore_repo": "app.retrieval.infrastructure.neo4j_vectorstore_repo",
    f"{KM}.infrastructure.persistence.postgres_vectorstore_repo": "app.retrieval.infrastructure.postgres_vectorstore_repo",
    f"{KM}.infrastructure.persistence.postgres_document_repo": "app.documents.infrastructure.postgres_document_repo",
    f"{KM}.infrastructure.persistence.postgres_conversation_repo": "app.conversations.infrastructure.postgres_conversation_repo",
    f"{KM}.infrastructure.persistence.postgres_summary_repo": "app.summaries.infrastructure.postgres_summary_repo",
    f"{KM}.infrastructure.text.text_chunker": "app.retrieval.infrastructure.text_chunker",
    f"{KM}.ui.api.routers.documents": "app.documents.ui.router",
    f"{KM}.ui.api.routers.qa": "app.retrieval.ui.router",
    f"{KM}.ui.api.routers.chat": "app.conversations.ui.router",
    f"{KM}.ui.api.routers.summarize": "app.summaries.ui.router",
    f"{KM}.ui.api.routers.translations": "app.shared.translations_router",
    f"{KM}.ui.api.sources": "app.retrieval.ui.sources",
}

KM_NAMES = {
    f"{KM}.domain.models": {
        "Document": "app.shared.kernel.document",
        "Answer": "app.retrieval.domain.models",
        "Entity": "app.retrieval.domain.models",
        "Relation": "app.retrieval.domain.models",
        "GraphFragment": "app.retrieval.domain.models",
        "Summary": "app.summaries.domain.models",
        "ChatMessage": "app.conversations.domain.models",
        "Conversation": "app.conversations.domain.models",
    },
    f"{KM}.domain.repositories": {
        "DocumentRepo": "app.documents.domain.repositories",
        "VectorStoreRepo": "app.retrieval.domain.repositories",
        "RAGService": "app.retrieval.domain.repositories",
        "RerankerService": "app.retrieval.domain.repositories",
        "AnswerJudge": "app.retrieval.domain.repositories",
        "KnowledgeGraphRepo": "app.retrieval.domain.repositories",
        "EntityExtractor": "app.retrieval.domain.repositories",
        "SummarizerService": "app.summaries.domain.repositories",
        "SummaryRepo": "app.summaries.domain.repositories",
        "ConversationRepo": "app.conversations.domain.repositories",
    },
}

ABC_HEADER = "from abc import ABC, abstractmethod\n"
KM_SPLITS = {
    "app/knowledge_management/domain/models.py": {
        "app/shared/kernel/document.py": (
            "from dataclasses import dataclass\nfrom typing import Any\n\n\n",
            ["Document"],
        ),
        "app/retrieval/domain/models.py": (
            "from dataclasses import dataclass\n\n"
            "from app.shared.kernel.document import Document\n\n\n",
            ["Answer", "Entity", "Relation", "GraphFragment"],
        ),
        "app/summaries/domain/models.py": ("from dataclasses import dataclass\n\n\n", ["Summary"]),
        "app/conversations/domain/models.py": (
            "from dataclasses import dataclass\n\n\n",
            ["ChatMessage", "Conversation"],
        ),
    },
    "app/knowledge_management/domain/repositories.py": {
        "app/documents/domain/repositories.py": (
            ABC_HEADER + "\nfrom app.shared.kernel.document import Document\n\n\n",
            ["DocumentRepo"],
        ),
        "app/retrieval/domain/repositories.py": (
            ABC_HEADER
            + "from collections.abc import AsyncIterator\n\n"
            + "from app.retrieval.domain.models import GraphFragment\n"
            + "from app.shared.kernel.document import Document\n\n\n",
            [
                "VectorStoreRepo",
                "RAGService",
                "RerankerService",
                "AnswerJudge",
                "KnowledgeGraphRepo",
                "EntityExtractor",
            ],
        ),
        "app/summaries/domain/repositories.py": (
            ABC_HEADER
            + "\nfrom app.shared.kernel.document import Document\n"
            + "from app.summaries.domain.models import Summary\n\n\n",
            ["SummarizerService", "SummaryRepo"],
        ),
        "app/conversations/domain/repositories.py": (
            ABC_HEADER + "\nfrom app.conversations.domain.models import Conversation\n\n\n",
            ["ConversationRepo"],
        ),
    },
}

DEPS_NAMES = {
    "app.shared.dependencies": {
        **dict.fromkeys(
            ["get_doc_repo", "get_upload_document_use_case", "get_delete_document_use_case"],
            "app.documents.dependencies",
        ),
        **dict.fromkeys(
            [
                "get_vector_repo",
                "get_graph_repo",
                "get_entity_extractor",
                "get_rag_service",
                "get_reranker_service",
                "get_ask_question_use_case",
            ],
            "app.retrieval.dependencies",
        ),
        **dict.fromkeys(
            [
                "get_conversation_repo",
                "get_chat_with_docs_use_case",
                "get_list_conversations_use_case",
                "get_get_conversation_use_case",
                "get_delete_conversation_use_case",
            ],
            "app.conversations.dependencies",
        ),
        **dict.fromkeys(
            [
                "get_summary_repo",
                "get_summarizer",
                "get_summarize_docs_use_case",
                "get_delete_summary_use_case",
            ],
            "app.summaries.dependencies",
        ),
    }
}

PHASES = {
    "km": (KM_MODULES, KM_NAMES, KM),
    "deps": ({}, DEPS_NAMES, "app.shared.dependencies"),
}


def resolve(node: ast.ImportFrom, file: Path) -> str:
    if node.level == 0:
        return node.module or ""
    package = list(file.with_suffix("").parts[:-1])
    base = package[: len(package) - (node.level - 1)]
    return ".".join(base + ([node.module] if node.module else []))


def render(alias: ast.alias) -> str:
    return f"{alias.name} as {alias.asname}" if alias.asname else alias.name


def rewrite(file: Path, modules: dict, names: dict, prefix: str, flagged: list) -> str | None:
    source = file.read_text()
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    edits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(a.name == prefix or a.name.startswith(prefix + ".") for a in node.names):
                raise SystemExit(f"plain import to handle by hand: {file}:{node.lineno}")
            continue
        if not isinstance(node, ast.ImportFrom):
            continue
        module = resolve(node, file)
        indent = " " * node.col_offset
        if module in names:
            groups: dict[str, list[str]] = {}
            for alias in node.names:
                groups.setdefault(names[module][alias.name], []).append(render(alias))
            text = "".join(
                f"{indent}from {target} import {', '.join(items)}\n"
                for target, items in groups.items()
            )
        elif module in modules:
            text = f"{indent}from {modules[module]} import {', '.join(map(render, node.names))}\n"
        elif any(f"{module}.{a.name}" in modules or f"{module}.{a.name}" in names for a in node.names):
            flagged.append(f"{file}:{node.lineno}")
            continue
        elif module == prefix or module.startswith(prefix + "."):
            raise SystemExit(f"unmapped module {module}: {file}:{node.lineno}")
        else:
            continue
        edits.append((node.lineno - 1, node.end_lineno, text))
    if not edits:
        return None
    for start, end, text in sorted(edits, reverse=True):
        lines[start:end] = [text]
    return "".join(lines)


def split(source_file: str, targets: dict) -> None:
    source = Path(source_file).read_text()
    lines = source.splitlines(keepends=True)
    segments = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.ClassDef):
            start = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
            segments[node.name] = "".join(lines[start : node.end_lineno])
    for target, (header, class_names) in targets.items():
        path = Path(target)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(header + "\n\n".join(segments.pop(name) for name in class_names))
    assert not segments, f"classes left behind in {source_file}: {list(segments)}"
    subprocess.run(["git", "rm", "-q", source_file], check=True)


def module_file(module: str) -> Path:
    base = Path(*module.split("."))
    return base / "__init__.py" if base.is_dir() else base.with_suffix(".py")


def move(modules: dict) -> None:
    sources = {old: module_file(old) for old in modules}
    for old, new in modules.items():
        source = sources[old]
        base = Path(*new.split("."))
        target = base / "__init__.py" if source.name == "__init__.py" else base.with_suffix(".py")
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "mv", str(source), str(target)], check=True)


def remove_old_package() -> None:
    old = Path("app/knowledge_management")
    leftovers = [p for p in old.rglob("*.py") if p.name != "__init__.py"]
    assert not leftovers, f"not moved: {leftovers}"
    subprocess.run(["git", "rm", "-rq", str(old)], check=True)
    shutil.rmtree(old, ignore_errors=True)


def add_missing_inits() -> None:
    for file in list(Path("app").rglob("*.py")):
        for directory in file.parents:
            if directory == Path("."):
                break
            init = directory / "__init__.py"
            if not init.exists():
                init.touch()
                print(f"created {init}")


def main() -> None:
    phase = sys.argv[1]
    modules, names, prefix = PHASES[phase]
    flagged: list[str] = []
    files = sorted([*Path("app").rglob("*.py"), *Path("tests").rglob("*.py")])
    rewritten = {f: text for f in files if (text := rewrite(f, modules, names, prefix, flagged))}
    for file, text in rewritten.items():
        file.write_text(text)
    print(f"rewrote imports in {len(rewritten)} files")
    if phase == "km":
        for source_file, targets in KM_SPLITS.items():
            split(source_file, targets)
        move(modules)
        remove_old_package()
        add_missing_inits()
    print("FIX BY HAND:" if flagged else "nothing flagged", *flagged, sep="\n  ")


main()
```

- [ ] **Step 2: Run it**

```bash
cd /Users/lsprezak/Repo/python/docs-analysis && python3 "$SCRATCH/relayout.py" km
```

Expected: `rewrote imports in N files`, `created …/__init__.py` lines for the new packages, and exactly two flagged locations:
- `app/main.py:<line>` (`from app.knowledge_management.ui.api.routers import (...)`)
- `app/shared/dependencies.py:<line>` (`from app.knowledge_management.infrastructure.persistence import factory`)

If it stops with `unmapped module …`, nothing was written yet: add the module to `KM_MODULES` per the File Map and rerun. Any other flagged line is fixed by hand the same way as the two below.

- [ ] **Step 3: Fix `app/main.py` routers by hand**

Replace

```python
from app.knowledge_management.ui.api.routers import (
    chat,
    documents,
    qa,
    summarize,
    translations,
)
```

with

```python
from app.conversations.ui import router as conversations_router
from app.documents.ui import router as documents_router
from app.retrieval.ui import router as retrieval_router
from app.shared import translations_router
from app.summaries.ui import router as summaries_router
```

and the router registrations (same order as before) with

```python
app.include_router(auth.router, prefix=settings.API_V1_STR)
app.include_router(documents_router.router, prefix=settings.API_V1_STR)
app.include_router(summaries_router.router, prefix=settings.API_V1_STR)
app.include_router(conversations_router.router, prefix=settings.API_V1_STR)
app.include_router(retrieval_router.router, prefix=settings.API_V1_STR)
app.include_router(translations_router.router, prefix=settings.API_V1_STR)
```

- [ ] **Step 4: Split the persistence factory**

Create `app/documents/infrastructure/factory.py`:

```python
from app.documents.domain.repositories import DocumentRepo
from app.documents.infrastructure.postgres_document_repo import PostgresDocumentRepo
from app.shared.config import settings
from app.shared.enums import PersistenceProvider


def create_document_repo() -> DocumentRepo:
    if settings.PERSISTENCE_PROVIDER == PersistenceProvider.POSTGRES:
        return PostgresDocumentRepo()
    raise NotImplementedError(
        f"No DocumentRepo adapter for provider '{settings.PERSISTENCE_PROVIDER}'"
    )
```

Create `app/conversations/infrastructure/factory.py`:

```python
from app.conversations.domain.repositories import ConversationRepo
from app.conversations.infrastructure.postgres_conversation_repo import PostgresConversationRepo
from app.shared.config import settings
from app.shared.enums import PersistenceProvider


def create_conversation_repo() -> ConversationRepo:
    if settings.PERSISTENCE_PROVIDER == PersistenceProvider.POSTGRES:
        return PostgresConversationRepo()
    raise NotImplementedError(
        f"No ConversationRepo adapter for provider '{settings.PERSISTENCE_PROVIDER}'"
    )
```

Create `app/summaries/infrastructure/factory.py`:

```python
from app.shared.config import settings
from app.shared.enums import PersistenceProvider
from app.summaries.domain.repositories import SummaryRepo
from app.summaries.infrastructure.postgres_summary_repo import PostgresSummaryRepo


def create_summary_repo() -> SummaryRepo:
    if settings.PERSISTENCE_PROVIDER == PersistenceProvider.POSTGRES:
        return PostgresSummaryRepo()
    raise NotImplementedError(
        f"No SummaryRepo adapter for provider '{settings.PERSISTENCE_PROVIDER}'"
    )
```

In `app/retrieval/infrastructure/factory.py`:
- delete the functions `create_document_repo`, `create_summary_repo`, `create_conversation_repo` (unused imports are removed by ruff in Step 6);
- change the docstring's first line to `"""Selects the adapters for the retrieval context: vector store, knowledge graph, extractor.`;
- in the docstring, replace `(\`app.shared.dependencies\`, the use cases, the routers)` with `(\`app.retrieval.dependencies\`, the use cases, the routers)`.

- [ ] **Step 5: Point `app/shared/dependencies.py` at the four factories**

Replace the flagged line

```python
from app.knowledge_management.infrastructure.persistence import factory
```

with

```python
from app.conversations.infrastructure import factory as conversations_factory
from app.documents.infrastructure import factory as documents_factory
from app.retrieval.infrastructure import factory as retrieval_factory
from app.summaries.infrastructure import factory as summaries_factory
```

and the calls:
- `factory.create_document_repo()` → `documents_factory.create_document_repo()`
- `factory.create_summary_repo()` → `summaries_factory.create_summary_repo()`
- `factory.create_conversation_repo()` → `conversations_factory.create_conversation_repo()`
- `factory.create_vector_store_repo()` → `retrieval_factory.create_vector_store_repo()`
- `factory.create_knowledge_graph_repo()` → `retrieval_factory.create_knowledge_graph_repo()`
- `factory.create_entity_extractor()` → `retrieval_factory.create_entity_extractor()`

- [ ] **Step 6: Sort imports, drop unused ones, format**

```bash
uv run ruff check --fix app tests && uv run ruff format app tests
```

Expected: fixes for `I001` / `F401` only; ends with no remaining errors.

- [ ] **Step 7: Verify nothing references the old package**

```bash
grep -rn "knowledge_management" app tests
```

Expected: no output. (Mentions inside docstrings/comments count too — reword each to the new module path.)

- [ ] **Step 8: Run tests and lint**

```bash
uv run pytest && ./scripts/lint.sh
```

Expected: same pass count as before the task, lint clean.

- [ ] **Step 9: Commit**

```bash
git add app tests
git commit -m "split knowledge_management into documents, retrieval, conversations, summaries packages

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Documents delegates indexing to retrieval

**Files:**
- Create: `app/retrieval/application/index_document.py`, `app/retrieval/application/remove_from_index.py`, `app/retrieval/api.py`
- Modify: `app/documents/application/upload_document.py`, `app/documents/application/delete_document.py`, `app/shared/dependencies.py` (providers `get_upload_document_use_case`, `get_delete_document_use_case`)
- Test: create `tests/test_retrieval_index.py`; modify `tests/test_upload_document.py`, `tests/test_delete_document.py`, `tests/test_integration_real_pipeline.py`

**Interfaces:**
- Consumes: Task 1 module paths.
- Produces:
  - `IndexDocumentUseCase(vector_repo: VectorStoreRepo, graph_repo: KnowledgeGraphRepo | None = None, entity_extractor: EntityExtractor | None = None)` with `async execute(document: Document, owner_id: str, pages: list[Document] | None = None) -> None`
  - `RemoveFromIndexUseCase(vector_repo: VectorStoreRepo, graph_repo: KnowledgeGraphRepo | None = None)` with `async execute(doc_id: str, owner_id: str) -> None`
  - `UploadDocumentUseCase(doc_repo: DocumentRepo, index_document: IndexDocumentUseCase)` — `execute` signature unchanged
  - `DeleteDocumentUseCase(doc_repo: DocumentRepo, remove_from_index: RemoveFromIndexUseCase)` — `execute` signature unchanged
  - `app/retrieval/api.py` exporting `IndexDocumentUseCase`, `RemoveFromIndexUseCase`

- [ ] **Step 1: Write the failing tests for the retrieval index**

Create `tests/test_retrieval_index.py`:

```python
from app.retrieval.application.index_document import IndexDocumentUseCase
from app.retrieval.application.remove_from_index import RemoveFromIndexUseCase
from app.retrieval.domain.models import Entity, GraphFragment
from app.retrieval.domain.repositories import EntityExtractor, KnowledgeGraphRepo
from app.shared.kernel.document import Document
from tests.fakes import StubVectorStoreRepo


class RecordingVectorRepo(StubVectorStoreRepo):
    def __init__(self, calls: list[tuple[str, str, str]]) -> None:
        self.calls = calls
        self.added: list[tuple[list[Document], str]] = []

    async def add_documents(self, documents: list[Document], owner_id: str) -> None:
        self.added.append((documents, owner_id))

    async def delete_by_document_id(self, doc_id: str, owner_id: str) -> None:
        self.calls.append(("vectors", doc_id, owner_id))


class RecordingGraphRepo(KnowledgeGraphRepo):
    def __init__(self, calls: list[tuple[str, str, str]]) -> None:
        self.calls = calls
        self.fragments: list[tuple[GraphFragment, str]] = []

    async def add_fragment(self, fragment: GraphFragment, owner_id: str) -> None:
        self.fragments.append((fragment, owner_id))

    async def search_related(self, query: str, owner_id: str, top_k: int = 4) -> list[Document]:
        raise NotImplementedError

    async def delete_by_document_id(self, doc_id: str, owner_id: str) -> None:
        self.calls.append(("graph", doc_id, owner_id))


class RecordingExtractor(EntityExtractor):
    def __init__(self) -> None:
        self.seen: list[Document] = []

    async def extract(self, document: Document) -> GraphFragment:
        self.seen.append(document)
        return GraphFragment(
            doc_id=document.id, entities=[Entity(name="quicksort", type="Algorithm")], relations=[]
        )


async def test_index_extracts_graph_facts_from_the_whole_document_not_the_pages():
    calls: list[tuple[str, str, str]] = []
    vector_repo = RecordingVectorRepo(calls)
    graph_repo = RecordingGraphRepo(calls)
    extractor = RecordingExtractor()
    document = Document(id="o1::r.pdf", content="page 1\n\npage 2", metadata={"filename": "r.pdf"})
    pages = [
        Document(id="ignored", content="page 1", metadata={"page": 1}),
        Document(id="ignored", content="page 2", metadata={"page": 2}),
    ]

    await IndexDocumentUseCase(vector_repo, graph_repo, extractor).execute(document, "o1", pages)

    added_documents, added_owner = vector_repo.added[0]
    assert added_owner == "o1"
    assert [page.metadata["page"] for page in added_documents] == [1, 2]
    # Relations span page boundaries, so the extractor sees the whole document.
    assert extractor.seen == [document]
    fragment, fragment_owner = graph_repo.fragments[0]
    assert fragment.doc_id == "o1::r.pdf"
    assert fragment_owner == "o1"


async def test_remove_from_index_retracts_vectors_then_graph_facts():
    calls: list[tuple[str, str, str]] = []

    await RemoveFromIndexUseCase(RecordingVectorRepo(calls), RecordingGraphRepo(calls)).execute(
        "o1::a.txt", "o1"
    )

    assert calls == [("vectors", "o1::a.txt", "o1"), ("graph", "o1::a.txt", "o1")]
```

- [ ] **Step 2: Update the use-case tests to the new constructors**

In `tests/test_upload_document.py`: add `from app.retrieval.application.index_document import IndexDocumentUseCase` and replace both `UploadDocumentUseCase(doc_repo, vector_repo)` with `UploadDocumentUseCase(doc_repo, IndexDocumentUseCase(vector_repo))`.

In `tests/test_delete_document.py`: add `from app.retrieval.application.remove_from_index import RemoveFromIndexUseCase` and replace all four `DeleteDocumentUseCase(doc_repo, vector_repo)` with `DeleteDocumentUseCase(doc_repo, RemoveFromIndexUseCase(vector_repo))`.

In `tests/test_integration_real_pipeline.py`: add `from app.retrieval.application.index_document import IndexDocumentUseCase` and replace both `UploadDocumentUseCase(doc_repo, vector_repo)` with `UploadDocumentUseCase(doc_repo, IndexDocumentUseCase(vector_repo))`.

- [ ] **Step 3: Run the tests to verify they fail**

```bash
uv run pytest tests/test_retrieval_index.py tests/test_upload_document.py tests/test_delete_document.py -v
```

Expected: collection errors `ModuleNotFoundError: No module named 'app.retrieval.application.index_document'`.

- [ ] **Step 4: Implement the index use cases**

Create `app/retrieval/application/index_document.py`:

```python
from app.retrieval.domain.null_entity_extractor import NullEntityExtractor
from app.retrieval.domain.null_knowledge_graph_repo import NullKnowledgeGraphRepo
from app.retrieval.domain.repositories import EntityExtractor, KnowledgeGraphRepo, VectorStoreRepo
from app.shared.kernel.document import Document


class IndexDocumentUseCase:
    def __init__(
        self,
        vector_repo: VectorStoreRepo,
        graph_repo: KnowledgeGraphRepo | None = None,
        entity_extractor: EntityExtractor | None = None,
    ):
        self.vector_repo = vector_repo
        self.graph_repo = graph_repo or NullKnowledgeGraphRepo()
        self.entity_extractor = entity_extractor or NullEntityExtractor()

    async def execute(
        self, document: Document, owner_id: str, pages: list[Document] | None = None
    ) -> None:
        # What goes into the vector store is either the pages (if the loader split them),
        # keeping the page number, or the whole document. Chunking into smaller pieces
        # happens further down, in the vector repository.
        if pages:
            vector_docs = [
                Document(
                    id=document.id,
                    content=page.content,
                    metadata={**document.metadata, "page": page.metadata.get("page")},
                )
                for page in pages
            ]
        else:
            vector_docs = [document]
        await self.vector_repo.add_documents(vector_docs, owner_id)

        # Feed the knowledge graph from the whole document, not the per-page split: relations
        # regularly span a page boundary, and the extractor needs the surrounding text to see
        # them. With no graph configured both calls are no-ops and cost nothing.
        fragment = await self.entity_extractor.extract(document)
        await self.graph_repo.add_fragment(fragment, owner_id)
```

Create `app/retrieval/application/remove_from_index.py`:

```python
from app.retrieval.domain.null_knowledge_graph_repo import NullKnowledgeGraphRepo
from app.retrieval.domain.repositories import KnowledgeGraphRepo, VectorStoreRepo


class RemoveFromIndexUseCase:
    def __init__(self, vector_repo: VectorStoreRepo, graph_repo: KnowledgeGraphRepo | None = None):
        self.vector_repo = vector_repo
        self.graph_repo = graph_repo or NullKnowledgeGraphRepo()

    async def execute(self, doc_id: str, owner_id: str) -> None:
        await self.vector_repo.delete_by_document_id(doc_id, owner_id)
        # Retract this document's facts too, or the graph keeps answering from a document the
        # user believes they deleted.
        await self.graph_repo.delete_by_document_id(doc_id, owner_id)
```

Create `app/retrieval/api.py`:

```python
"""The retrieval context's public surface: other contexts import from here and nowhere else."""

from app.retrieval.application.index_document import IndexDocumentUseCase
from app.retrieval.application.remove_from_index import RemoveFromIndexUseCase

__all__ = ["IndexDocumentUseCase", "RemoveFromIndexUseCase"]
```

(`__all__` is required: mypy strict does not treat plain imports as re-exports.)

- [ ] **Step 5: Make documents delegate to it**

Replace the body of `app/documents/application/upload_document.py` with:

```python
from typing import Any

from app.documents.domain.repositories import DocumentRepo
from app.retrieval.api import IndexDocumentUseCase
from app.shared.kernel.document import Document
from app.shared.kernel.document_identity import namespaced_document_id


class UploadDocumentUseCase:
    def __init__(self, doc_repo: DocumentRepo, index_document: IndexDocumentUseCase):
        self.doc_repo = doc_repo
        self.index_document = index_document

    async def execute(
        self,
        doc_id: str,
        content: str,
        metadata: dict[str, Any],
        owner_id: str,
        pages: list[Document] | None = None,
    ) -> Document:
        # Namespaced per user so two users can upload a file with the same name without
        # colliding on the key; the rule itself lives in the domain. The original name stays
        # in the metadata ("filename") for display.
        document_id = namespaced_document_id(owner_id, doc_id)
        document = Document(id=document_id, content=content, metadata=metadata)
        await self.doc_repo.save(document, owner_id)
        await self.index_document.execute(document, owner_id, pages)
        return document
```

In `app/documents/application/delete_document.py`:
- replace the imports of `NullKnowledgeGraphRepo`, `KnowledgeGraphRepo`, `VectorStoreRepo` with `from app.retrieval.api import RemoveFromIndexUseCase` (keep `DocumentRepo`);
- replace the class with:

```python
class DeleteDocumentUseCase:
    def __init__(self, doc_repo: DocumentRepo, remove_from_index: RemoveFromIndexUseCase):
        self.doc_repo = doc_repo
        self.remove_from_index = remove_from_index

    async def execute(self, doc_id: str, owner_id: str) -> None:
        # get_by_id is filtered by owner_id — someone else's document is never found, so we
        # touch neither the file nor the vectors.
        doc = await self.doc_repo.get_by_id(doc_id, owner_id)
        if doc is None:
            return
        if "file_path" in doc.metadata:
            await to_thread.run_sync(_remove_file_if_within_storage, doc.metadata["file_path"])

        await self.remove_from_index.execute(doc_id, owner_id)
        await self.doc_repo.delete(doc_id, owner_id)
```

- [ ] **Step 6: Update the two providers in `app/shared/dependencies.py`**

Add `from app.retrieval.api import IndexDocumentUseCase, RemoveFromIndexUseCase` and change only the returns:

```python
    return UploadDocumentUseCase(
        doc_repo, IndexDocumentUseCase(vector_repo, graph_repo, entity_extractor)
    )
```

```python
    return DeleteDocumentUseCase(doc_repo, RemoveFromIndexUseCase(vector_repo, graph_repo))
```

- [ ] **Step 7: Run tests and lint**

```bash
uv run pytest && ./scripts/lint.sh
```

Expected: all pass, including the two new tests.

- [ ] **Step 8: Commit**

```bash
git add app/retrieval app/documents app/shared/dependencies.py tests/test_retrieval_index.py tests/test_upload_document.py tests/test_delete_document.py tests/test_integration_real_pipeline.py
git commit -m "documents delegates vector and graph indexing to retrieval use cases

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Ask and chat retrieve through `RetrievalPipeline`

**Files:**
- Modify: `app/retrieval/application/ask_question.py`, `app/retrieval/application/retrieval_pipeline.py` (docstring), `app/conversations/application/chat_with_docs.py`, `app/retrieval/api.py`, `app/shared/dependencies.py` (providers `get_ask_question_use_case`, `get_chat_with_docs_use_case`)
- Test: modify `tests/test_ask_question.py`, `tests/test_chat_with_docs.py`, `tests/test_integration_real_pipeline.py`

**Interfaces:**
- Consumes: `RetrievalPipeline(vector_repo, reranker, candidate_count=20, top_k=4, graph_repo=None)` with `async retrieve(query: str, owner_id: str) -> list[Document]` (moved in Task 1, unchanged).
- Produces:
  - `AskQuestionUseCase(retrieval_pipeline: RetrievalPipeline, rag_service: RAGService)`
  - `ChatWithDocsUseCase(conversation_repo: ConversationRepo, retrieval_pipeline: RetrievalPipeline, rag_service: RAGService)` — `execute` / `execute_stream` signatures unchanged
  - `app/retrieval/api.py` additionally exporting `RetrievalPipeline`, `RAGService`, `Answer`

- [ ] **Step 1: Update the tests to the new constructors**

`tests/test_ask_question.py`: add `from app.retrieval.application.retrieval_pipeline import RetrievalPipeline` and replace the construction with

```python
    use_case = AskQuestionUseCase(
        RetrievalPipeline(vector_repo, reranker, candidate_count=20, top_k=2), rag_service
    )
```

`tests/test_chat_with_docs.py`: add `from app.retrieval.application.retrieval_pipeline import RetrievalPipeline`; replace every `ChatWithDocsUseCase(vec, rag, conv_repo, _passthrough_reranker())` (6 places) with `ChatWithDocsUseCase(conv_repo, RetrievalPipeline(vec, _passthrough_reranker()), rag)`, and in `test_chat_reranks_candidates_before_answering` replace `ChatWithDocsUseCase(vec, rag, conv_repo, reranker, candidate_count=5, top_k=2)` with `ChatWithDocsUseCase(conv_repo, RetrievalPipeline(vec, reranker, candidate_count=5, top_k=2), rag)`.

`tests/test_integration_real_pipeline.py`: add the same import and replace both `AskQuestionUseCase(...)` constructions with

```python
    ask = AskQuestionUseCase(
        RetrievalPipeline(vector_repo, NoOpReranker(), candidate_count=20, top_k=4),
        LangChainRAGService(llm=fake_llm),
    )
```

- [ ] **Step 2: Run them to verify they fail**

```bash
uv run pytest tests/test_ask_question.py tests/test_chat_with_docs.py -v
```

Expected: FAIL with `TypeError: ... missing 1 required positional argument: 'reranker'` from the old constructors.

- [ ] **Step 3: Use the pipeline in ask**

Replace `app/retrieval/application/ask_question.py` with:

```python
from app.retrieval.application.retrieval_pipeline import RetrievalPipeline
from app.retrieval.domain.models import Answer
from app.retrieval.domain.repositories import RAGService


class AskQuestionUseCase:
    def __init__(self, retrieval_pipeline: RetrievalPipeline, rag_service: RAGService):
        self.retrieval_pipeline = retrieval_pipeline
        self.rag_service = rag_service

    async def execute(self, question_text: str, owner_id: str) -> Answer:
        # Retrieval is limited to the asker's own documents (owner_id).
        relevant_docs = await self.retrieval_pipeline.retrieve(question_text, owner_id)
        answer_text = await self.rag_service.answer_question(question_text, relevant_docs)

        return Answer(text=answer_text, sources=relevant_docs)
```

In `app/retrieval/application/retrieval_pipeline.py` replace the first docstring paragraph

```
    """The shared `search(N candidates) → rerank → top_k` step.

    Mirrors exactly the pattern the QA/chat use cases follow, so the evaluation measures the
    retrieval that really reaches production. Shared by the retrieval and the generation
    evaluator (DRY).
```

with

```
    """The shared `search(N candidates) → rerank → top_k` step.

    Used by the QA and chat use cases and by both evaluators, so the evaluation measures the
    retrieval that really reaches production.
```

- [ ] **Step 4: Export from the retrieval API**

Replace `app/retrieval/api.py` with:

```python
"""The retrieval context's public surface: other contexts import from here and nowhere else."""

from app.retrieval.application.index_document import IndexDocumentUseCase
from app.retrieval.application.remove_from_index import RemoveFromIndexUseCase
from app.retrieval.application.retrieval_pipeline import RetrievalPipeline
from app.retrieval.domain.models import Answer
from app.retrieval.domain.repositories import RAGService

__all__ = [
    "Answer",
    "IndexDocumentUseCase",
    "RAGService",
    "RemoveFromIndexUseCase",
    "RetrievalPipeline",
]
```

- [ ] **Step 5: Use the pipeline in chat**

In `app/conversations/application/chat_with_docs.py`:
- replace the imports of `Answer`, `NullKnowledgeGraphRepo`, `KnowledgeGraphRepo`, `RAGService`, `RerankerService`, `VectorStoreRepo`, `CandidateRetriever` with `from app.retrieval.api import Answer, RAGService, RetrievalPipeline` (keep `ChatMessage`, `Conversation`, `ConversationRepo`, `Document`, `EntityNotFoundException`);
- replace `__init__` with

```python
    def __init__(
        self,
        conversation_repo: ConversationRepo,
        retrieval_pipeline: RetrievalPipeline,
        rag_service: RAGService,
    ):
        self.conversation_repo = conversation_repo
        self.retrieval_pipeline = retrieval_pipeline
        self.rag_service = rag_service
```

- in `_prepare_context` replace

```python
        candidates = await self.retriever.retrieve(
            search_query, owner_id, candidate_count=self.candidate_count
        )
        relevant_documents = await self.reranker.rerank(search_query, candidates, top_k=self.top_k)
```

with

```python
        relevant_documents = await self.retrieval_pipeline.retrieve(search_query, owner_id)
```

- [ ] **Step 6: Update the two providers in `app/shared/dependencies.py`**

Add `from app.retrieval.api import RetrievalPipeline` (merge into the existing `app.retrieval.api` import) and change only the returns:

```python
    return AskQuestionUseCase(
        RetrievalPipeline(
            vector_repo,
            reranker,
            candidate_count=settings.RETRIEVAL_CANDIDATE_COUNT,
            top_k=settings.RETRIEVAL_TOP_K,
            graph_repo=graph_repo,
        ),
        rag_service,
    )
```

```python
    return ChatWithDocsUseCase(
        conversation_repo,
        RetrievalPipeline(
            vector_repo,
            reranker,
            candidate_count=settings.RETRIEVAL_CANDIDATE_COUNT,
            top_k=settings.RETRIEVAL_TOP_K,
            graph_repo=graph_repo,
        ),
        rag_service,
    )
```

- [ ] **Step 7: Run tests and lint**

```bash
uv run pytest && ./scripts/lint.sh
```

Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add app/retrieval app/conversations app/shared/dependencies.py tests/test_ask_question.py tests/test_chat_with_docs.py tests/test_integration_real_pipeline.py
git commit -m "ask and chat retrieve through RetrievalPipeline instead of repeating search and rerank

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Per-context dependency injection and public APIs

**Files:**
- Create: `app/retrieval/dependencies.py`, `app/documents/dependencies.py`, `app/documents/api.py`, `app/conversations/dependencies.py`, `app/summaries/dependencies.py`
- Modify: `app/retrieval/api.py`, `app/main.py`, `app/retrieval/application/evaluation/run_evaluation.py`, `app/summaries/application/summarize_docs.py`, `app/conversations/ui/router.py`, routers' imports (via script), `tests/test_lifespan.py`, tests importing providers (via script)
- Delete: `app/shared/dependencies.py`

**Interfaces:**
- Consumes: Task 2 and 3 constructors.
- Produces:
  - each `app/X/dependencies.py`: `init() -> None`, `async shutdown() -> None`, and the providers listed in `DEPS_NAMES` of the Task 1 script
  - `app/retrieval/dependencies.py` additionally: `get_retrieval_pipeline`, `get_index_document_use_case`, `get_remove_from_index_use_case`
  - `app/retrieval/api.py` exports: `Answer, IndexDocumentUseCase, RAGService, RemoveFromIndexUseCase, RetrievalPipeline, format_sources, get_index_document_use_case, get_rag_service, get_remove_from_index_use_case, get_retrieval_pipeline`
  - `app/documents/api.py` exports: `DocumentRepo, get_doc_repo`

- [ ] **Step 1: Update the lifespan test first**

Replace the top half of `tests/test_lifespan.py` (imports and the first test) with:

```python
"""The application lifespan has to actually release connections on shutdown.

`TestClient(app)` used bare never triggers lifespan — only entering it as a context manager
does, which is why the omission went unnoticed: the pool was simply never disposed.
"""

from fastapi.testclient import TestClient

from app.main import app
from app.retrieval import dependencies
from tests.fakes import StubVectorStoreRepo


class ClosableVectorRepo(StubVectorStoreRepo):
    """Records whether the lifespan asked the adapter to release its connection."""

    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


async def test_shutdown_closes_the_vector_repo_and_clears_singletons(monkeypatch):
    repo = ClosableVectorRepo()
    monkeypatch.setattr(dependencies, "_vector_repo", repo)

    await dependencies.shutdown()

    assert repo.closed
    # Cleared, so the next startup builds a repo that is not backed by a closed driver.
    assert dependencies._vector_repo is None
```

(`test_lifespan_runs_shutdown_on_client_exit` and `_record` stay unchanged.)

- [ ] **Step 2: Run it to verify it fails**

```bash
uv run pytest tests/test_lifespan.py -v
```

Expected: collection error `ModuleNotFoundError: No module named 'app.retrieval.dependencies'`.

- [ ] **Step 3: Create `app/retrieval/dependencies.py`**

```python
"""Dependency injection for the retrieval context.

Every provider is typed against the port, never a concrete adapter; the only code that names
adapters is `infrastructure/factory.py`. The repositories are process-wide singletons built
eagerly by `init()` in the application lifespan, and lazily (under a lock) for callers that
run without one, such as the evaluation harness.
"""

import threading
from typing import Annotated

from fastapi import Depends

from app.retrieval.application.ask_question import AskQuestionUseCase
from app.retrieval.application.index_document import IndexDocumentUseCase
from app.retrieval.application.remove_from_index import RemoveFromIndexUseCase
from app.retrieval.application.retrieval_pipeline import RetrievalPipeline
from app.retrieval.domain.repositories import (
    EntityExtractor,
    KnowledgeGraphRepo,
    RerankerService,
    VectorStoreRepo,
)
from app.retrieval.infrastructure import factory
from app.retrieval.infrastructure.langchain_rag_service import LangChainRAGService
from app.retrieval.infrastructure.reranker_factory import RerankerFactory
from app.shared.config import settings
from app.shared.llm.llm_factory import LLMFactory

_singleton_lock = threading.Lock()

_vector_repo: VectorStoreRepo | None = None
_graph_repo: KnowledgeGraphRepo | None = None
_entity_extractor: EntityExtractor | None = None


def get_vector_repo() -> VectorStoreRepo:
    global _vector_repo
    with _singleton_lock:
        if _vector_repo is None:
            _vector_repo = factory.create_vector_store_repo()
        return _vector_repo


def get_graph_repo() -> KnowledgeGraphRepo:
    global _graph_repo
    with _singleton_lock:
        if _graph_repo is None:
            _graph_repo = factory.create_knowledge_graph_repo()
        return _graph_repo


def get_entity_extractor() -> EntityExtractor:
    global _entity_extractor
    with _singleton_lock:
        if _entity_extractor is None:
            _entity_extractor = factory.create_entity_extractor()
        return _entity_extractor


def init() -> None:
    """Builds the singletons up front, so an unreachable Neo4j fails startup, not a query."""
    get_vector_repo()
    get_graph_repo()
    get_entity_extractor()


async def shutdown() -> None:
    """Closes the drivers the vector store and the graph may own, and clears the singletons so
    a later startup in the same process (tests, an ASGI reload) builds them fresh."""
    global _vector_repo, _graph_repo, _entity_extractor
    if _vector_repo is not None:
        await _vector_repo.close()
    if _graph_repo is not None:
        await _graph_repo.close()
    _vector_repo = _graph_repo = _entity_extractor = None


def get_rag_service() -> LangChainRAGService:
    return LangChainRAGService(llm=LLMFactory.get_llm())


def get_reranker_service() -> RerankerService:
    return RerankerFactory.get_reranker()


def get_retrieval_pipeline(
    vector_repo: Annotated[VectorStoreRepo, Depends(get_vector_repo)],
    reranker: Annotated[RerankerService, Depends(get_reranker_service)],
    graph_repo: Annotated[KnowledgeGraphRepo, Depends(get_graph_repo)],
) -> RetrievalPipeline:
    return RetrievalPipeline(
        vector_repo,
        reranker,
        candidate_count=settings.RETRIEVAL_CANDIDATE_COUNT,
        top_k=settings.RETRIEVAL_TOP_K,
        graph_repo=graph_repo,
    )


def get_ask_question_use_case(
    retrieval_pipeline: Annotated[RetrievalPipeline, Depends(get_retrieval_pipeline)],
    rag_service: Annotated[LangChainRAGService, Depends(get_rag_service)],
) -> AskQuestionUseCase:
    return AskQuestionUseCase(retrieval_pipeline, rag_service)


def get_index_document_use_case(
    vector_repo: Annotated[VectorStoreRepo, Depends(get_vector_repo)],
    graph_repo: Annotated[KnowledgeGraphRepo, Depends(get_graph_repo)],
    entity_extractor: Annotated[EntityExtractor, Depends(get_entity_extractor)],
) -> IndexDocumentUseCase:
    return IndexDocumentUseCase(vector_repo, graph_repo, entity_extractor)


def get_remove_from_index_use_case(
    vector_repo: Annotated[VectorStoreRepo, Depends(get_vector_repo)],
    graph_repo: Annotated[KnowledgeGraphRepo, Depends(get_graph_repo)],
) -> RemoveFromIndexUseCase:
    return RemoveFromIndexUseCase(vector_repo, graph_repo)
```

- [ ] **Step 4: Extend `app/retrieval/api.py`**

```python
"""The retrieval context's public surface: other contexts import from here and nowhere else."""

from app.retrieval.application.index_document import IndexDocumentUseCase
from app.retrieval.application.remove_from_index import RemoveFromIndexUseCase
from app.retrieval.application.retrieval_pipeline import RetrievalPipeline
from app.retrieval.dependencies import (
    get_index_document_use_case,
    get_rag_service,
    get_remove_from_index_use_case,
    get_retrieval_pipeline,
)
from app.retrieval.domain.models import Answer
from app.retrieval.domain.repositories import RAGService
from app.retrieval.ui.sources import format_sources

__all__ = [
    "Answer",
    "IndexDocumentUseCase",
    "RAGService",
    "RemoveFromIndexUseCase",
    "RetrievalPipeline",
    "format_sources",
    "get_index_document_use_case",
    "get_rag_service",
    "get_remove_from_index_use_case",
    "get_retrieval_pipeline",
]
```

- [ ] **Step 5: Create `app/documents/dependencies.py` and `app/documents/api.py`**

`app/documents/dependencies.py`:

```python
import threading
from typing import Annotated

from fastapi import Depends

from app.documents.application.delete_document import DeleteDocumentUseCase
from app.documents.application.upload_document import UploadDocumentUseCase
from app.documents.domain.repositories import DocumentRepo
from app.documents.infrastructure import factory
from app.retrieval.api import (
    IndexDocumentUseCase,
    RemoveFromIndexUseCase,
    get_index_document_use_case,
    get_remove_from_index_use_case,
)

_singleton_lock = threading.Lock()

_doc_repo: DocumentRepo | None = None


def get_doc_repo() -> DocumentRepo:
    global _doc_repo
    with _singleton_lock:
        if _doc_repo is None:
            _doc_repo = factory.create_document_repo()
        return _doc_repo


def init() -> None:
    get_doc_repo()


async def shutdown() -> None:
    # Nothing to close: the repository borrows the shared pool that `dispose_engine()` closes.
    global _doc_repo
    _doc_repo = None


def get_upload_document_use_case(
    doc_repo: Annotated[DocumentRepo, Depends(get_doc_repo)],
    index_document: Annotated[IndexDocumentUseCase, Depends(get_index_document_use_case)],
) -> UploadDocumentUseCase:
    return UploadDocumentUseCase(doc_repo, index_document)


def get_delete_document_use_case(
    doc_repo: Annotated[DocumentRepo, Depends(get_doc_repo)],
    remove_from_index: Annotated[RemoveFromIndexUseCase, Depends(get_remove_from_index_use_case)],
) -> DeleteDocumentUseCase:
    return DeleteDocumentUseCase(doc_repo, remove_from_index)
```

`app/documents/api.py`:

```python
"""The documents context's public surface: other contexts import from here and nowhere else."""

from app.documents.dependencies import get_doc_repo
from app.documents.domain.repositories import DocumentRepo

__all__ = ["DocumentRepo", "get_doc_repo"]
```

- [ ] **Step 6: Create `app/conversations/dependencies.py`**

```python
import threading
from typing import Annotated

from fastapi import Depends

from app.conversations.application.chat_with_docs import ChatWithDocsUseCase
from app.conversations.application.manage_conversations import (
    DeleteConversationUseCase,
    GetConversationUseCase,
    ListConversationsUseCase,
)
from app.conversations.domain.repositories import ConversationRepo
from app.conversations.infrastructure import factory
from app.retrieval.api import RAGService, RetrievalPipeline, get_rag_service, get_retrieval_pipeline

_singleton_lock = threading.Lock()

_conversation_repo: ConversationRepo | None = None


def get_conversation_repo() -> ConversationRepo:
    global _conversation_repo
    with _singleton_lock:
        if _conversation_repo is None:
            _conversation_repo = factory.create_conversation_repo()
        return _conversation_repo


def init() -> None:
    get_conversation_repo()


async def shutdown() -> None:
    # Nothing to close: the repository borrows the shared pool that `dispose_engine()` closes.
    global _conversation_repo
    _conversation_repo = None


def get_chat_with_docs_use_case(
    conversation_repo: Annotated[ConversationRepo, Depends(get_conversation_repo)],
    retrieval_pipeline: Annotated[RetrievalPipeline, Depends(get_retrieval_pipeline)],
    rag_service: Annotated[RAGService, Depends(get_rag_service)],
) -> ChatWithDocsUseCase:
    return ChatWithDocsUseCase(conversation_repo, retrieval_pipeline, rag_service)


def get_list_conversations_use_case(
    conversation_repo: Annotated[ConversationRepo, Depends(get_conversation_repo)],
) -> ListConversationsUseCase:
    return ListConversationsUseCase(conversation_repo)


def get_get_conversation_use_case(
    conversation_repo: Annotated[ConversationRepo, Depends(get_conversation_repo)],
) -> GetConversationUseCase:
    return GetConversationUseCase(conversation_repo)


def get_delete_conversation_use_case(
    conversation_repo: Annotated[ConversationRepo, Depends(get_conversation_repo)],
) -> DeleteConversationUseCase:
    return DeleteConversationUseCase(conversation_repo)
```

- [ ] **Step 7: Create `app/summaries/dependencies.py`**

```python
import threading
from typing import Annotated

from fastapi import Depends

from app.documents.api import DocumentRepo, get_doc_repo
from app.shared.llm.llm_factory import LLMFactory
from app.summaries.application.delete_summary import DeleteSummaryUseCase
from app.summaries.application.summarize_docs import SummarizeDocsUseCase
from app.summaries.domain.repositories import SummaryRepo
from app.summaries.infrastructure import factory
from app.summaries.infrastructure.langchain_summarizer import LangChainSummarizer

_singleton_lock = threading.Lock()

_summary_repo: SummaryRepo | None = None


def get_summary_repo() -> SummaryRepo:
    global _summary_repo
    with _singleton_lock:
        if _summary_repo is None:
            _summary_repo = factory.create_summary_repo()
        return _summary_repo


def init() -> None:
    get_summary_repo()


async def shutdown() -> None:
    # Nothing to close: the repository borrows the shared pool that `dispose_engine()` closes.
    global _summary_repo
    _summary_repo = None


def get_summarizer() -> LangChainSummarizer:
    return LangChainSummarizer(llm=LLMFactory.get_llm())


def get_summarize_docs_use_case(
    doc_repo: Annotated[DocumentRepo, Depends(get_doc_repo)],
    summarizer: Annotated[LangChainSummarizer, Depends(get_summarizer)],
    summary_repo: Annotated[SummaryRepo, Depends(get_summary_repo)],
) -> SummarizeDocsUseCase:
    return SummarizeDocsUseCase(doc_repo, summarizer, summary_repo)


def get_delete_summary_use_case(
    summary_repo: Annotated[SummaryRepo, Depends(get_summary_repo)],
) -> DeleteSummaryUseCase:
    return DeleteSummaryUseCase(summary_repo)
```

- [ ] **Step 8: Rewire `app/main.py` lifespan**

Replace `from app.shared.dependencies import init_repositories, shutdown_repositories` with

```python
from app.conversations import dependencies as conversations_dependencies
from app.documents import dependencies as documents_dependencies
from app.retrieval import dependencies as retrieval_dependencies
from app.summaries import dependencies as summaries_dependencies
```

and in `lifespan` replace `init_repositories()` with

```python
    documents_dependencies.init()
    retrieval_dependencies.init()
    conversations_dependencies.init()
    summaries_dependencies.init()
```

and `await shutdown_repositories()` with

```python
    await documents_dependencies.shutdown()
    await retrieval_dependencies.shutdown()
    await conversations_dependencies.shutdown()
    await summaries_dependencies.shutdown()
```

(the lifespan docstring stays as is).

- [ ] **Step 9: Rewire the evaluation harness**

In `app/retrieval/application/evaluation/run_evaluation.py`, function `main`: replace `from app.shared.dependencies import shutdown_repositories` with `from app.retrieval.dependencies import shutdown` and `await shutdown_repositories()` with `await shutdown()`. (`get_vector_repo` / `get_graph_repo` imports are rewritten by the script in Step 11.)

- [ ] **Step 10: Route the remaining cross-context imports through the APIs**

- `app/summaries/application/summarize_docs.py`: `from app.documents.domain.repositories import DocumentRepo` → `from app.documents.api import DocumentRepo`.
- `app/conversations/ui/router.py`: `from app.retrieval.ui.sources import format_sources` → `from app.retrieval.api import format_sources`.

- [ ] **Step 11: Rewrite provider imports with the script and remove the old module**

```bash
python3 "$SCRATCH/relayout.py" deps && git rm -q app/shared/dependencies.py
```

Expected: `rewrote imports in N files`, `nothing flagged`. If it stops with a `KeyError` or `unmapped module`, a file still imports `init_repositories`/`shutdown_repositories` or `from app.shared import dependencies` — fix that file per Steps 8–9 and rerun.

- [ ] **Step 12: Sort imports and format**

```bash
uv run ruff check --fix app tests && uv run ruff format app tests
```

- [ ] **Step 13: Check for import cycles and leftovers**

```bash
uv run python -c "import app.main" && grep -rn "shared.dependencies\|shared import dependencies" app tests
```

Expected: the import succeeds; grep prints nothing.

- [ ] **Step 14: Run tests and lint**

```bash
uv run pytest && ./scripts/lint.sh
```

Expected: all pass.

- [ ] **Step 15: Commit**

```bash
git add app tests
git commit -m "per-context dependency injection and public api modules; remove shared dependencies

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: README and boundary verification

**Files:**
- Modify: `README.md` (Layout section, evaluation command)

**Interfaces:**
- Consumes: final layout from Tasks 1–4.

- [ ] **Step 1: Verify the context boundaries**

```bash
for ctx in documents retrieval conversations summaries; do
  grep -rnE "(from|import) app\.$ctx\." app --include='*.py' \
    | grep -v "^app/$ctx/" | grep -v "^app/main.py:" | grep -vE "from app\.$ctx\.api import"
done
```

Expected: no output. Any line printed is an internal import across contexts: re-export the name from that context's `api.py` (with `__all__`) and import it from there.

- [ ] **Step 2: Verify the evaluation CLI starts at its new path**

```bash
uv run python -m app.retrieval.application.evaluation.run_evaluation --help
```

Expected: argparse usage text, exit code 0.

- [ ] **Step 3: Update the README**

In the "Layout" section, replace the sentence `Two bounded contexts, each in domain / application / infrastructure / ui layers:` and the `app/` tree under it with:

~~~markdown
Five bounded contexts, each in domain / application / infrastructure / ui layers. A context
imports another only through its `api.py`:

```
app/
  identity/        users, registration, login, JWT
  documents/       upload, listing, deletion of stored documents
  retrieval/       vector store, knowledge graph, rerank, answers, evaluation harness
  conversations/   chat and its persisted history
  summaries/       summaries of selected documents
  shared/          config, database pool, exceptions, storage, rate limiting,
                   kernel/ (Document and its id rules), llm/ (model factory)
client/                  React SPA
migrations/              Alembic
eval/                    golden set for the evaluation harness
tests/  contracts/       one suite per port, parametrized over adapters
```

Dependencies point one way: documents → retrieval, conversations → retrieval,
summaries → documents.
~~~

In "Evaluation harness" replace `app.knowledge_management.application.evaluation.run_evaluation` with `app.retrieval.application.evaluation.run_evaluation`. Then:

```bash
grep -n "knowledge_management" README.md
```

Expected: no output.

- [ ] **Step 4: Final full check**

```bash
uv run pytest && ./scripts/lint.sh
```

Expected: all pass.

- [ ] **Step 5: Commit**

`README.md` already had an uncommitted change from the user before this work (a two-line deletion). Show `git diff README.md` and ask the user whether to commit it together with this task's edits; do not decide for them. Then:

```bash
git add README.md
git commit -m "document the bounded-context layout and the new evaluation module path

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
