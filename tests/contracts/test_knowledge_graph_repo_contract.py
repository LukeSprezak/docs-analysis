"""Contract suite for the `KnowledgeGraphRepo` port.

The null implementation is deliberately included as a parameter. It is what runs whenever
`KNOWLEDGE_GRAPH_PROVIDER=none`, so "returns nothing, raises nothing" is a contract the
system depends on, not a stub excused from testing. The assertions that require facts to come
back are marked `needs_storage` and skipped for it.

The behaviours pinned here:

* facts are isolated per owner — one user's graph never answers another user's question;
* a fact asserted by two documents survives deleting one of them (the property that makes
  `doc_ids` a list rather than a single value);
* re-adding a document replaces its previous facts instead of accumulating them.
"""

import uuid
from collections.abc import AsyncIterator, Callable

import pytest

from app.knowledge_management.domain.models import Entity, GraphFragment, Relation
from app.knowledge_management.domain.null_knowledge_graph_repo import NullKnowledgeGraphRepo
from app.knowledge_management.domain.repositories import KnowledgeGraphRepo
from app.knowledge_management.infrastructure.persistence.neo4j_knowledge_graph_repo import (
    Neo4jKnowledgeGraphRepo,
)
from tests.contracts.test_vector_store_repo_contract import (
    NEO4J_TEST_PASSWORD,
    NEO4J_TEST_URI,
    NEO4J_TEST_USERNAME,
)

GraphRepoFactory = Callable[[], KnowledgeGraphRepo]

POSTGRES = Entity(name="PostgreSQL", type="Database")
PGVECTOR = Entity(name="pgvector", type="Extension")
NEO4J = Entity(name="Neo4j", type="Database")


def fragment(doc_id: str, *relations: Relation) -> GraphFragment:
    entities = {entity for relation in relations for entity in (relation.source, relation.target)}
    return GraphFragment(doc_id=doc_id, entities=list(entities), relations=list(relations))


SUPPORTS = Relation(source=POSTGRES, target=PGVECTOR, type="SUPPORTS")
COMPETES = Relation(source=POSTGRES, target=NEO4J, type="COMPETES_WITH")


@pytest.fixture
def null_factory() -> GraphRepoFactory:
    return NullKnowledgeGraphRepo


@pytest.fixture
async def neo4j_graph_factory() -> AsyncIterator[GraphRepoFactory]:
    """Live Neo4j, isolated by a per-test node label and relationship type."""
    created: list[Neo4jKnowledgeGraphRepo] = []

    def build() -> KnowledgeGraphRepo:
        suffix = uuid.uuid4().hex
        repo = Neo4jKnowledgeGraphRepo(
            url=NEO4J_TEST_URI,
            username=NEO4J_TEST_USERNAME,
            password=NEO4J_TEST_PASSWORD,
            node_label=f"contract_entity_{suffix}",
            relationship_type=f"CONTRACT_REL_{suffix}",
            index_name=f"contract_graph_{suffix}",
        )
        created.append(repo)
        return repo

    yield build

    for repo in created:
        repo.graph.query(f"MATCH (n:`{repo.node_label}`) DETACH DELETE n")
        repo.graph.query(f"DROP INDEX {repo.index_name} IF EXISTS")
        repo.graph.query(f"DROP INDEX {repo.index_name}_owner IF EXISTS")
        await repo.close()


GRAPH_ADAPTERS = [
    pytest.param("null_factory", id="null"),
    pytest.param("neo4j_graph_factory", id="neo4j", marks=pytest.mark.integration),
]

# Assertions that only make sense once facts are actually stored — the null repository is
# contractually empty, so it is excluded from these rather than special-cased inside them.
STORING_ADAPTERS = [pytest.param("neo4j_graph_factory", id="neo4j", marks=pytest.mark.integration)]


@pytest.fixture(params=GRAPH_ADAPTERS)
def graph_repo(request: pytest.FixtureRequest) -> KnowledgeGraphRepo:
    factory: GraphRepoFactory = request.getfixturevalue(request.param)
    return factory()


@pytest.fixture(params=STORING_ADAPTERS)
def storing_graph_repo(request: pytest.FixtureRequest) -> KnowledgeGraphRepo:
    factory: GraphRepoFactory = request.getfixturevalue(request.param)
    return factory()


# --- Every adapter, including the null one ----------------------------------------------


async def test_search_on_empty_graph_returns_empty(graph_repo: KnowledgeGraphRepo) -> None:
    assert await graph_repo.search_related("PostgreSQL", owner_id="u1") == []


async def test_delete_on_empty_graph_is_noop(graph_repo: KnowledgeGraphRepo) -> None:
    await graph_repo.delete_by_document_id("does-not-exist", owner_id="u1")  # does not raise


async def test_adding_an_empty_fragment_is_noop(graph_repo: KnowledgeGraphRepo) -> None:
    await graph_repo.add_fragment(GraphFragment("d.txt", [], []), owner_id="u1")
    assert await graph_repo.search_related("anything", owner_id="u1") == []


# --- Adapters that actually store facts ---------------------------------------------------


async def test_search_returns_facts_connected_to_the_named_entity(
    storing_graph_repo: KnowledgeGraphRepo,
) -> None:
    await storing_graph_repo.add_fragment(fragment("d.txt", SUPPORTS), owner_id="u1")

    results = await storing_graph_repo.search_related("PostgreSQL", owner_id="u1", top_k=10)

    assert [doc.content for doc in results] == ["PostgreSQL SUPPORTS pgvector"]
    assert results[0].metadata["source"] == "knowledge_graph"
    assert results[0].metadata["doc_ids"] == ["d.txt"]


async def test_results_carry_the_plain_filename_for_citation_and_scoring(
    storing_graph_repo: KnowledgeGraphRepo,
) -> None:
    # Document ids are namespaced per owner, but citations and the eval harness match on the
    # bare file name — a graph hit without `filename` scores as a miss against every golden
    # set, making the graph look worthless regardless of its actual quality.
    await storing_graph_repo.add_fragment(fragment("alice::manual.pdf", SUPPORTS), owner_id="alice")

    results = await storing_graph_repo.search_related("PostgreSQL", owner_id="alice", top_k=10)

    assert results[0].metadata["filename"] == "manual.pdf"
    assert results[0].metadata["doc_id"] == "alice::manual.pdf"


async def test_search_finds_facts_where_the_entity_is_the_target(
    storing_graph_repo: KnowledgeGraphRepo,
) -> None:
    # Querying the object of a relation must surface it too — otherwise half the graph is
    # unreachable depending on which way the extractor happened to orient the fact.
    await storing_graph_repo.add_fragment(fragment("d.txt", SUPPORTS), owner_id="u1")

    results = await storing_graph_repo.search_related("pgvector", owner_id="u1", top_k=10)

    assert [doc.content for doc in results] == ["PostgreSQL SUPPORTS pgvector"]


async def test_facts_are_isolated_by_owner(storing_graph_repo: KnowledgeGraphRepo) -> None:
    await storing_graph_repo.add_fragment(fragment("alice.txt", SUPPORTS), owner_id="alice")
    await storing_graph_repo.add_fragment(fragment("bob.txt", COMPETES), owner_id="bob")

    alice_results = await storing_graph_repo.search_related("PostgreSQL", "alice", top_k=10)

    assert [doc.content for doc in alice_results] == ["PostgreSQL SUPPORTS pgvector"]


async def test_fact_survives_deleting_only_one_of_the_documents_asserting_it(
    storing_graph_repo: KnowledgeGraphRepo,
) -> None:
    # The reason doc_ids is a list: two documents state the same fact, so removing one must
    # not retract it. Storing a single id would make this delete wipe the fact entirely.
    await storing_graph_repo.add_fragment(fragment("first.txt", SUPPORTS), owner_id="u1")
    await storing_graph_repo.add_fragment(fragment("second.txt", SUPPORTS), owner_id="u1")

    await storing_graph_repo.delete_by_document_id("first.txt", owner_id="u1")

    results = await storing_graph_repo.search_related("PostgreSQL", owner_id="u1", top_k=10)
    assert [doc.content for doc in results] == ["PostgreSQL SUPPORTS pgvector"]
    assert results[0].metadata["doc_ids"] == ["second.txt"]


async def test_deleting_the_last_document_asserting_a_fact_retracts_it(
    storing_graph_repo: KnowledgeGraphRepo,
) -> None:
    await storing_graph_repo.add_fragment(fragment("only.txt", SUPPORTS), owner_id="u1")

    await storing_graph_repo.delete_by_document_id("only.txt", owner_id="u1")

    assert await storing_graph_repo.search_related("PostgreSQL", owner_id="u1", top_k=10) == []


async def test_spelling_variants_of_an_entity_collapse_into_one_node(
    storing_graph_repo: KnowledgeGraphRepo,
) -> None:
    # Two documents naming the same thing differently must build one entity, not two. If they
    # split, each document's facts stay in its own island and a traversal reaches half the
    # corpus — the failure is silent, because both islands look correct on their own.
    loud = Entity(name="POSTGRES.", type="Database")
    quiet = Entity(name="postgres", type="Database")
    await storing_graph_repo.add_fragment(
        fragment("a.txt", Relation(source=loud, target=PGVECTOR, type="SUPPORTS")), owner_id="u1"
    )
    await storing_graph_repo.add_fragment(
        fragment("b.txt", Relation(source=quiet, target=NEO4J, type="COMPETES_WITH")),
        owner_id="u1",
    )

    results = await storing_graph_repo.search_related("postgres", owner_id="u1", top_k=10)

    # Both facts hang off the single merged entity, and it is cited by its first spelling.
    assert {doc.content for doc in results} == {
        "POSTGRES. SUPPORTS pgvector",
        "POSTGRES. COMPETES_WITH Neo4j",
    }


async def test_readding_a_document_replaces_its_previous_facts(
    storing_graph_repo: KnowledgeGraphRepo,
) -> None:
    # A re-upload must not leave behind a fact the new version dropped.
    await storing_graph_repo.add_fragment(fragment("d.txt", SUPPORTS), owner_id="u1")
    await storing_graph_repo.add_fragment(fragment("d.txt", COMPETES), owner_id="u1")

    results = await storing_graph_repo.search_related("PostgreSQL", owner_id="u1", top_k=10)

    assert [doc.content for doc in results] == ["PostgreSQL COMPETES_WITH Neo4j"]
