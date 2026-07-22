"""The application lifespan has to actually release connections on shutdown.

`TestClient(app)` used bare never triggers lifespan — only entering it as a context manager
does, which is why the omission went unnoticed: the pool was simply never disposed.
"""

from fastapi.testclient import TestClient

from app.main import app
from app.shared import dependencies
from app.shared.dependencies import shutdown_repositories
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

    await shutdown_repositories()

    assert repo.closed
    # Cleared, so the next startup builds a repo that is not backed by a closed driver.
    assert dependencies._vector_repo is None


def test_lifespan_runs_shutdown_on_client_exit(monkeypatch):
    disposed: list[bool] = []
    monkeypatch.setattr("app.main.dispose_engine", lambda: _record(disposed))

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200

    assert disposed == [True], "the lifespan must dispose the shared pool on shutdown"


async def _record(disposed: list[bool]) -> None:
    disposed.append(True)
