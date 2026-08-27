from collections.abc import Callable
from typing import Any

import pytest

from app.identity.dependencies import get_current_user
from app.identity.domain.models import User
from app.main import app
from app.shared.config import settings
from app.shared.rate_limit import limiter

TEST_USER = User(id="test-user-id", email="test@example.com", hashed_password="x")

# What the `override_dependency` fixture hands a test: (dependency, provider) -> None.
InstallOverride = Callable[[Callable[..., Any], Callable[..., Any]], None]


@pytest.fixture(autouse=True)
def authenticated_test_user():
    """Every request in the suite arrives authenticated as TEST_USER.

    Teardown removes this one key rather than calling `clear()`: the override dict lives on
    the module-level `app`, so clearing it drops whatever else is installed — including
    overrides belonging to a fixture that has not finished yet."""
    app.dependency_overrides[get_current_user] = lambda: TEST_USER
    yield
    app.dependency_overrides.pop(get_current_user, None)


@pytest.fixture
def without_test_user():
    """Lets the real authentication guard run, instead of the autouse override above.

    Used by the tests that are about the guard itself. Nothing to undo: the autouse fixture
    reinstalls the override before the next test."""
    app.dependency_overrides.pop(get_current_user, None)


@pytest.fixture
def override_dependency():
    """Installs a dependency override for one test and removes exactly it afterwards.

    Tests used to assign into `app.dependency_overrides` and finish with `clear()`. That
    worked only because the autouse fixture above reinstalled itself before each test: a test
    failing before its own `clear()` left the override behind, and the suite depended on
    running in one particular order. Teardown here runs whether the test passes or not."""
    installed: list[Callable[..., Any]] = []

    def install(dependency: Callable[..., Any], provider: Callable[..., Any]) -> None:
        app.dependency_overrides[dependency] = provider
        installed.append(dependency)

    yield install

    for dependency in installed:
        app.dependency_overrides.pop(dependency, None)


@pytest.fixture(autouse=True)
def disable_rate_limiting():
    """The limiter is disabled during testing—repeated requests from the same address
    (TestClient) would hit the limit. The test designed to check rate limiting intentionally
    enables it (limiter.enabled = True) and resets the counters."""
    limiter.enabled = False
    limiter.reset()
    yield
    limiter.enabled = False
    limiter.reset()


@pytest.fixture(autouse=True)
def storage_in_tmp_path(tmp_path, monkeypatch):
    """Uploads land in this test's own directory, not in the project's `storage/`.

    Without it every upload test writes into the working tree, leaving files behind that the
    next run reads back — and two runs in parallel would fight over the same names."""
    monkeypatch.setattr(settings, "STORAGE_DOCUMENTS_DIR", str(tmp_path / "documents"))
