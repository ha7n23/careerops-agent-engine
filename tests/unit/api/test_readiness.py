"""Tests for the service-readiness endpoint."""

from collections.abc import Iterator
from typing import cast

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.pool import StaticPool

from careerops_agent_engine.api.dependencies import (
    get_database_engine,
)
from careerops_agent_engine.infrastructure.database.readiness import (
    LANGGRAPH_MANAGED_TABLES,
    REQUIRED_CAREEROPS_TABLES,
    get_expected_alembic_heads,
)
from careerops_agent_engine.main import app


class FailingDatabaseEngine:
    """Simulate an unavailable database connection."""

    def connect(self) -> None:
        """Raise the same SQLAlchemy error class used by real failures."""

        raise OperationalError(
            statement="SELECT 1",
            params={},
            orig=RuntimeError("Database unavailable."),
        )


@pytest.fixture
def ready_engine() -> Iterator[Engine]:
    """Provide a reachable database with all required runtime schemas."""

    engine = create_engine(
        "sqlite+pysqlite://",
        connect_args={
            "check_same_thread": False,
        },
        poolclass=StaticPool,
    )

    with engine.begin() as connection:
        connection.execute(
            text("CREATE TABLE alembic_version (version_num VARCHAR(64) NOT NULL)")
        )

        for head in get_expected_alembic_heads():
            connection.execute(
                text("INSERT INTO alembic_version (version_num) VALUES (:version_num)"),
                {"version_num": head},
            )

        required_tables = REQUIRED_CAREEROPS_TABLES | LANGGRAPH_MANAGED_TABLES

        for table_name in sorted(required_tables):
            connection.execute(text(f'CREATE TABLE "{table_name}" (id INTEGER)'))

    try:
        yield engine

    finally:
        engine.dispose()


def test_readiness_returns_ready_when_database_is_reachable(
    ready_engine: Engine,
) -> None:
    """A reachable database should make CareerOps ready."""

    app.dependency_overrides[get_database_engine] = lambda: ready_engine

    try:
        with TestClient(app) as client:
            response = client.get("/ready")

    finally:
        app.dependency_overrides.pop(
            get_database_engine,
            None,
        )

    assert response.status_code == 200

    assert response.json() == {
        "status": "ready",
        "database": "ok",
    }


def test_readiness_returns_503_when_database_is_unavailable() -> None:
    """Database failure should make the service not ready."""

    failing_engine = cast(
        Engine,
        FailingDatabaseEngine(),
    )

    app.dependency_overrides[get_database_engine] = lambda: failing_engine

    try:
        with TestClient(app) as client:
            response = client.get("/ready")

    finally:
        app.dependency_overrides.pop(
            get_database_engine,
            None,
        )

    assert response.status_code == 503

    assert response.json() == {"detail": ("Service dependencies are not ready.")}


def test_readiness_returns_503_when_schema_is_not_initialized() -> None:
    """A reachable but empty database must not be considered ready."""

    engine = create_engine("sqlite+pysqlite:///:memory:")

    app.dependency_overrides[get_database_engine] = lambda: engine

    try:
        with TestClient(app) as client:
            response = client.get("/ready")

    finally:
        app.dependency_overrides.pop(
            get_database_engine,
            None,
        )

        engine.dispose()

    assert response.status_code == 503

    assert response.json() == {"detail": ("Service dependencies are not ready.")}
