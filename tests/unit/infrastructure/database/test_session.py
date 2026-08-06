"""Tests for SQLAlchemy engine and session configuration."""

from sqlalchemy import text

from careerops_agent_engine.core.config import Settings
from careerops_agent_engine.infrastructure.database.session import (
    check_database_connection,
    create_database_engine,
    create_session_factory,
)


def build_sqlite_settings() -> Settings:
    """Create isolated in-memory database settings."""

    return Settings(
        database_url="sqlite+pysqlite:///:memory:",
        langgraph_database_uri=("postgresql://unused:unused@localhost/unused"),
    )


def test_database_connection_check_accepts_live_engine() -> None:
    """The health check should execute against a reachable database."""

    engine = create_database_engine(build_sqlite_settings())

    try:
        check_database_connection(engine)
    finally:
        engine.dispose()


def test_session_factory_creates_working_session() -> None:
    """The session factory should create usable SQLAlchemy sessions."""

    engine = create_database_engine(build_sqlite_settings())
    session_factory = create_session_factory(engine)

    try:
        with session_factory() as session:
            result = session.execute(text("SELECT 1")).scalar_one()

        assert result == 1
    finally:
        engine.dispose()
