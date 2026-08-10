"""Database readiness checks for the CareerOps runtime."""

from functools import lru_cache

from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine

import careerops_agent_engine.infrastructure.database.models  # noqa: F401
from careerops_agent_engine.infrastructure.database.base import Base

ALEMBIC_CONFIG_PATH = "alembic.ini"

LANGGRAPH_MANAGED_TABLES = frozenset(
    {
        "checkpoint_migrations",
        "checkpoints",
        "checkpoint_blobs",
        "checkpoint_writes",
    }
)

REQUIRED_CAREEROPS_TABLES = frozenset(Base.metadata.tables)


class DatabaseSchemaNotReadyError(RuntimeError):
    """Required CareerOps database schemas are not ready."""


@lru_cache
def get_expected_alembic_heads() -> frozenset[str]:
    """Return the migration heads shipped with this application."""

    config = Config(ALEMBIC_CONFIG_PATH)

    script_directory = ScriptDirectory.from_config(config)

    return frozenset(script_directory.get_heads())


def check_database_readiness(
    engine: Engine,
) -> None:
    """Raise when connectivity or required database schemas are unavailable."""

    expected_heads = get_expected_alembic_heads()

    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))

        current_heads = frozenset(
            connection.execute(
                text("SELECT version_num FROM alembic_version")
            ).scalars()
        )

        table_names = frozenset(inspect(connection).get_table_names())

    if current_heads != expected_heads:
        raise DatabaseSchemaNotReadyError(
            "CareerOps database migrations are not at the current Alembic head."
        )

    missing_careerops = REQUIRED_CAREEROPS_TABLES - table_names

    if missing_careerops:
        raise DatabaseSchemaNotReadyError(
            "Required CareerOps database tables are unavailable."
        )

    missing_langgraph = LANGGRAPH_MANAGED_TABLES - table_names

    if missing_langgraph:
        raise DatabaseSchemaNotReadyError(
            "Required LangGraph checkpoint tables are unavailable."
        )
