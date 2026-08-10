"""Initialize all PostgreSQL schemas required by CareerOps."""

from alembic import command
from alembic.config import Config

from careerops_agent_engine.infrastructure.database.checkpoint import (
    open_postgres_checkpointer,
)

ALEMBIC_CONFIG_PATH = "alembic.ini"


def main() -> None:
    """Upgrade CareerOps and LangGraph database schemas."""

    alembic_config = Config(ALEMBIC_CONFIG_PATH)

    command.upgrade(
        alembic_config,
        "head",
    )

    print("CareerOps Alembic schema ready.")

    with open_postgres_checkpointer() as checkpointer:
        checkpointer.setup()

    print("LangGraph PostgreSQL checkpoint schema ready.")

    print("CareerOps database initialization successful.")


if __name__ == "__main__":
    main()
