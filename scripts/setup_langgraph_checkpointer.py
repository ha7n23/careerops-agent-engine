"""Create or upgrade LangGraph checkpoint tables in PostgreSQL."""

from careerops_agent_engine.infrastructure.database.checkpoint import (
    open_postgres_checkpointer,
)


def main() -> None:
    """Initialise the LangGraph PostgreSQL checkpoint schema."""

    with open_postgres_checkpointer() as checkpointer:
        checkpointer.setup()

    print("LangGraph PostgreSQL checkpoint schema ready.")


if __name__ == "__main__":
    main()
