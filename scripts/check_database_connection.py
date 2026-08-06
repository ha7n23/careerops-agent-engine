"""Verify connectivity to the configured CareerOps database."""

from careerops_agent_engine.core.config import get_settings
from careerops_agent_engine.infrastructure.database.session import (
    check_database_connection,
    create_database_engine,
)


def main() -> None:
    """Connect to PostgreSQL and run a minimal query."""

    settings = get_settings()
    engine = create_database_engine(settings)

    try:
        check_database_connection(engine)
        print("CareerOps PostgreSQL connection successful.")
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
