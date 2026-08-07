"""LangGraph PostgreSQL checkpoint infrastructure."""

from collections.abc import Iterator
from contextlib import contextmanager

from langgraph.checkpoint.postgres import PostgresSaver

from careerops_agent_engine.core.config import Settings, get_settings


@contextmanager
def open_postgres_checkpointer(
    settings: Settings | None = None,
) -> Iterator[PostgresSaver]:
    """Open a PostgreSQL-backed LangGraph checkpointer.

    The caller owns the context lifetime. This avoids returning a saver
    whose underlying database connection has already been closed.
    """

    resolved_settings = settings or get_settings()

    with PostgresSaver.from_conn_string(
        resolved_settings.langgraph_database_uri
    ) as checkpointer:
        yield checkpointer
