"""SQLAlchemy engine and session configuration."""

from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from careerops_agent_engine.core.config import Settings, get_settings


def create_database_engine(
    settings: Settings | None = None,
) -> Engine:
    """Create the configured SQLAlchemy database engine."""

    resolved_settings = settings or get_settings()

    engine_options: dict[str, Any] = {
        "pool_pre_ping": True,
    }

    if not resolved_settings.database_url.startswith("sqlite"):
        engine_options.update(
            {
                "pool_size": resolved_settings.database_pool_size,
                "max_overflow": (resolved_settings.database_max_overflow),
            }
        )

    return create_engine(
        resolved_settings.database_url,
        **engine_options,
    )


def create_session_factory(
    engine: Engine,
) -> sessionmaker[Session]:
    """Create sessions bound to the supplied engine."""

    return sessionmaker(
        bind=engine,
        class_=Session,
        autoflush=False,
        expire_on_commit=False,
    )


def check_database_connection(engine: Engine) -> None:
    """Raise an exception when the database cannot be reached."""

    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))
