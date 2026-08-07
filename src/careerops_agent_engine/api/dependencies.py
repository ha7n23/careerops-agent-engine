"""FastAPI dependency providers."""

from functools import lru_cache
from typing import Annotated

from fastapi import Header, HTTPException, status
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from careerops_agent_engine.application.ports.evidence_repository import (
    EvidenceRepository,
)
from careerops_agent_engine.application.services.job_analysis import (
    JobAnalysisService,
)
from careerops_agent_engine.infrastructure.database.session import (
    create_database_engine,
    create_session_factory,
)
from careerops_agent_engine.infrastructure.llm.factory import (
    create_evidence_discovery_runner,
    create_requirement_extractor,
)
from careerops_agent_engine.infrastructure.repositories.sqlalchemy_evidence import (
    SqlAlchemyEvidenceRepository,
)


def get_authenticated_user_id(
    x_user_id: Annotated[
        str | None,
        Header(alias="X-User-ID"),
    ] = None,
) -> str:
    """Return the authenticated development user identifier."""

    if x_user_id is None or not x_user_id.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="The X-User-ID header is required.",
        )

    return x_user_id.strip()


@lru_cache
def get_database_engine() -> Engine:
    """Create one reusable SQLAlchemy engine per process."""

    return create_database_engine()


@lru_cache
def get_database_session_factory() -> sessionmaker[Session]:
    """Create one reusable database session factory."""

    return create_session_factory(get_database_engine())


@lru_cache
def get_evidence_repository() -> EvidenceRepository:
    """Create the PostgreSQL approved-evidence repository."""

    return SqlAlchemyEvidenceRepository(get_database_session_factory())


@lru_cache
def get_job_analysis_service() -> JobAnalysisService:
    """Create one reusable job-analysis service per process."""

    repository = get_evidence_repository()

    return JobAnalysisService(
        requirement_extractor=create_requirement_extractor(),
        evidence_discovery_runner=(create_evidence_discovery_runner(repository)),
    )
