"""FastAPI dependency providers."""

from functools import lru_cache
from typing import Annotated

from fastapi import Header, HTTPException, status

from careerops_agent_engine.application.services.job_analysis import (
    JobAnalysisService,
)
from careerops_agent_engine.infrastructure.llm.factory import (
    create_evidence_discovery_runner,
    create_requirement_extractor,
)
from careerops_agent_engine.infrastructure.repositories.development_evidence import (
    create_development_evidence_repository,
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
def get_job_analysis_service() -> JobAnalysisService:
    """Create one reusable job-analysis service per process."""

    repository = create_development_evidence_repository()
    requirement_extractor = create_requirement_extractor()
    evidence_discovery_runner = create_evidence_discovery_runner(repository)

    return JobAnalysisService(
        requirement_extractor=requirement_extractor,
        evidence_discovery_runner=evidence_discovery_runner,
    )
