"""FastAPI dependency providers."""

from functools import lru_cache

from careerops_agent_engine.application.services.job_analysis import (
    JobAnalysisService,
)
from careerops_agent_engine.infrastructure.llm.factory import (
    create_requirement_extractor,
)


@lru_cache
def get_job_analysis_service() -> JobAnalysisService:
    """Create one reusable job-analysis service per application process."""

    extractor = create_requirement_extractor()

    return JobAnalysisService(
        requirement_extractor=extractor,
    )
