"""Application port for job-requirement extraction."""

from typing import Protocol

from careerops_agent_engine.domain.models.job import (
    JobRequirementExtraction,
)


class RequirementExtractor(Protocol):
    """Extract structured requirements from a job description."""

    def extract(
        self,
        job_description: str,
        *,
        job_id: str,
    ) -> JobRequirementExtraction:
        """Return validated requirements extracted from one job."""

        ...
