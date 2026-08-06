"""Application port for agentic evidence discovery."""

from typing import Protocol

from careerops_agent_engine.domain.models.evidence import EvidenceMatch
from careerops_agent_engine.domain.models.job import JobRequirement


class EvidenceDiscoveryRunner(Protocol):
    """Find approved evidence supporting one job requirement."""

    def discover(
        self,
        requirement: JobRequirement,
        *,
        user_id: str,
    ) -> EvidenceMatch:
        """Return a validated evidence match for one requirement."""

        ...
