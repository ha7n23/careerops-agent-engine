"""Application port for agentic evidence discovery."""

from collections.abc import Sequence
from typing import Protocol

from careerops_agent_engine.domain.models.evidence import EvidenceMatch
from careerops_agent_engine.domain.models.job import JobRequirement


class EvidenceDiscoveryRunner(Protocol):
    """Find approved evidence supporting job requirements."""

    def discover(
        self,
        requirement: JobRequirement,
        *,
        user_id: str,
    ) -> EvidenceMatch:
        """Return a validated evidence match for one requirement."""

        ...

    def discover_for_requirements(
        self,
        requirements: Sequence[JobRequirement],
        *,
        user_id: str,
    ) -> list[EvidenceMatch]:
        """Discover evidence for a complete requirement set."""

        ...
