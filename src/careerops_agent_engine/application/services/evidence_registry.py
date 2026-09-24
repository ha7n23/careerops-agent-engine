"""Read-only access to the approved Evidence Registry."""

from careerops_agent_engine.application.exceptions import (
    CareerEvidenceUnavailableError,
)
from careerops_agent_engine.application.ports.evidence_repository import (
    EvidenceRepository,
)
from careerops_agent_engine.domain.models.evidence import CareerEvidence


class EvidenceRegistryService:
    """Expose approved evidence within one authenticated user boundary."""

    def __init__(
        self,
        repository: EvidenceRepository,
    ) -> None:
        """Store the approved-evidence repository."""

        self._repository = repository

    def list_approved(
        self,
        *,
        user_id: str,
        limit: int = 100,
    ) -> list[CareerEvidence]:
        """List only approved evidence belonging to the user."""

        return self._repository.list_approved(
            user_id=user_id,
            limit=limit,
        )

    def get_approved(
        self,
        *,
        user_id: str,
        evidence_id: str,
    ) -> CareerEvidence:
        """Retrieve one approved record without leaking ownership."""

        evidence = self._repository.get_approved(
            user_id=user_id,
            evidence_id=evidence_id,
        )

        if evidence is None:
            raise CareerEvidenceUnavailableError(
                "The approved evidence record is unavailable."
            )

        return evidence
