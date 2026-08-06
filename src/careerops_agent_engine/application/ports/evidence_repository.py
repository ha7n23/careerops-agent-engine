"""Application port for accessing approved career evidence."""

from typing import Protocol

from careerops_agent_engine.domain.models.evidence import CareerEvidence


class EvidenceRepository(Protocol):
    """Retrieve career evidence within one trusted user boundary."""

    def search_approved(
        self,
        *,
        user_id: str,
        query: str,
        limit: int = 5,
    ) -> list[CareerEvidence]:
        """Search approved evidence belonging to one user."""

        ...

    def get_approved(
        self,
        *,
        user_id: str,
        evidence_id: str,
    ) -> CareerEvidence | None:
        """Retrieve one approved evidence record belonging to a user."""

        ...

    def list_approved(
        self,
        *,
        user_id: str,
        limit: int = 100,
    ) -> list[CareerEvidence]:
        """List approved evidence belonging to one user."""

        ...
