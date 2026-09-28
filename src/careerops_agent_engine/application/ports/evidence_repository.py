"""Application port for accessing approved career evidence."""

from dataclasses import dataclass
from typing import Protocol

from careerops_agent_engine.domain.enums import (
    EvidenceCategory,
    EvidenceLifecycleStatus,
)
from careerops_agent_engine.domain.models.evidence import (
    CareerEvidence,
    CareerEvidenceEdit,
)


@dataclass(frozen=True, slots=True)
class EvidenceRegistryPage:
    """One stable page of user-scoped approved evidence."""

    items: list[CareerEvidence]
    total: int


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

    def query_approved(
        self,
        *,
        user_id: str,
        query: str | None,
        category: EvidenceCategory | None,
        lifecycle_status: EvidenceLifecycleStatus,
        offset: int,
        limit: int,
    ) -> EvidenceRegistryPage:
        """Search, filter and page approved evidence for one user."""

        ...

    def get_approved_for_management(
        self,
        *,
        user_id: str,
        evidence_id: str,
    ) -> CareerEvidence | None:
        """Retrieve active or archived approved evidence for its owner."""

        ...

    def edit_approved(
        self,
        *,
        user_id: str,
        evidence_id: str,
        edit: CareerEvidenceEdit,
    ) -> CareerEvidence | None:
        """Apply an audited edit to one approved user-owned record."""

        ...

    def set_lifecycle_status(
        self,
        *,
        user_id: str,
        evidence_id: str,
        lifecycle_status: EvidenceLifecycleStatus,
    ) -> CareerEvidence | None:
        """Idempotently archive or restore approved evidence."""

        ...
