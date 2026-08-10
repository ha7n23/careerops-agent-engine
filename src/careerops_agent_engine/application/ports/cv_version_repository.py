"""Application port for persistent CV versions."""

from typing import Protocol

from careerops_agent_engine.domain.enums import (
    CVArtifactVerificationStatus,
)
from careerops_agent_engine.domain.models.cv_version import (
    CVVersion,
    RenderedCVArtifact,
)


class CVVersionRepository(Protocol):
    """Persist and retrieve versioned CV business records."""

    def save(
        self,
        *,
        user_id: str,
        version: CVVersion,
    ) -> None:
        """Persist one CV version idempotently."""

        ...

    def get(
        self,
        *,
        user_id: str,
        cv_version_id: str,
    ) -> CVVersion | None:
        """Return one CV version inside its user boundary."""

        ...

    def list_for_cv(
        self,
        *,
        user_id: str,
        cv_id: str,
    ) -> list[CVVersion]:
        """Return ordered versions of one user CV family."""

        ...

    def attach_artifact(
        self,
        *,
        user_id: str,
        cv_version_id: str,
        artifact: RenderedCVArtifact,
    ) -> CVVersion:
        """Attach one newly rendered artifact idempotently."""

        ...

    def set_artifact_verification(
        self,
        *,
        user_id: str,
        cv_version_id: str,
        artifact_id: str,
        verification_status: CVArtifactVerificationStatus,
        verification_notes: list[str],
    ) -> CVVersion:
        """Apply one terminal verification result."""

        ...
