"""Application port for deterministic generated-artifact verification."""

from dataclasses import dataclass
from typing import Protocol

from careerops_agent_engine.domain.enums import (
    CVArtifactFormat,
)
from careerops_agent_engine.domain.models.cv_version import (
    CVVersion,
)


@dataclass(frozen=True)
class CVArtifactVerificationResult:
    """Deterministic verification result before persistence."""

    passed: bool
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Require failed verification to explain why."""

        if not self.passed and not self.notes:
            raise ValueError("Failed artifact verification requires notes.")


class CVArtifactVerifier(Protocol):
    """Verify one generated CV artifact format."""

    @property
    def artifact_format(self) -> CVArtifactFormat:
        """Return the artifact format supported by this verifier."""

        ...

    def verify(
        self,
        *,
        version: CVVersion,
        data: bytes,
    ) -> CVArtifactVerificationResult:
        """Verify rendered bytes against their immutable CV version."""

        ...
