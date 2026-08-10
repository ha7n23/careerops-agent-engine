"""Verify stored generated CV artifacts and persist the result."""

from dataclasses import dataclass
from hashlib import sha256

from careerops_agent_engine.application.exceptions import (
    CVArtifactVerificationError,
)
from careerops_agent_engine.application.ports.artifact_storage import (
    ArtifactStorage,
)
from careerops_agent_engine.application.ports.cv_artifact_verifier import (
    CVArtifactVerificationResult,
    CVArtifactVerifier,
)
from careerops_agent_engine.application.ports.cv_version_repository import (
    CVVersionRepository,
)
from careerops_agent_engine.domain.enums import (
    CVArtifactVerificationStatus,
)
from careerops_agent_engine.domain.models.cv_version import (
    CVVersion,
    RenderedCVArtifact,
)


@dataclass(frozen=True)
class CVArtifactVerificationExecutionResult:
    """Persisted result of verifying one generated CV artifact."""

    version: CVVersion
    artifact: RenderedCVArtifact
    passed: bool
    notes: tuple[str, ...]


class CVArtifactVerificationService:
    """Verify persisted artifact bytes against immutable CV metadata."""

    def __init__(
        self,
        *,
        version_repository: CVVersionRepository,
        artifact_storage: ArtifactStorage,
        verifier: CVArtifactVerifier,
    ) -> None:
        """Store verification dependencies."""

        self._version_repository = version_repository

        self._artifact_storage = artifact_storage

        self._verifier = verifier

    def verify(
        self,
        *,
        user_id: str,
        cv_version_id: str,
    ) -> CVArtifactVerificationExecutionResult:
        """Verify one persisted artifact and persist its terminal result."""

        version = self._version_repository.get(
            user_id=user_id,
            cv_version_id=cv_version_id,
        )

        if version is None:
            raise CVArtifactVerificationError("The CV version is unavailable.")

        artifact = next(
            (
                candidate
                for candidate in version.artifacts
                if candidate.artifact_format is self._verifier.artifact_format
            ),
            None,
        )

        if artifact is None:
            raise CVArtifactVerificationError(
                "The requested CV artifact is unavailable."
            )

        try:
            data = self._artifact_storage.read(
                user_id=user_id,
                storage_key=artifact.storage_key,
            )

        except FileNotFoundError as exc:
            raise CVArtifactVerificationError(
                "The stored CV artifact is unavailable."
            ) from exc

        verification = verify_artifact_integrity(
            version=version,
            artifact=artifact,
            data=data,
            verifier=self._verifier,
        )

        status = (
            CVArtifactVerificationStatus.VERIFIED
            if verification.passed
            else CVArtifactVerificationStatus.FAILED
        )

        persisted_version = self._version_repository.set_artifact_verification(
            user_id=user_id,
            cv_version_id=cv_version_id,
            artifact_id=(artifact.artifact_id),
            verification_status=status,
            verification_notes=list(verification.notes),
        )

        persisted_artifact = next(
            (
                candidate
                for candidate in persisted_version.artifacts
                if candidate.artifact_id == artifact.artifact_id
            ),
            None,
        )

        if persisted_artifact is None:
            raise RuntimeError("Verified CV artifact became unavailable.")

        return CVArtifactVerificationExecutionResult(
            version=persisted_version,
            artifact=persisted_artifact,
            passed=verification.passed,
            notes=verification.notes,
        )


def verify_artifact_integrity(
    *,
    version: CVVersion,
    artifact: RenderedCVArtifact,
    data: bytes,
    verifier: CVArtifactVerifier,
) -> CVArtifactVerificationResult:
    """Verify immutable file metadata before format-specific checks."""

    if len(data) != artifact.size_bytes:
        return CVArtifactVerificationResult(
            passed=False,
            notes=("Stored CV artifact size does not match persisted metadata.",),
        )

    digest = sha256(data).hexdigest()

    if digest != artifact.sha256_hex:
        return CVArtifactVerificationResult(
            passed=False,
            notes=("Stored CV artifact checksum does not match persisted metadata.",),
        )

    return verifier.verify(
        version=version,
        data=data,
    )
