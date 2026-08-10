"""Secure retrieval of generated CV versions and verified artifact bytes."""

from dataclasses import dataclass
from hashlib import sha256

from careerops_agent_engine.application.exceptions import (
    CVArtifactRetrievalError,
)
from careerops_agent_engine.application.ports.artifact_storage import (
    ArtifactStorage,
)
from careerops_agent_engine.application.ports.cv_version_repository import (
    CVVersionRepository,
)
from careerops_agent_engine.domain.enums import (
    CVArtifactFormat,
    CVArtifactVerificationStatus,
)
from careerops_agent_engine.domain.models.cv_version import (
    CVVersion,
    RenderedCVArtifact,
)

DOCX_MEDIA_TYPE = (
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
)

PDF_MEDIA_TYPE = "application/pdf"


@dataclass(frozen=True)
class RetrievedCVArtifact:
    """Verified artifact bytes safe to return through the API."""

    artifact: RenderedCVArtifact
    filename: str
    media_type: str
    data: bytes


class CVArtifactRetrievalService:
    """Retrieve user-owned CV versions and verified artifact bytes."""

    def __init__(
        self,
        *,
        version_repository: CVVersionRepository,
        artifact_storage: ArtifactStorage,
    ) -> None:
        """Store retrieval dependencies."""

        self._version_repository = version_repository

        self._artifact_storage = artifact_storage

    def get_version(
        self,
        *,
        user_id: str,
        cv_version_id: str,
    ) -> CVVersion:
        """Return one user-owned CV version."""

        version = self._version_repository.get(
            user_id=user_id,
            cv_version_id=cv_version_id,
        )

        if version is None:
            raise CVArtifactRetrievalError("The CV version is unavailable.")

        return version

    def get_verified_artifact(
        self,
        *,
        user_id: str,
        cv_version_id: str,
        artifact_format: CVArtifactFormat,
    ) -> RetrievedCVArtifact:
        """Return verified immutable bytes for one artifact format."""

        version = self.get_version(
            user_id=user_id,
            cv_version_id=cv_version_id,
        )

        artifact = next(
            (
                candidate
                for candidate in version.artifacts
                if candidate.artifact_format is artifact_format
            ),
            None,
        )

        if artifact is None:
            raise CVArtifactRetrievalError("The requested CV artifact is unavailable.")

        if artifact.verification_status is not CVArtifactVerificationStatus.VERIFIED:
            raise CVArtifactRetrievalError("The requested CV artifact is not verified.")

        try:
            data = self._artifact_storage.read(
                user_id=user_id,
                storage_key=artifact.storage_key,
            )

        except FileNotFoundError as exc:
            raise CVArtifactRetrievalError(
                "The stored CV artifact is unavailable."
            ) from exc

        validate_retrieved_artifact_bytes(
            artifact=artifact,
            data=data,
        )

        return RetrievedCVArtifact(
            artifact=artifact,
            filename=build_download_filename(
                cv_version_id=cv_version_id,
                artifact_format=artifact_format,
            ),
            media_type=artifact_media_type(artifact_format),
            data=data,
        )


def validate_retrieved_artifact_bytes(
    *,
    artifact: RenderedCVArtifact,
    data: bytes,
) -> None:
    """Recheck immutable storage bytes before serving them."""

    if len(data) != artifact.size_bytes:
        raise CVArtifactRetrievalError(
            "Stored CV artifact size does not match persisted metadata."
        )

    digest = sha256(data).hexdigest()

    if digest != artifact.sha256_hex:
        raise CVArtifactRetrievalError(
            "Stored CV artifact checksum does not match persisted metadata."
        )


def build_download_filename(
    *,
    cv_version_id: str,
    artifact_format: CVArtifactFormat,
) -> str:
    """Build a safe generated download filename."""

    return f"{cv_version_id}.{artifact_format.value}"


def artifact_media_type(
    artifact_format: CVArtifactFormat,
) -> str:
    """Return the canonical media type for one artifact format."""

    if artifact_format is CVArtifactFormat.DOCX:
        return DOCX_MEDIA_TYPE

    if artifact_format is CVArtifactFormat.PDF:
        return PDF_MEDIA_TYPE

    raise CVArtifactRetrievalError("Unsupported CV artifact format.")
