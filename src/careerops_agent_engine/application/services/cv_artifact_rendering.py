"""Render, securely store and persist generated CV artifacts."""

from contextlib import suppress
from dataclasses import dataclass
from hashlib import sha256

from careerops_agent_engine.application.exceptions import (
    CVRenderingError,
)
from careerops_agent_engine.application.ports.artifact_storage import (
    ArtifactStorage,
)
from careerops_agent_engine.application.ports.cv_renderer import (
    CVTemplateRenderer,
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


@dataclass(frozen=True)
class CVArtifactRenderingResult:
    """Authoritative result of rendering and persisting one artifact."""

    version: CVVersion
    artifact: RenderedCVArtifact


class CVArtifactRenderingService:
    """Render and persist an immutable generated CV artifact."""

    def __init__(
        self,
        *,
        version_repository: CVVersionRepository,
        artifact_storage: ArtifactStorage,
        renderer: CVTemplateRenderer,
    ) -> None:
        """Store rendering dependencies."""

        self._version_repository = version_repository

        self._artifact_storage = artifact_storage

        self._renderer = renderer

    def render_and_store(
        self,
        *,
        user_id: str,
        cv_version_id: str,
    ) -> CVArtifactRenderingResult:
        """Render one version and atomically bridge storage with metadata."""

        version = self._version_repository.get(
            user_id=user_id,
            cv_version_id=cv_version_id,
        )

        if version is None:
            raise CVRenderingError("The CV version is unavailable.")

        rendered = self._renderer.render(version=version)

        digest = sha256(rendered.data).hexdigest()

        artifact_id = build_rendered_artifact_id(
            cv_version_id=cv_version_id,
            artifact_format=(rendered.artifact_format),
            sha256_hex=digest,
        )

        write_result = self._artifact_storage.save(
            user_id=user_id,
            cv_version_id=cv_version_id,
            artifact_id=artifact_id,
            artifact_format=(rendered.artifact_format),
            data=rendered.data,
        )

        candidate = RenderedCVArtifact(
            artifact_id=artifact_id,
            artifact_format=(rendered.artifact_format),
            storage_key=(write_result.storage_key),
            sha256_hex=digest,
            size_bytes=len(rendered.data),
            verification_status=(CVArtifactVerificationStatus.PENDING),
            verification_notes=[],
        )

        try:
            persisted_version = self._version_repository.attach_artifact(
                user_id=user_id,
                cv_version_id=cv_version_id,
                artifact=candidate,
            )

        except Exception:
            if write_result.created:
                self._delete_compensating_artifact(
                    user_id=user_id,
                    storage_key=(write_result.storage_key),
                )

            raise

        persisted_artifact = next(
            (
                artifact
                for artifact in persisted_version.artifacts
                if artifact.artifact_format is rendered.artifact_format
            ),
            None,
        )

        if persisted_artifact is None:
            raise RuntimeError("Persisted CV artifact became unavailable.")

        if not same_rendered_file_identity(
            persisted=persisted_artifact,
            candidate=candidate,
        ):
            raise RuntimeError(
                "Persisted CV artifact identity does not match rendered output."
            )

        return CVArtifactRenderingResult(
            version=persisted_version,
            artifact=persisted_artifact,
        )

    def _delete_compensating_artifact(
        self,
        *,
        user_id: str,
        storage_key: str,
    ) -> None:
        """Remove a newly written file when metadata persistence fails."""

        with suppress(FileNotFoundError):
            self._artifact_storage.delete(
                user_id=user_id,
                storage_key=storage_key,
            )


def build_rendered_artifact_id(
    *,
    cv_version_id: str,
    artifact_format: CVArtifactFormat,
    sha256_hex: str,
) -> str:
    """Build stable artifact identity from version, format and bytes."""

    payload = f"{cv_version_id}:{artifact_format.value}:{sha256_hex}"

    digest = sha256(payload.encode("utf-8")).hexdigest()[:16].upper()

    return f"ART-{digest}"


def same_rendered_file_identity(
    *,
    persisted: RenderedCVArtifact,
    candidate: RenderedCVArtifact,
) -> bool:
    """Compare immutable stored-file identity while ignoring verification."""

    return (
        persisted.artifact_id == candidate.artifact_id
        and persisted.artifact_format is candidate.artifact_format
        and persisted.storage_key == candidate.storage_key
        and persisted.sha256_hex == candidate.sha256_hex
        and persisted.size_bytes == candidate.size_bytes
    )
