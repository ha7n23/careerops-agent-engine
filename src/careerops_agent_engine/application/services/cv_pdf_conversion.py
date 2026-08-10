"""Convert verified stored DOCX artifacts into persisted PDF artifacts."""

from contextlib import suppress
from dataclasses import dataclass
from hashlib import sha256

from careerops_agent_engine.application.exceptions import (
    CVPDFConversionError,
)
from careerops_agent_engine.application.ports.artifact_storage import (
    ArtifactStorage,
)
from careerops_agent_engine.application.ports.cv_version_repository import (
    CVVersionRepository,
)
from careerops_agent_engine.application.ports.pdf_converter import (
    PDFConverter,
)
from careerops_agent_engine.application.services.cv_artifact_rendering import (
    build_rendered_artifact_id,
    same_rendered_file_identity,
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
class CVPDFConversionExecutionResult:
    """Authoritative result of DOCX-to-PDF conversion."""

    version: CVVersion
    artifact: RenderedCVArtifact
    reused_existing: bool


class CVPDFConversionService:
    """Convert one verified DOCX into an immutable stored PDF artifact."""

    def __init__(
        self,
        *,
        version_repository: CVVersionRepository,
        artifact_storage: ArtifactStorage,
        converter: PDFConverter,
    ) -> None:
        """Store PDF-conversion dependencies."""

        self._version_repository = version_repository
        self._artifact_storage = artifact_storage
        self._converter = converter

    def convert_and_store(
        self,
        *,
        user_id: str,
        cv_version_id: str,
    ) -> CVPDFConversionExecutionResult:
        """Convert a verified DOCX unless a PDF already exists."""

        version = self._version_repository.get(
            user_id=user_id,
            cv_version_id=cv_version_id,
        )

        if version is None:
            raise CVPDFConversionError("The CV version is unavailable.")

        existing_pdf = find_artifact(
            version=version,
            artifact_format=CVArtifactFormat.PDF,
        )

        if existing_pdf is not None:
            self._validate_existing_artifact_bytes(
                user_id=user_id,
                artifact=existing_pdf,
            )

            return CVPDFConversionExecutionResult(
                version=version,
                artifact=existing_pdf,
                reused_existing=True,
            )

        docx = find_artifact(
            version=version,
            artifact_format=CVArtifactFormat.DOCX,
        )

        if docx is None:
            raise CVPDFConversionError(
                "A stored DOCX artifact is required before PDF conversion."
            )

        if docx.verification_status is not CVArtifactVerificationStatus.VERIFIED:
            raise CVPDFConversionError(
                "PDF conversion requires a verified DOCX artifact."
            )

        docx_data = self._read_verified_source_docx(
            user_id=user_id,
            artifact=docx,
        )

        converted = self._converter.convert_docx(
            docx_data=docx_data,
            source_filename=(f"{cv_version_id}.docx"),
        )

        digest = sha256(converted.data).hexdigest()

        artifact_id = build_rendered_artifact_id(
            cv_version_id=cv_version_id,
            artifact_format=CVArtifactFormat.PDF,
            sha256_hex=digest,
        )

        write_result = self._artifact_storage.save(
            user_id=user_id,
            cv_version_id=cv_version_id,
            artifact_id=artifact_id,
            artifact_format=CVArtifactFormat.PDF,
            data=converted.data,
        )

        candidate = RenderedCVArtifact(
            artifact_id=artifact_id,
            artifact_format=CVArtifactFormat.PDF,
            storage_key=write_result.storage_key,
            sha256_hex=digest,
            size_bytes=len(converted.data),
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
                self._delete_compensating_pdf(
                    user_id=user_id,
                    storage_key=(write_result.storage_key),
                )

            raise

        persisted_pdf = find_artifact(
            version=persisted_version,
            artifact_format=CVArtifactFormat.PDF,
        )

        if persisted_pdf is None:
            raise RuntimeError("Persisted PDF artifact became unavailable.")

        if not same_rendered_file_identity(
            persisted=persisted_pdf,
            candidate=candidate,
        ):
            raise RuntimeError(
                "Persisted PDF identity does not match the converted PDF output."
            )

        return CVPDFConversionExecutionResult(
            version=persisted_version,
            artifact=persisted_pdf,
            reused_existing=False,
        )

    def _read_verified_source_docx(
        self,
        *,
        user_id: str,
        artifact: RenderedCVArtifact,
    ) -> bytes:
        """Read the DOCX and re-check its immutable metadata."""

        try:
            data = self._artifact_storage.read(
                user_id=user_id,
                storage_key=artifact.storage_key,
            )

        except FileNotFoundError as exc:
            raise CVPDFConversionError(
                "The verified DOCX bytes are unavailable."
            ) from exc

        validate_artifact_bytes(
            artifact=artifact,
            data=data,
            label="DOCX",
        )

        return data

    def _validate_existing_artifact_bytes(
        self,
        *,
        user_id: str,
        artifact: RenderedCVArtifact,
    ) -> None:
        """Validate an existing PDF before reusing it."""

        try:
            data = self._artifact_storage.read(
                user_id=user_id,
                storage_key=artifact.storage_key,
            )

        except FileNotFoundError as exc:
            raise CVPDFConversionError(
                "The persisted PDF bytes are unavailable."
            ) from exc

        validate_artifact_bytes(
            artifact=artifact,
            data=data,
            label="PDF",
        )

    def _delete_compensating_pdf(
        self,
        *,
        user_id: str,
        storage_key: str,
    ) -> None:
        """Remove newly created PDF bytes after metadata failure."""

        with suppress(FileNotFoundError):
            self._artifact_storage.delete(
                user_id=user_id,
                storage_key=storage_key,
            )


def find_artifact(
    *,
    version: CVVersion,
    artifact_format: CVArtifactFormat,
) -> RenderedCVArtifact | None:
    """Find one artifact format in a CV version."""

    return next(
        (
            artifact
            for artifact in version.artifacts
            if artifact.artifact_format is artifact_format
        ),
        None,
    )


def validate_artifact_bytes(
    *,
    artifact: RenderedCVArtifact,
    data: bytes,
    label: str,
) -> None:
    """Check stored immutable bytes against persisted metadata."""

    if len(data) != artifact.size_bytes:
        raise CVPDFConversionError(
            f"Stored {label} size does not match persisted metadata."
        )

    digest = sha256(data).hexdigest()

    if digest != artifact.sha256_hex:
        raise CVPDFConversionError(
            f"Stored {label} checksum does not match persisted metadata."
        )
