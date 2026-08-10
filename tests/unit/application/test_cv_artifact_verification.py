"""Tests for persisted generated-artifact verification."""

from hashlib import sha256
from pathlib import Path

import pytest

from careerops_agent_engine.application.exceptions import (
    CVArtifactVerificationError,
)
from careerops_agent_engine.application.ports.cv_artifact_verifier import (
    CVArtifactVerificationResult,
)
from careerops_agent_engine.application.services.cv_artifact_verification import (
    CVArtifactVerificationService,
)
from careerops_agent_engine.domain.enums import (
    ApprovalStatus,
    CVArtifactFormat,
    CVArtifactVerificationStatus,
    CVChangeApplicationMode,
    CVSection,
    CVVersionStatus,
)
from careerops_agent_engine.domain.models.cv_application import (
    AppliedCVChange,
)
from careerops_agent_engine.domain.models.cv_version import (
    CVVersion,
    CVVersionProvenance,
    RenderedCVArtifact,
)
from careerops_agent_engine.domain.models.structured_cv import (
    StructuredCV,
    StructuredCVSection,
)
from careerops_agent_engine.infrastructure.storage.local_artifact_storage import (
    LocalArtifactStorage,
)

DOCX_BYTES = b"stored-docx-bytes"


def build_version(
    *,
    artifact: RenderedCVArtifact | None = None,
) -> CVVersion:
    """Create one rendered CV version."""

    return CVVersion(
        cv_version_id="CVV-001",
        version_number=1,
        status=(
            CVVersionStatus.RENDERED
            if artifact is not None
            else CVVersionStatus.ASSEMBLED
        ),
        structured_cv=StructuredCV(
            cv_id="CV-001",
            source_document_id="DOC-001",
            sections=[
                StructuredCVSection(
                    section=CVSection.PROJECTS,
                    heading="Projects",
                    free_text="CareerOps",
                )
            ],
        ),
        applied_changes=[
            AppliedCVChange(
                change_id="CHG-001",
                proposal_id="CVP-001",
                section=CVSection.PROJECTS,
                application_mode=(CVChangeApplicationMode.ANCHORED_REPLACEMENT),
                source_anchor="Original",
                original_text="Original",
                applied_text="CareerOps",
                anchor_evidence_ids=["EVD-001"],
                requirement_ids=["REQ-001"],
                supporting_evidence_ids=["EVD-001"],
            )
        ],
        provenance=CVVersionProvenance(
            job_id="JOB-001",
            thread_id="THR-001",
            source_document_id="DOC-001",
            requirement_ids=["REQ-001"],
            supporting_evidence_ids=["EVD-001"],
            final_proposal_ids=["CVP-001"],
            review_status=(ApprovalStatus.APPROVED),
            template_id="careerops-standard",
            template_version="1.0.0",
            workflow_version="1.0.0",
        ),
        artifacts=([artifact] if artifact is not None else []),
    )


def build_artifact(
    *,
    sha256_hex: str | None = None,
    size_bytes: int | None = None,
) -> RenderedCVArtifact:
    """Create persisted pending DOCX metadata."""

    return RenderedCVArtifact(
        artifact_id="ART-001",
        artifact_format=CVArtifactFormat.DOCX,
        storage_key=("usr-test/CVV-001/ART-001.docx"),
        sha256_hex=(
            sha256_hex if sha256_hex is not None else sha256(DOCX_BYTES).hexdigest()
        ),
        size_bytes=(size_bytes if size_bytes is not None else len(DOCX_BYTES)),
        verification_status=(CVArtifactVerificationStatus.PENDING),
        verification_notes=[],
    )


class PassingVerifier:
    """Accept bytes after generic integrity checks."""

    @property
    def artifact_format(self) -> CVArtifactFormat:
        """Verify DOCX artifacts."""

        return CVArtifactFormat.DOCX

    def verify(
        self,
        *,
        version: CVVersion,
        data: bytes,
    ) -> CVArtifactVerificationResult:
        """Return deterministic success."""

        del version

        assert data == DOCX_BYTES

        return CVArtifactVerificationResult(passed=True)


class FakeVersionRepository:
    """Full protocol fake for verification orchestration."""

    def __init__(
        self,
        version: CVVersion | None,
    ) -> None:
        self.version = version

    def save(
        self,
        *,
        user_id: str,
        version: CVVersion,
    ) -> None:
        """Saving versions is outside this test boundary."""

        del user_id
        del version

        raise AssertionError("Verification must not save CV versions.")

    def get(
        self,
        *,
        user_id: str,
        cv_version_id: str,
    ) -> CVVersion | None:
        """Return the owned fake version."""

        if (
            user_id != "USER-001"
            or self.version is None
            or self.version.cv_version_id != cv_version_id
        ):
            return None

        return self.version

    def list_for_cv(
        self,
        *,
        user_id: str,
        cv_id: str,
    ) -> list[CVVersion]:
        """Listing is outside this test boundary."""

        del user_id
        del cv_id

        raise AssertionError("Verification must not list CV versions.")

    def attach_artifact(
        self,
        *,
        user_id: str,
        cv_version_id: str,
        artifact: RenderedCVArtifact,
    ) -> CVVersion:
        """Rendering attachment is outside this test boundary."""

        del user_id
        del cv_version_id
        del artifact

        raise AssertionError("Verification must not attach artifacts.")

    def set_artifact_verification(
        self,
        *,
        user_id: str,
        cv_version_id: str,
        artifact_id: str,
        verification_status: CVArtifactVerificationStatus,
        verification_notes: list[str],
    ) -> CVVersion:
        """Persist the terminal verification result."""

        if (
            user_id != "USER-001"
            or self.version is None
            or self.version.cv_version_id != cv_version_id
        ):
            raise ValueError("CV version is unavailable.")

        updated_artifacts = [
            artifact.model_copy(
                update={
                    "verification_status": (
                        verification_status
                        if artifact.artifact_id == artifact_id
                        else artifact.verification_status
                    ),
                    "verification_notes": (
                        verification_notes
                        if artifact.artifact_id == artifact_id
                        else artifact.verification_notes
                    ),
                }
            )
            for artifact in self.version.artifacts
        ]

        self.version = self.version.model_copy(
            update={"artifacts": (updated_artifacts)}
        )

        return self.version


def test_verification_service_marks_valid_artifact_verified(
    tmp_path: Path,
) -> None:
    """A valid immutable artifact should receive VERIFIED status."""

    storage = LocalArtifactStorage(tmp_path)

    write = storage.save(
        user_id="USER-001",
        cv_version_id="CVV-001",
        artifact_id="ART-001",
        artifact_format=CVArtifactFormat.DOCX,
        data=DOCX_BYTES,
    )

    artifact = build_artifact().model_copy(update={"storage_key": (write.storage_key)})

    repository = FakeVersionRepository(build_version(artifact=artifact))

    result = CVArtifactVerificationService(
        version_repository=repository,
        artifact_storage=storage,
        verifier=PassingVerifier(),
    ).verify(
        user_id="USER-001",
        cv_version_id="CVV-001",
    )

    assert result.passed is True

    assert result.artifact.verification_status is CVArtifactVerificationStatus.VERIFIED


def test_checksum_mismatch_marks_artifact_failed(
    tmp_path: Path,
) -> None:
    """Stored bytes inconsistent with metadata must fail before parsing."""

    storage = LocalArtifactStorage(tmp_path)

    write = storage.save(
        user_id="USER-001",
        cv_version_id="CVV-001",
        artifact_id="ART-001",
        artifact_format=CVArtifactFormat.DOCX,
        data=DOCX_BYTES,
    )

    artifact = build_artifact(sha256_hex="a" * 64).model_copy(
        update={"storage_key": (write.storage_key)}
    )

    repository = FakeVersionRepository(build_version(artifact=artifact))

    result = CVArtifactVerificationService(
        version_repository=repository,
        artifact_storage=storage,
        verifier=PassingVerifier(),
    ).verify(
        user_id="USER-001",
        cv_version_id="CVV-001",
    )

    assert result.passed is False

    assert result.artifact.verification_status is CVArtifactVerificationStatus.FAILED

    assert "checksum" in result.notes[0]


def test_missing_artifact_fails_before_storage_read(
    tmp_path: Path,
) -> None:
    """A CV version without the requested format cannot be verified."""

    service = CVArtifactVerificationService(
        version_repository=(FakeVersionRepository(build_version())),
        artifact_storage=(LocalArtifactStorage(tmp_path)),
        verifier=PassingVerifier(),
    )

    with pytest.raises(
        CVArtifactVerificationError,
        match="artifact is unavailable",
    ):
        service.verify(
            user_id="USER-001",
            cv_version_id="CVV-001",
        )


def test_missing_version_is_user_safe(
    tmp_path: Path,
) -> None:
    """Unknown or cross-user versions must not leak artifact state."""

    service = CVArtifactVerificationService(
        version_repository=(FakeVersionRepository(None)),
        artifact_storage=(LocalArtifactStorage(tmp_path)),
        verifier=PassingVerifier(),
    )

    with pytest.raises(
        CVArtifactVerificationError,
        match="version is unavailable",
    ):
        service.verify(
            user_id="USER-001",
            cv_version_id="CVV-MISSING",
        )
