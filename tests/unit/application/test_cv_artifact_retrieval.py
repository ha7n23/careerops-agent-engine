"""Tests for secure generated-CV artifact retrieval."""

from hashlib import sha256
from pathlib import Path

import pytest

from careerops_agent_engine.application.exceptions import (
    CVArtifactRetrievalError,
)
from careerops_agent_engine.application.services.cv_artifact_retrieval import (
    CVArtifactRetrievalService,
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

ARTIFACT_BYTES = b"verified-artifact-bytes"


def build_version(
    *,
    artifact: RenderedCVArtifact | None = None,
) -> CVVersion:
    """Create one retrievable CV version."""

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
            workflow_version="final-cv-v1",
        ),
        artifacts=([artifact] if artifact is not None else []),
    )


class FakeVersionRepository:
    """Full protocol fake for retrieval tests."""

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
        """Writes are outside retrieval."""

        del user_id
        del version

        raise AssertionError("Retrieval must not save versions.")

    def get(
        self,
        *,
        user_id: str,
        cv_version_id: str,
    ) -> CVVersion | None:
        """Return only the owned version."""

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
        """Listing is outside this test."""

        del user_id
        del cv_id

        raise AssertionError("Retrieval must not list versions.")

    def attach_artifact(
        self,
        *,
        user_id: str,
        cv_version_id: str,
        artifact: RenderedCVArtifact,
    ) -> CVVersion:
        """Writes are outside retrieval."""

        del user_id
        del cv_version_id
        del artifact

        raise AssertionError("Retrieval must not attach artifacts.")

    def set_artifact_verification(
        self,
        *,
        user_id: str,
        cv_version_id: str,
        artifact_id: str,
        verification_status: CVArtifactVerificationStatus,
        verification_notes: list[str],
    ) -> CVVersion:
        """Writes are outside retrieval."""

        del user_id
        del cv_version_id
        del artifact_id
        del verification_status
        del verification_notes

        raise AssertionError("Retrieval must not verify artifacts.")


def prepare_artifact(
    *,
    tmp_path: Path,
    verification_status: CVArtifactVerificationStatus = (
        CVArtifactVerificationStatus.VERIFIED
    ),
) -> tuple[
    RenderedCVArtifact,
    LocalArtifactStorage,
]:
    """Store bytes and create matching metadata."""

    storage = LocalArtifactStorage(tmp_path)

    write = storage.save(
        user_id="USER-001",
        cv_version_id="CVV-001",
        artifact_id="ART-001",
        artifact_format=CVArtifactFormat.DOCX,
        data=ARTIFACT_BYTES,
    )

    artifact = RenderedCVArtifact(
        artifact_id="ART-001",
        artifact_format=CVArtifactFormat.DOCX,
        storage_key=write.storage_key,
        sha256_hex=sha256(ARTIFACT_BYTES).hexdigest(),
        size_bytes=len(ARTIFACT_BYTES),
        verification_status=verification_status,
        verification_notes=[],
    )

    return artifact, storage


def test_get_version_enforces_user_boundary(
    tmp_path: Path,
) -> None:
    """Cross-user retrieval must look unavailable."""

    service = CVArtifactRetrievalService(
        version_repository=(FakeVersionRepository(build_version())),
        artifact_storage=(LocalArtifactStorage(tmp_path)),
    )

    with pytest.raises(
        CVArtifactRetrievalError,
        match="version is unavailable",
    ):
        service.get_version(
            user_id="USER-OTHER",
            cv_version_id="CVV-001",
        )


def test_verified_artifact_bytes_are_returned(
    tmp_path: Path,
) -> None:
    """Verified immutable bytes should be safe to retrieve."""

    artifact, storage = prepare_artifact(tmp_path=tmp_path)

    service = CVArtifactRetrievalService(
        version_repository=(FakeVersionRepository(build_version(artifact=artifact))),
        artifact_storage=storage,
    )

    result = service.get_verified_artifact(
        user_id="USER-001",
        cv_version_id="CVV-001",
        artifact_format=(CVArtifactFormat.DOCX),
    )

    assert result.data == ARTIFACT_BYTES
    assert result.filename == "CVV-001.docx"

    assert result.artifact.artifact_id == "ART-001"


def test_pending_artifact_cannot_be_downloaded(
    tmp_path: Path,
) -> None:
    """Downloads cannot bypass artifact verification."""

    artifact, storage = prepare_artifact(
        tmp_path=tmp_path,
        verification_status=(CVArtifactVerificationStatus.PENDING),
    )

    service = CVArtifactRetrievalService(
        version_repository=(FakeVersionRepository(build_version(artifact=artifact))),
        artifact_storage=storage,
    )

    with pytest.raises(
        CVArtifactRetrievalError,
        match="not verified",
    ):
        service.get_verified_artifact(
            user_id="USER-001",
            cv_version_id="CVV-001",
            artifact_format=(CVArtifactFormat.DOCX),
        )


def test_checksum_mismatch_blocks_download(
    tmp_path: Path,
) -> None:
    """Corrupted stored bytes must never be sent to the client."""

    artifact, storage = prepare_artifact(tmp_path=tmp_path)

    corrupted = artifact.model_copy(update={"sha256_hex": "a" * 64})

    service = CVArtifactRetrievalService(
        version_repository=(FakeVersionRepository(build_version(artifact=corrupted))),
        artifact_storage=storage,
    )

    with pytest.raises(
        CVArtifactRetrievalError,
        match="checksum",
    ):
        service.get_verified_artifact(
            user_id="USER-001",
            cv_version_id="CVV-001",
            artifact_format=(CVArtifactFormat.DOCX),
        )
