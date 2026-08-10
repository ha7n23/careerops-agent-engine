"""Tests for verified-DOCX to persisted-PDF orchestration."""

from hashlib import sha256
from pathlib import Path

import pytest

from careerops_agent_engine.application.exceptions import (
    CVPDFConversionError,
)
from careerops_agent_engine.application.ports.pdf_converter import (
    ConvertedPDFDocument,
)
from careerops_agent_engine.application.services.cv_pdf_conversion import (
    CVPDFConversionService,
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

DOCX_BYTES = b"verified-docx"
PDF_BYTES = b"%PDF-test-pdf-bytes"


def build_artifact(
    *,
    artifact_format: CVArtifactFormat,
    artifact_id: str,
    storage_key: str,
    data: bytes,
    verification_status: CVArtifactVerificationStatus,
) -> RenderedCVArtifact:
    """Create immutable persisted artifact metadata."""

    return RenderedCVArtifact(
        artifact_id=artifact_id,
        artifact_format=artifact_format,
        storage_key=storage_key,
        sha256_hex=sha256(data).hexdigest(),
        size_bytes=len(data),
        verification_status=verification_status,
        verification_notes=[],
    )


def build_version(
    *,
    artifacts: list[RenderedCVArtifact],
) -> CVVersion:
    """Create one persisted rendered CV version."""

    return CVVersion(
        cv_version_id="CVV-001",
        version_number=1,
        status=CVVersionStatus.RENDERED,
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
        artifacts=artifacts,
    )


class FakePDFConverter:
    """Return deterministic PDF-like bytes."""

    def __init__(self) -> None:
        self.call_count = 0

    def convert_docx(
        self,
        *,
        docx_data: bytes,
        source_filename: str,
    ) -> ConvertedPDFDocument:
        """Convert only the expected verified source."""

        self.call_count += 1

        assert docx_data == DOCX_BYTES
        assert source_filename == ("CVV-001.docx")

        return ConvertedPDFDocument(
            filename="CVV-001.pdf",
            media_type="application/pdf",
            data=PDF_BYTES,
        )


class FakeVersionRepository:
    """Full repository-protocol fake for PDF orchestration."""

    def __init__(
        self,
        version: CVVersion | None,
        *,
        fail_attach: bool = False,
    ) -> None:
        self.version = version
        self.fail_attach = fail_attach

    def save(
        self,
        *,
        user_id: str,
        version: CVVersion,
    ) -> None:
        """Saving versions is outside this test boundary."""

        del user_id
        del version

        raise AssertionError("PDF conversion must not save CV versions.")

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

        raise AssertionError("PDF conversion must not list versions.")

    def attach_artifact(
        self,
        *,
        user_id: str,
        cv_version_id: str,
        artifact: RenderedCVArtifact,
    ) -> CVVersion:
        """Attach one PDF or simulate metadata failure."""

        if self.fail_attach:
            raise ValueError("Simulated PDF metadata failure.")

        if (
            user_id != "USER-001"
            or self.version is None
            or self.version.cv_version_id != cv_version_id
        ):
            raise ValueError("CV version is unavailable.")

        self.version = self.version.model_copy(
            update={
                "status": (CVVersionStatus.RENDERED),
                "artifacts": [
                    *self.version.artifacts,
                    artifact,
                ],
            }
        )

        return self.version

    def set_artifact_verification(
        self,
        *,
        user_id: str,
        cv_version_id: str,
        artifact_id: str,
        verification_status: CVArtifactVerificationStatus,
        verification_notes: list[str],
    ) -> CVVersion:
        """Verification belongs to the later PDF-verification step."""

        del user_id
        del cv_version_id
        del artifact_id
        del verification_status
        del verification_notes

        raise AssertionError("PDF conversion must not verify artifacts.")


def prepare_version_and_storage(
    *,
    tmp_path: Path,
    docx_status: CVArtifactVerificationStatus = (CVArtifactVerificationStatus.VERIFIED),
) -> tuple[
    CVVersion,
    LocalArtifactStorage,
]:
    """Persist source DOCX bytes and build matching metadata."""

    storage = LocalArtifactStorage(tmp_path)

    write = storage.save(
        user_id="USER-001",
        cv_version_id="CVV-001",
        artifact_id="ART-DOCX",
        artifact_format=CVArtifactFormat.DOCX,
        data=DOCX_BYTES,
    )

    docx = build_artifact(
        artifact_format=CVArtifactFormat.DOCX,
        artifact_id="ART-DOCX",
        storage_key=write.storage_key,
        data=DOCX_BYTES,
        verification_status=docx_status,
    )

    return (
        build_version(artifacts=[docx]),
        storage,
    )


def test_verified_docx_is_converted_stored_and_attached(
    tmp_path: Path,
) -> None:
    """A verified DOCX should produce one pending PDF artifact."""

    version, storage = prepare_version_and_storage(tmp_path=tmp_path)

    converter = FakePDFConverter()

    repository = FakeVersionRepository(version)

    result = CVPDFConversionService(
        version_repository=repository,
        artifact_storage=storage,
        converter=converter,
    ).convert_and_store(
        user_id="USER-001",
        cv_version_id="CVV-001",
    )

    assert converter.call_count == 1

    assert result.reused_existing is False

    assert result.artifact.artifact_format is CVArtifactFormat.PDF

    assert result.artifact.verification_status is CVArtifactVerificationStatus.PENDING

    assert (
        storage.read(
            user_id="USER-001",
            storage_key=(result.artifact.storage_key),
        )
        == PDF_BYTES
    )


def test_unverified_docx_cannot_be_converted(
    tmp_path: Path,
) -> None:
    """PDF creation cannot bypass DOCX verification."""

    version, storage = prepare_version_and_storage(
        tmp_path=tmp_path,
        docx_status=(CVArtifactVerificationStatus.PENDING),
    )

    converter = FakePDFConverter()

    with pytest.raises(
        CVPDFConversionError,
        match="verified DOCX",
    ):
        CVPDFConversionService(
            version_repository=(FakeVersionRepository(version)),
            artifact_storage=storage,
            converter=converter,
        ).convert_and_store(
            user_id="USER-001",
            cv_version_id="CVV-001",
        )

    assert converter.call_count == 0


def test_existing_pdf_is_reused_without_reconversion(
    tmp_path: Path,
) -> None:
    """An existing immutable PDF should short-circuit LibreOffice."""

    version, storage = prepare_version_and_storage(tmp_path=tmp_path)

    pdf_write = storage.save(
        user_id="USER-001",
        cv_version_id="CVV-001",
        artifact_id="ART-PDF",
        artifact_format=CVArtifactFormat.PDF,
        data=PDF_BYTES,
    )

    pdf = build_artifact(
        artifact_format=CVArtifactFormat.PDF,
        artifact_id="ART-PDF",
        storage_key=pdf_write.storage_key,
        data=PDF_BYTES,
        verification_status=(CVArtifactVerificationStatus.PENDING),
    )

    version = version.model_copy(
        update={
            "artifacts": [
                *version.artifacts,
                pdf,
            ]
        }
    )

    converter = FakePDFConverter()

    result = CVPDFConversionService(
        version_repository=(FakeVersionRepository(version)),
        artifact_storage=storage,
        converter=converter,
    ).convert_and_store(
        user_id="USER-001",
        cv_version_id="CVV-001",
    )

    assert converter.call_count == 0
    assert result.reused_existing is True
    assert result.artifact == pdf


def test_metadata_failure_removes_newly_created_pdf(
    tmp_path: Path,
) -> None:
    """Failed PDF metadata persistence should compensate new file bytes."""

    version, storage = prepare_version_and_storage(tmp_path=tmp_path)

    converter = FakePDFConverter()

    with pytest.raises(
        ValueError,
        match="Simulated PDF metadata failure",
    ):
        CVPDFConversionService(
            version_repository=(
                FakeVersionRepository(
                    version,
                    fail_attach=True,
                )
            ),
            artifact_storage=storage,
            converter=converter,
        ).convert_and_store(
            user_id="USER-001",
            cv_version_id="CVV-001",
        )

    pdf_files = list(tmp_path.rglob("*.pdf"))

    assert pdf_files == []


def test_missing_version_fails_before_conversion(
    tmp_path: Path,
) -> None:
    """Unknown or cross-user versions cannot produce PDFs."""

    converter = FakePDFConverter()

    with pytest.raises(
        CVPDFConversionError,
        match="version is unavailable",
    ):
        CVPDFConversionService(
            version_repository=(FakeVersionRepository(None)),
            artifact_storage=(LocalArtifactStorage(tmp_path)),
            converter=converter,
        ).convert_and_store(
            user_id="USER-001",
            cv_version_id="CVV-MISSING",
        )

    assert converter.call_count == 0
