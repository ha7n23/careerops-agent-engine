"""Tests for resumable complete final-CV generation."""

from careerops_agent_engine.application.exceptions import (
    FinalCVGenerationError,
)
from careerops_agent_engine.application.services.cv_artifact_rendering import (
    CVArtifactRenderingResult,
)
from careerops_agent_engine.application.services.cv_artifact_verification import (
    CVArtifactVerificationExecutionResult,
)
from careerops_agent_engine.application.services.cv_pdf_conversion import (
    CVPDFConversionExecutionResult,
)
from careerops_agent_engine.application.services.cv_version_builder import (
    CVVersionBuilder,
)
from careerops_agent_engine.application.services.final_cv_assembly import (
    FinalCVAssemblyExecutionResult,
)
from careerops_agent_engine.application.services.final_cv_generation import (
    FinalCVGenerationService,
)
from careerops_agent_engine.domain.enums import (
    ApprovalStatus,
    CareerDocumentFormat,
    CareerDocumentStatus,
    CVArtifactFormat,
    CVArtifactVerificationStatus,
    CVChangeApplicationMode,
    CVSection,
    CVVersionStatus,
    JobAnalysisRunStatus,
)
from careerops_agent_engine.domain.models.audit import (
    JobAnalysisRunSnapshot,
)
from careerops_agent_engine.domain.models.cv import (
    CVChangeProposal,
)
from careerops_agent_engine.domain.models.cv_application import (
    AppliedCVChange,
    StructuredCVTailoringResult,
)
from careerops_agent_engine.domain.models.cv_version import (
    CVVersion,
    RenderedCVArtifact,
)
from careerops_agent_engine.domain.models.document import (
    CareerDocument,
)
from careerops_agent_engine.domain.models.structured_cv import (
    StructuredCV,
    StructuredCVSection,
)


def build_assembly(
    *,
    thread_id: str = "THR-001",
    job_id: str = "JOB-001",
    proposal_id: str = "CVP-001",
) -> FinalCVAssemblyExecutionResult:
    """Create trusted deterministic final-CV assembly."""

    proposal = CVChangeProposal(
        proposal_id=proposal_id,
        section=CVSection.PROJECTS,
        proposed_text=("Built CareerOps using Python and FastAPI."),
        requirement_ids=[
            "REQ-001",
        ],
        supporting_evidence_ids=[
            "EVD-001",
        ],
        confidence_score=1.0,
    )

    change = AppliedCVChange(
        change_id=f"CHG-{proposal_id}",
        proposal_id=proposal_id,
        section=CVSection.PROJECTS,
        application_mode=(CVChangeApplicationMode.ANCHORED_REPLACEMENT),
        source_anchor=("Built CareerOps using Python."),
        original_text=("Built CareerOps using Python."),
        applied_text=("Built CareerOps using Python and FastAPI."),
        anchor_evidence_ids=[
            "EVD-001",
        ],
        requirement_ids=[
            "REQ-001",
        ],
        supporting_evidence_ids=[
            "EVD-001",
        ],
    )

    return FinalCVAssemblyExecutionResult(
        job_run=JobAnalysisRunSnapshot(
            thread_id=thread_id,
            user_id="USER-001",
            job_id=job_id,
            status=(JobAnalysisRunStatus.COMPLETED),
            review_status=(ApprovalStatus.APPROVED),
            final_cv_proposals=[proposal],
        ),
        source_document=CareerDocument(
            document_id="DOC-001",
            original_filename="cv.docx",
            document_format=(CareerDocumentFormat.DOCX),
            media_type=(
                "application/vnd.openxmlformats-officedocument."
                "wordprocessingml.document"
            ),
            size_bytes=100,
            sha256_hex="a" * 64,
            storage_key=("documents/user/DOC-001.docx"),
            status=(CareerDocumentStatus.EXTRACTED),
        ),
        approved_evidence=(),
        tailoring_result=(
            StructuredCVTailoringResult(
                structured_cv=StructuredCV(
                    cv_id="CV-001",
                    source_document_id="DOC-001",
                    sections=[
                        StructuredCVSection(
                            section=(CVSection.PROJECTS),
                            heading="Projects",
                            free_text=(
                                "CareerOps\nBuilt CareerOps using Python and FastAPI."
                            ),
                        )
                    ],
                ),
                applied_changes=[change],
            )
        ),
    )


class FakeAssemblyService:
    """Return one trusted assembly result."""

    def __init__(
        self,
        assembly: FinalCVAssemblyExecutionResult,
    ) -> None:
        self.assembly = assembly
        self.call_count = 0

    def assemble(
        self,
        *,
        user_id: str,
        thread_id: str,
        source_document_id: str,
    ) -> FinalCVAssemblyExecutionResult:
        """Return the configured user workflow."""

        self.call_count += 1

        assert user_id == "USER-001"
        assert thread_id == self.assembly.job_run.thread_id
        assert source_document_id == "DOC-001"

        return self.assembly


class FakeVersionRepository:
    """In-memory full CV-version repository protocol."""

    def __init__(self) -> None:
        self.versions: dict[
            str,
            CVVersion,
        ] = {}

    def save(
        self,
        *,
        user_id: str,
        version: CVVersion,
    ) -> None:
        """Persist an immutable version."""

        assert user_id == "USER-001"

        existing = self.versions.get(version.cv_version_id)

        if existing is not None and existing != version:
            raise ValueError("Version collision.")

        self.versions[version.cv_version_id] = version

    def get(
        self,
        *,
        user_id: str,
        cv_version_id: str,
    ) -> CVVersion | None:
        """Return the user's version."""

        if user_id != "USER-001":
            return None

        return self.versions.get(cv_version_id)

    def list_for_cv(
        self,
        *,
        user_id: str,
        cv_id: str,
    ) -> list[CVVersion]:
        """Return ordered family versions."""

        if user_id != "USER-001":
            return []

        return sorted(
            (
                version
                for version in self.versions.values()
                if version.structured_cv.cv_id == cv_id
            ),
            key=lambda version: version.version_number,
        )

    def attach_artifact(
        self,
        *,
        user_id: str,
        cv_version_id: str,
        artifact: RenderedCVArtifact,
    ) -> CVVersion:
        """Attach one artifact."""

        version = self.get(
            user_id=user_id,
            cv_version_id=cv_version_id,
        )

        if version is None:
            raise ValueError("Version unavailable.")

        existing = [
            item
            for item in version.artifacts
            if item.artifact_format is artifact.artifact_format
        ]

        artifacts = (
            version.artifacts
            if existing
            else [
                *version.artifacts,
                artifact,
            ]
        )

        updated = version.model_copy(
            update={
                "status": (CVVersionStatus.RENDERED),
                "artifacts": artifacts,
            }
        )

        self.versions[cv_version_id] = updated

        return updated

    def set_artifact_verification(
        self,
        *,
        user_id: str,
        cv_version_id: str,
        artifact_id: str,
        verification_status: CVArtifactVerificationStatus,
        verification_notes: list[str],
    ) -> CVVersion:
        """Apply terminal artifact verification."""

        version = self.get(
            user_id=user_id,
            cv_version_id=cv_version_id,
        )

        if version is None:
            raise ValueError("Version unavailable.")

        artifacts = [
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
            for artifact in version.artifacts
        ]

        formats = {artifact.artifact_format for artifact in artifacts}

        all_verified = all(
            artifact.verification_status is CVArtifactVerificationStatus.VERIFIED
            for artifact in artifacts
        )

        status = (
            CVVersionStatus.VERIFIED
            if (
                formats
                == {
                    CVArtifactFormat.DOCX,
                    CVArtifactFormat.PDF,
                }
                and all_verified
            )
            else CVVersionStatus.RENDERED
        )

        updated = version.model_copy(
            update={
                "status": status,
                "artifacts": artifacts,
            }
        )

        self.versions[cv_version_id] = updated

        return updated


class FakeDocxRenderer:
    """Attach one pending DOCX."""

    def __init__(
        self,
        repository: FakeVersionRepository,
    ) -> None:
        self.repository = repository
        self.call_count = 0

    def render_and_store(
        self,
        *,
        user_id: str,
        cv_version_id: str,
    ) -> CVArtifactRenderingResult:
        """Attach deterministic pending DOCX metadata."""

        self.call_count += 1

        artifact = RenderedCVArtifact(
            artifact_id=(f"ART-DOCX-{cv_version_id}"),
            artifact_format=(CVArtifactFormat.DOCX),
            storage_key=(f"artifacts/{cv_version_id}.docx"),
            sha256_hex="b" * 64,
            size_bytes=100,
            verification_status=(CVArtifactVerificationStatus.PENDING),
            verification_notes=[],
        )

        version = self.repository.attach_artifact(
            user_id=user_id,
            cv_version_id=cv_version_id,
            artifact=artifact,
        )

        return CVArtifactRenderingResult(
            version=version,
            artifact=artifact,
        )


class FakeVerifier:
    """Verify exactly one artifact format."""

    def __init__(
        self,
        *,
        repository: FakeVersionRepository,
        artifact_format: CVArtifactFormat,
    ) -> None:
        self.repository = repository
        self.artifact_format = artifact_format
        self.call_count = 0

    def verify(
        self,
        *,
        user_id: str,
        cv_version_id: str,
    ) -> CVArtifactVerificationExecutionResult:
        """Mark the configured artifact verified."""

        self.call_count += 1

        version = self.repository.get(
            user_id=user_id,
            cv_version_id=cv_version_id,
        )

        assert version is not None

        artifact = next(
            item
            for item in version.artifacts
            if item.artifact_format is self.artifact_format
        )

        version = self.repository.set_artifact_verification(
            user_id=user_id,
            cv_version_id=cv_version_id,
            artifact_id=(artifact.artifact_id),
            verification_status=(CVArtifactVerificationStatus.VERIFIED),
            verification_notes=[],
        )

        persisted_artifact = next(
            item
            for item in version.artifacts
            if item.artifact_id == artifact.artifact_id
        )

        return CVArtifactVerificationExecutionResult(
            version=version,
            artifact=persisted_artifact,
            passed=True,
            notes=(),
        )


class FakePDFConverter:
    """Attach one pending PDF."""

    def __init__(
        self,
        repository: FakeVersionRepository,
    ) -> None:
        self.repository = repository
        self.call_count = 0

    def convert_and_store(
        self,
        *,
        user_id: str,
        cv_version_id: str,
    ) -> CVPDFConversionExecutionResult:
        """Attach deterministic PDF metadata."""

        self.call_count += 1

        artifact = RenderedCVArtifact(
            artifact_id=(f"ART-PDF-{cv_version_id}"),
            artifact_format=(CVArtifactFormat.PDF),
            storage_key=(f"artifacts/{cv_version_id}.pdf"),
            sha256_hex="c" * 64,
            size_bytes=100,
            verification_status=(CVArtifactVerificationStatus.PENDING),
            verification_notes=[],
        )

        version = self.repository.attach_artifact(
            user_id=user_id,
            cv_version_id=cv_version_id,
            artifact=artifact,
        )

        return CVPDFConversionExecutionResult(
            version=version,
            artifact=artifact,
            reused_existing=False,
        )


def build_service(
    *,
    assembly: FinalCVAssemblyExecutionResult,
    repository: FakeVersionRepository,
) -> tuple[
    FinalCVGenerationService,
    FakeDocxRenderer,
    FakeVerifier,
    FakePDFConverter,
    FakeVerifier,
]:
    """Create final generation around observable fake stages."""

    docx_renderer = FakeDocxRenderer(repository)

    docx_verifier = FakeVerifier(
        repository=repository,
        artifact_format=(CVArtifactFormat.DOCX),
    )

    pdf_converter = FakePDFConverter(repository)

    pdf_verifier = FakeVerifier(
        repository=repository,
        artifact_format=(CVArtifactFormat.PDF),
    )

    service = FinalCVGenerationService(
        assembly_service=(FakeAssemblyService(assembly)),
        version_builder=CVVersionBuilder(),
        version_repository=repository,
        docx_rendering_service=(docx_renderer),
        docx_verification_service=(docx_verifier),
        pdf_conversion_service=(pdf_converter),
        pdf_verification_service=(pdf_verifier),
        template_id="careerops-standard",
        template_version="1.0.0",
        workflow_version="final-cv-v1",
        llm_references=[],
    )

    return (
        service,
        docx_renderer,
        docx_verifier,
        pdf_converter,
        pdf_verifier,
    )


def test_generate_creates_and_verifies_version_one() -> None:
    """First generation should complete DOCX and PDF lifecycle."""

    repository = FakeVersionRepository()

    (
        service,
        docx_renderer,
        docx_verifier,
        pdf_converter,
        pdf_verifier,
    ) = build_service(
        assembly=build_assembly(),
        repository=repository,
    )

    result = service.generate(
        user_id="USER-001",
        thread_id="THR-001",
        source_document_id="DOC-001",
    )

    assert result.reused_existing_version is False

    assert result.version.version_number == 1
    assert result.version.parent_version_id is None

    assert result.version.status is CVVersionStatus.VERIFIED

    assert docx_renderer.call_count == 1
    assert docx_verifier.call_count == 1
    assert pdf_converter.call_count == 1
    assert pdf_verifier.call_count == 1


def test_retry_reuses_verified_version_without_reprocessing() -> None:
    """Exact retry should recover the existing verified version."""

    repository = FakeVersionRepository()

    (
        service,
        docx_renderer,
        docx_verifier,
        pdf_converter,
        pdf_verifier,
    ) = build_service(
        assembly=build_assembly(),
        repository=repository,
    )

    first = service.generate(
        user_id="USER-001",
        thread_id="THR-001",
        source_document_id="DOC-001",
    )

    second = service.generate(
        user_id="USER-001",
        thread_id="THR-001",
        source_document_id="DOC-001",
    )

    assert second.reused_existing_version is True

    assert second.version.cv_version_id == first.version.cv_version_id

    assert docx_renderer.call_count == 1
    assert docx_verifier.call_count == 1
    assert pdf_converter.call_count == 1
    assert pdf_verifier.call_count == 1


def test_new_job_creates_next_version_in_same_cv_family() -> None:
    """A different accepted job run should create the next CV version."""

    repository = FakeVersionRepository()

    first_service, *_ = build_service(
        assembly=build_assembly(),
        repository=repository,
    )

    first = first_service.generate(
        user_id="USER-001",
        thread_id="THR-001",
        source_document_id="DOC-001",
    )

    second_service, *_ = build_service(
        assembly=build_assembly(
            thread_id="THR-002",
            job_id="JOB-002",
            proposal_id="CVP-002",
        ),
        repository=repository,
    )

    second = second_service.generate(
        user_id="USER-001",
        thread_id="THR-002",
        source_document_id="DOC-001",
    )

    assert second.version.version_number == 2

    assert second.version.parent_version_id == first.version.cv_version_id

    assert second.version.cv_version_id != first.version.cv_version_id


def test_failed_existing_artifact_cannot_be_replaced() -> None:
    """Resume must stop if an immutable artifact already failed."""

    repository = FakeVersionRepository()

    assembly = build_assembly()

    version = CVVersionBuilder().build(
        assembly=assembly,
        version_number=1,
        parent_version_id=None,
        template_id="careerops-standard",
        template_version="1.0.0",
        workflow_version="final-cv-v1",
        llm_references=[],
    )

    failed_docx = RenderedCVArtifact(
        artifact_id="ART-FAILED",
        artifact_format=(CVArtifactFormat.DOCX),
        storage_key="artifacts/failed.docx",
        sha256_hex="d" * 64,
        size_bytes=100,
        verification_status=(CVArtifactVerificationStatus.FAILED),
        verification_notes=["DOCX verification failed."],
    )

    repository.versions[version.cv_version_id] = version.model_copy(
        update={
            "status": (CVVersionStatus.RENDERED),
            "artifacts": [failed_docx],
        }
    )

    service, *_ = build_service(
        assembly=assembly,
        repository=repository,
    )

    try:
        service.generate(
            user_id="USER-001",
            thread_id="THR-001",
            source_document_id="DOC-001",
        )

    except FinalCVGenerationError as exc:
        assert "failed immutable artifact" in str(exc)

    else:
        raise AssertionError("Failed immutable artifact should block generation.")
