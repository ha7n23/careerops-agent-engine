"""Live proof of the complete CareerOps DOCX and PDF artifact pipeline."""

import os
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from uuid import uuid4
from zipfile import ZipFile

import pytest
from docx import Document
from pypdf import PdfReader
from sqlalchemy import delete

from careerops_agent_engine.application.services.cv_artifact_rendering import (
    CVArtifactRenderingService,
)
from careerops_agent_engine.application.services.cv_artifact_verification import (
    CVArtifactVerificationService,
)
from careerops_agent_engine.application.services.cv_pdf_conversion import (
    CVPDFConversionService,
)
from careerops_agent_engine.core.config import get_settings
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
from careerops_agent_engine.domain.models.cv import (
    CVChangeProposal,
)
from careerops_agent_engine.domain.models.cv_application import (
    AppliedCVChange,
)
from careerops_agent_engine.domain.models.cv_version import (
    CVVersion,
    CVVersionProvenance,
)
from careerops_agent_engine.domain.models.structured_cv import (
    StructuredCV,
    StructuredCVSection,
)
from careerops_agent_engine.infrastructure.database.models.cv_evidence import (
    CareerDocumentRecord,
)
from careerops_agent_engine.infrastructure.database.models.cv_version import (
    CVArtifactRecord,
    CVVersionRecord,
)
from careerops_agent_engine.infrastructure.database.models.job_analysis import (
    JobAnalysisRunRecord,
)
from careerops_agent_engine.infrastructure.database.session import (
    create_database_engine,
    create_session_factory,
)
from careerops_agent_engine.infrastructure.documents.careerops_docx_renderer import (
    CareerOpsStandardDocxRenderer,
)
from careerops_agent_engine.infrastructure.documents.careerops_docx_verifier import (
    CareerOpsStandardDocxVerifier,
)
from careerops_agent_engine.infrastructure.documents.careerops_pdf_verifier import (
    CareerOpsStandardPDFVerifier,
)
from careerops_agent_engine.infrastructure.documents.libreoffice_pdf_converter import (
    LibreOfficePDFConverter,
)
from careerops_agent_engine.infrastructure.repositories.sqlalchemy_cv_versions import (
    SqlAlchemyCVVersionRepository,
)
from careerops_agent_engine.infrastructure.storage.local_artifact_storage import (
    LocalArtifactStorage,
)

pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    os.getenv("RUN_LIVE_DOCUMENT_TESTS") != "true",
    reason="Live document integration tests are disabled.",
)
def test_real_docx_to_verified_pdf_pipeline(
    tmp_path: Path,
) -> None:
    """Prove the full persisted CareerOps document-generation pipeline."""

    settings = get_settings()

    suffix = uuid4().hex[:8].upper()

    user_id = f"USER-LIVE-DOC-{suffix}"
    document_id = f"DOC-LIVE-{suffix}"
    thread_id = f"THR-LIVE-DOC-{suffix}"
    job_id = f"JOB-LIVE-DOC-{suffix}"
    cv_id = f"CV-LIVE-{suffix}"
    cv_version_id = f"CVV-LIVE-{suffix}"
    proposal_id = f"CVP-LIVE-{suffix}"
    change_id = f"CHG-LIVE-{suffix}"
    requirement_id = f"REQ-LIVE-{suffix}"
    evidence_id = f"EVD-LIVE-{suffix}"

    original_text = "Built CareerOps using Python."

    final_text = "Built CareerOps using Python and FastAPI."

    proposal = CVChangeProposal(
        proposal_id=proposal_id,
        section=CVSection.PROJECTS,
        current_text=original_text,
        proposed_text=final_text,
        requirement_ids=[
            requirement_id,
        ],
        supporting_evidence_ids=[
            evidence_id,
        ],
        confidence_score=1.0,
    )

    version = CVVersion(
        cv_version_id=cv_version_id,
        version_number=1,
        status=CVVersionStatus.ASSEMBLED,
        structured_cv=StructuredCV(
            cv_id=cv_id,
            source_document_id=document_id,
            preamble_text=("CareerOps Test Candidate\ncandidate@example.com"),
            sections=[
                StructuredCVSection(
                    section=CVSection.PROFILE,
                    heading="Profile",
                    free_text=(
                        "AI engineer building evidence-grounded LLM applications."
                    ),
                ),
                StructuredCVSection(
                    section=CVSection.PROJECTS,
                    heading="Projects",
                    free_text=(f"CareerOps Agent Engine\n{final_text}"),
                ),
            ],
        ),
        applied_changes=[
            AppliedCVChange(
                change_id=change_id,
                proposal_id=proposal_id,
                section=CVSection.PROJECTS,
                application_mode=(CVChangeApplicationMode.ANCHORED_REPLACEMENT),
                source_anchor=original_text,
                original_text=original_text,
                applied_text=final_text,
                anchor_evidence_ids=[
                    evidence_id,
                ],
                requirement_ids=[
                    requirement_id,
                ],
                supporting_evidence_ids=[
                    evidence_id,
                ],
            )
        ],
        provenance=CVVersionProvenance(
            job_id=job_id,
            thread_id=thread_id,
            source_document_id=document_id,
            requirement_ids=[
                requirement_id,
            ],
            supporting_evidence_ids=[
                evidence_id,
            ],
            final_proposal_ids=[
                proposal_id,
            ],
            review_status=(ApprovalStatus.APPROVED),
            template_id="careerops-standard",
            template_version="1.0.0",
            workflow_version="1.0.0",
        ),
        artifacts=[],
    )

    engine = create_database_engine(settings)

    session_factory = create_session_factory(engine)

    version_repository = SqlAlchemyCVVersionRepository(session_factory)

    artifact_storage = LocalArtifactStorage(tmp_path / "artifacts")

    try:
        # ---------------------------------------------------------
        # 1. Create the trusted PostgreSQL lineage required by
        #    CVVersion persistence.
        # ---------------------------------------------------------
        with session_factory.begin() as session:
            session.add(
                CareerDocumentRecord(
                    user_id=user_id,
                    document_id=document_id,
                    original_filename=("careerops-live-source.docx"),
                    document_format=(CareerDocumentFormat.DOCX.value),
                    media_type=(
                        "application/vnd.openxmlformats-officedocument."
                        "wordprocessingml.document"
                    ),
                    size_bytes=128,
                    sha256_hex="a" * 64,
                    storage_key=(f"documents/live/{document_id}.docx"),
                    status=(CareerDocumentStatus.EXTRACTED.value),
                )
            )

            session.add(
                JobAnalysisRunRecord(
                    thread_id=thread_id,
                    user_id=user_id,
                    job_id=job_id,
                    status=(JobAnalysisRunStatus.COMPLETED.value),
                    role_title=("Junior AI Engineer"),
                    fit_score=100.0,
                    review_status=(ApprovalStatus.APPROVED.value),
                    cv_proposals=[proposal.model_dump(mode="json")],
                    claim_verification_reports=[],
                    reviewable_proposal_ids=[proposal_id],
                    blocked_proposal_ids=[],
                    final_cv_proposals=[proposal.model_dump(mode="json")],
                )
            )

        # ---------------------------------------------------------
        # 2. Persist the immutable assembled CV version.
        # ---------------------------------------------------------
        version_repository.save(
            user_id=user_id,
            version=version,
        )

        persisted = version_repository.get(
            user_id=user_id,
            cv_version_id=(cv_version_id),
        )

        assert persisted == version

        # ---------------------------------------------------------
        # 3. Render and privately persist the real DOCX.
        # ---------------------------------------------------------
        docx_result = CVArtifactRenderingService(
            version_repository=(version_repository),
            artifact_storage=(artifact_storage),
            renderer=(CareerOpsStandardDocxRenderer()),
        ).render_and_store(
            user_id=user_id,
            cv_version_id=cv_version_id,
        )

        assert docx_result.artifact.artifact_format is CVArtifactFormat.DOCX

        assert (
            docx_result.artifact.verification_status
            is CVArtifactVerificationStatus.PENDING
        )

        docx_bytes = artifact_storage.read(
            user_id=user_id,
            storage_key=(docx_result.artifact.storage_key),
        )

        assert sha256(docx_bytes).hexdigest() == (docx_result.artifact.sha256_hex)

        with ZipFile(
            BytesIO(docx_bytes),
            "r",
        ) as package:
            assert "word/document.xml" in package.namelist()

        reopened_docx = Document(BytesIO(docx_bytes))

        docx_visible_text = "\n".join(
            paragraph.text for paragraph in reopened_docx.paragraphs
        )

        assert final_text in docx_visible_text
        assert requirement_id not in docx_visible_text
        assert evidence_id not in docx_visible_text

        # ---------------------------------------------------------
        # 4. Verify the persisted DOCX.
        # ---------------------------------------------------------
        docx_verification = CVArtifactVerificationService(
            version_repository=(version_repository),
            artifact_storage=(artifact_storage),
            verifier=(CareerOpsStandardDocxVerifier()),
        ).verify(
            user_id=user_id,
            cv_version_id=cv_version_id,
        )

        assert docx_verification.passed is True

        assert (
            docx_verification.artifact.verification_status
            is CVArtifactVerificationStatus.VERIFIED
        )

        assert docx_verification.version.status is CVVersionStatus.RENDERED

        # ---------------------------------------------------------
        # 5. Run the actual installed LibreOffice executable.
        # ---------------------------------------------------------
        pdf_conversion = CVPDFConversionService(
            version_repository=(version_repository),
            artifact_storage=(artifact_storage),
            converter=(
                LibreOfficePDFConverter(
                    executable=(settings.libreoffice_executable),
                    timeout_seconds=(settings.pdf_conversion_timeout_seconds),
                )
            ),
        ).convert_and_store(
            user_id=user_id,
            cv_version_id=cv_version_id,
        )

        assert pdf_conversion.reused_existing is False

        assert pdf_conversion.artifact.artifact_format is CVArtifactFormat.PDF

        assert (
            pdf_conversion.artifact.verification_status
            is CVArtifactVerificationStatus.PENDING
        )

        pdf_bytes = artifact_storage.read(
            user_id=user_id,
            storage_key=(pdf_conversion.artifact.storage_key),
        )

        assert pdf_bytes.startswith(b"%PDF-")

        assert sha256(pdf_bytes).hexdigest() == (pdf_conversion.artifact.sha256_hex)

        # ---------------------------------------------------------
        # 6. Reopen the actual LibreOffice PDF before verification.
        # ---------------------------------------------------------
        reader = PdfReader(BytesIO(pdf_bytes))

        assert len(reader.pages) >= 1

        pdf_visible_text = "\n".join(page.extract_text() or "" for page in reader.pages)

        assert "CareerOps Agent Engine" in pdf_visible_text

        assert requirement_id not in (pdf_visible_text)

        assert evidence_id not in (pdf_visible_text)

        # ---------------------------------------------------------
        # 7. Verify PDF content and promote complete version.
        # ---------------------------------------------------------
        pdf_verification = CVArtifactVerificationService(
            version_repository=(version_repository),
            artifact_storage=(artifact_storage),
            verifier=(CareerOpsStandardPDFVerifier()),
        ).verify(
            user_id=user_id,
            cv_version_id=cv_version_id,
        )

        assert pdf_verification.passed is True

        assert (
            pdf_verification.artifact.verification_status
            is CVArtifactVerificationStatus.VERIFIED
        )

        assert pdf_verification.version.status is CVVersionStatus.VERIFIED

        assert {
            artifact.artifact_format for artifact in pdf_verification.version.artifacts
        } == {
            CVArtifactFormat.DOCX,
            CVArtifactFormat.PDF,
        }

        assert all(
            artifact.verification_status is CVArtifactVerificationStatus.VERIFIED
            for artifact in pdf_verification.version.artifacts
        )

        # ---------------------------------------------------------
        # 8. Prove the final lifecycle was really persisted in DB.
        # ---------------------------------------------------------
        with session_factory() as session:
            version_record = session.get(
                CVVersionRecord,
                cv_version_id,
            )

            assert version_record is not None

            assert version_record.status == (CVVersionStatus.VERIFIED.value)

            artifact_records = (
                session.query(CVArtifactRecord)
                .filter(CVArtifactRecord.cv_version_id == cv_version_id)
                .all()
            )

            assert len(artifact_records) == 2

            assert {record.artifact_format for record in artifact_records} == {
                CVArtifactFormat.DOCX.value,
                CVArtifactFormat.PDF.value,
            }

            assert all(
                record.verification_status
                == (CVArtifactVerificationStatus.VERIFIED.value)
                for record in artifact_records
            )

        print()
        print("=== CareerOps live document pipeline proof ===")
        print(f"user_id: {user_id}")
        print(f"cv_version_id: {cv_version_id}")
        print("DOCX: rendered, stored, reopened, verified")
        print("PDF: LibreOffice converted, stored, reopened, verified")
        print("final CVVersion status: verified")
        print("REAL DOCX + PDF PIPELINE PASSED")

    finally:
        # ---------------------------------------------------------
        # Live-test database cleanup.
        # tmp_path automatically removes generated files.
        # ---------------------------------------------------------
        with session_factory.begin() as session:
            session.execute(
                delete(CVArtifactRecord).where(
                    CVArtifactRecord.cv_version_id == cv_version_id
                )
            )

            session.execute(
                delete(CVVersionRecord).where(
                    CVVersionRecord.cv_version_id == cv_version_id
                )
            )

            session.execute(
                delete(JobAnalysisRunRecord).where(
                    JobAnalysisRunRecord.thread_id == thread_id
                )
            )

            session.execute(
                delete(CareerDocumentRecord).where(
                    CareerDocumentRecord.user_id == user_id,
                    CareerDocumentRecord.document_id == document_id,
                )
            )

        engine.dispose()
