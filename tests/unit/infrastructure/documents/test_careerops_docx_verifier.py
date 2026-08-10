"""Tests for deterministic CareerOps DOCX verification."""

from io import BytesIO

from docx import Document

from careerops_agent_engine.domain.enums import (
    ApprovalStatus,
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
)
from careerops_agent_engine.domain.models.structured_cv import (
    StructuredCV,
    StructuredCVSection,
)
from careerops_agent_engine.infrastructure.documents.careerops_docx_renderer import (
    CareerOpsStandardDocxRenderer,
)
from careerops_agent_engine.infrastructure.documents.careerops_docx_verifier import (
    CareerOpsStandardDocxVerifier,
)


def build_version() -> CVVersion:
    """Create one renderable verified-source CV version."""

    return CVVersion(
        cv_version_id="CVV-001",
        version_number=1,
        status=CVVersionStatus.ASSEMBLED,
        structured_cv=StructuredCV(
            cv_id="CV-001",
            source_document_id="DOC-001",
            preamble_text=("Example Candidate\ncandidate@example.com"),
            sections=[
                StructuredCVSection(
                    section=CVSection.PROJECTS,
                    heading="Projects",
                    free_text=("CareerOps\nBuilt CareerOps using Python and FastAPI."),
                )
            ],
        ),
        applied_changes=[
            AppliedCVChange(
                change_id="CHG-001",
                proposal_id="CVP-001",
                section=CVSection.PROJECTS,
                application_mode=(CVChangeApplicationMode.ANCHORED_REPLACEMENT),
                source_anchor=("Built CareerOps using Python."),
                original_text=("Built CareerOps using Python."),
                applied_text=("Built CareerOps using Python and FastAPI."),
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
        artifacts=[],
    )


def test_verifier_accepts_exact_standard_renderer_output() -> None:
    """Renderer output should satisfy its deterministic verification contract."""

    version = build_version()

    rendered = CareerOpsStandardDocxRenderer().render(version=version)

    result = CareerOpsStandardDocxVerifier().verify(
        version=version,
        data=rendered.data,
    )

    assert result.passed is True
    assert result.notes == ()


def test_verifier_rejects_invalid_docx_package() -> None:
    """Arbitrary bytes must never pass DOCX verification."""

    result = CareerOpsStandardDocxVerifier().verify(
        version=build_version(),
        data=b"not-a-docx",
    )

    assert result.passed is False

    assert "invalid" in result.notes[0].lower()


def test_verifier_rejects_visible_content_mutation() -> None:
    """A valid DOCX with changed text must fail the CV contract."""

    version = build_version()

    document = Document()

    document.add_paragraph("Unexpected replacement text.")

    stream = BytesIO()

    document.save(stream)

    result = CareerOpsStandardDocxVerifier().verify(
        version=version,
        data=stream.getvalue(),
    )

    assert result.passed is False

    assert "does not match" in result.notes[0]


def test_verifier_rejects_internal_provenance_leak() -> None:
    """Internal CareerOps IDs must never become visible CV content."""

    version = build_version()

    leaked = version.model_copy(
        update={
            "structured_cv": (
                version.structured_cv.model_copy(
                    update={"preamble_text": ("Example Candidate\nREQ-001")}
                )
            )
        }
    )

    rendered = CareerOpsStandardDocxRenderer().render(version=leaked)

    result = CareerOpsStandardDocxVerifier().verify(
        version=leaked,
        data=rendered.data,
    )

    assert result.passed is False

    assert "provenance identifiers" in result.notes[0]
