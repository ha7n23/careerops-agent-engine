"""Tests for deterministic CareerOps PDF verification."""

from io import BytesIO

from pypdf import PdfWriter
from pypdf.generic import (
    DecodedStreamObject,
    DictionaryObject,
    NameObject,
)

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
from careerops_agent_engine.infrastructure.documents.careerops_pdf_verifier import (
    CareerOpsStandardPDFVerifier,
)


def build_version() -> CVVersion:
    """Create one immutable CV version for PDF verification."""

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
                    section=CVSection.PROFILE,
                    heading="Profile",
                    free_text=("AI engineer building grounded LLM applications."),
                ),
                StructuredCVSection(
                    section=CVSection.PROJECTS,
                    heading="Projects",
                    free_text=("CareerOps\nBuilt CareerOps using Python and FastAPI."),
                ),
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


def build_pdf_with_text(
    text: str,
) -> bytes:
    """Create a minimal valid one-page PDF with extractable text."""

    writer = PdfWriter()

    page = writer.add_blank_page(
        width=612,
        height=792,
    )

    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )

    font_reference = writer._add_object(font)

    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_reference})}
    )

    escaped = (
        text.replace(
            "\\",
            "\\\\",
        )
        .replace(
            "(",
            "\\(",
        )
        .replace(
            ")",
            "\\)",
        )
    )

    content = DecodedStreamObject()

    content.set_data((f"BT /F1 10 Tf 36 760 Td ({escaped}) Tj ET").encode("latin-1"))

    page[NameObject("/Contents")] = writer._add_object(content)

    stream = BytesIO()

    writer.write(stream)

    return stream.getvalue()


def expected_pdf_text() -> str:
    """Return all approved visible content in renderer order."""

    return (
        "Example Candidate "
        "candidate@example.com "
        "PROFILE "
        "AI engineer building grounded LLM applications. "
        "PROJECTS "
        "CareerOps "
        "Built CareerOps using Python and FastAPI."
    )


def test_verifier_accepts_expected_pdf_content() -> None:
    """All approved visible content in order should pass."""

    result = CareerOpsStandardPDFVerifier().verify(
        version=build_version(),
        data=build_pdf_with_text(expected_pdf_text()),
    )

    assert result.passed is True
    assert result.notes == ()


def test_verifier_tolerates_pdf_whitespace_differences() -> None:
    """Converter whitespace differences should not cause false failures."""

    text = (
        "Example   Candidate    "
        "candidate@example.com   "
        "PROFILE    "
        "AI engineer building grounded "
        "LLM applications.    "
        "PROJECTS    CareerOps    "
        "Built CareerOps using Python "
        "and FastAPI."
    )

    result = CareerOpsStandardPDFVerifier().verify(
        version=build_version(),
        data=build_pdf_with_text(text),
    )

    assert result.passed is True


def test_verifier_rejects_invalid_pdf_bytes() -> None:
    """Arbitrary bytes cannot pass PDF verification."""

    result = CareerOpsStandardPDFVerifier().verify(
        version=build_version(),
        data=b"not-a-pdf",
    )

    assert result.passed is False

    assert "not a PDF" in result.notes[0]


def test_verifier_rejects_missing_cv_content() -> None:
    """A structurally valid PDF cannot omit approved CV content."""

    result = CareerOpsStandardPDFVerifier().verify(
        version=build_version(),
        data=build_pdf_with_text("Example Candidate PROFILE CareerOps"),
    )

    assert result.passed is False

    assert "does not match" in result.notes[0]


def test_verifier_rejects_internal_provenance_leak() -> None:
    """Internal CareerOps identifiers must remain outside the PDF."""

    leaked_text = f"{expected_pdf_text()} REQ-001"

    result = CareerOpsStandardPDFVerifier().verify(
        version=build_version(),
        data=build_pdf_with_text(leaked_text),
    )

    assert result.passed is False

    assert "provenance identifiers" in result.notes[0]
