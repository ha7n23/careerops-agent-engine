"""Tests for deterministic CareerOps DOCX rendering."""

from io import BytesIO
from zipfile import ZipFile

import pytest
from docx import Document

from careerops_agent_engine.application.exceptions import (
    CVRenderingError,
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
    CVBullet,
    CVEntry,
    StructuredCV,
    StructuredCVSection,
)
from careerops_agent_engine.infrastructure.documents.careerops_docx_renderer import (
    CANONICAL_ZIP_TIMESTAMP,
    DOCX_MEDIA_TYPE,
    CareerOpsStandardDocxRenderer,
)


def build_version(
    *,
    template_id: str = "careerops-standard",
    template_version: str = "1.0.0",
) -> CVVersion:
    """Create one renderable immutable CV version."""

    structured_cv = StructuredCV(
        cv_id="CV-001",
        source_document_id="DOC-001",
        preamble_text=("Example Candidate\ncandidate@example.com | +44 0000 000000"),
        sections=[
            StructuredCVSection(
                section=CVSection.PROFILE,
                heading="Profile",
                free_text=("AI engineer building grounded LLM applications."),
            ),
            StructuredCVSection(
                section=CVSection.PROJECTS,
                heading="Projects",
                free_text=(
                    "CareerOps\n"
                    "Built CareerOps using Python "
                    "and FastAPI.\n"
                    "Untouched project text."
                ),
            ),
            StructuredCVSection(
                section=CVSection.EDUCATION,
                heading="Education",
                entries=[
                    CVEntry(
                        entry_id="ENT-001",
                        section=(CVSection.EDUCATION),
                        title=("BSc Computer Science"),
                        subtitle=("Example University"),
                        date_text="2023–2026",
                        bullets=[
                            CVBullet(
                                bullet_id="BLT-001",
                                text=("Completed an AI engineering capstone."),
                                supporting_evidence_ids=["EVD-001"],
                                requirement_ids=["REQ-001"],
                            )
                        ],
                    )
                ],
            ),
        ],
    )

    change = AppliedCVChange(
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

    provenance = CVVersionProvenance(
        job_id="JOB-001",
        thread_id="THR-001",
        source_document_id="DOC-001",
        requirement_ids=["REQ-001"],
        supporting_evidence_ids=["EVD-001"],
        final_proposal_ids=["CVP-001"],
        review_status=(ApprovalStatus.APPROVED),
        template_id=template_id,
        template_version=(template_version),
        workflow_version="1.0.0",
    )

    return CVVersion(
        cv_version_id="CVV-001",
        version_number=1,
        status=CVVersionStatus.ASSEMBLED,
        structured_cv=structured_cv,
        applied_changes=[change],
        provenance=provenance,
        artifacts=[],
    )


def visible_paragraph_text(
    data: bytes,
) -> list[str]:
    """Read visible paragraph text back from rendered DOCX bytes."""

    document = Document(BytesIO(data))

    return [paragraph.text for paragraph in document.paragraphs]


def test_renderer_produces_valid_editable_docx() -> None:
    """The standard renderer should return an openable DOCX package."""

    rendered = CareerOpsStandardDocxRenderer().render(version=build_version())

    assert rendered.filename == ("CVV-001.docx")

    assert rendered.media_type == DOCX_MEDIA_TYPE

    with ZipFile(
        BytesIO(rendered.data),
        "r",
    ) as package:
        assert "word/document.xml" in package.namelist()

    assert visible_paragraph_text(rendered.data)


def test_renderer_preserves_user_cv_text_and_section_order() -> None:
    """Rendering may style content but must not paraphrase final text."""

    rendered = CareerOpsStandardDocxRenderer().render(version=build_version())

    paragraphs = visible_paragraph_text(rendered.data)

    expected_in_order = [
        "Example Candidate",
        ("candidate@example.com | +44 0000 000000"),
        "PROFILE",
        ("AI engineer building grounded LLM applications."),
        "PROJECTS",
        "CareerOps",
        ("Built CareerOps using Python and FastAPI."),
        "Untouched project text.",
        "EDUCATION",
        "BSc Computer Science",
        ("Example University | 2023–2026"),
        ("Completed an AI engineering capstone."),
    ]

    positions = [paragraphs.index(text) for text in expected_in_order]

    assert positions == sorted(positions)


def test_renderer_does_not_expose_internal_provenance_ids() -> None:
    """Internal evidence/workflow IDs belong in metadata, not CV text."""

    rendered = CareerOpsStandardDocxRenderer().render(version=build_version())

    visible_text = "\n".join(visible_paragraph_text(rendered.data))

    assert "EVD-001" not in visible_text
    assert "REQ-001" not in visible_text
    assert "CVP-001" not in visible_text


def test_renderer_requires_matching_template_provenance() -> None:
    """A CV version cannot silently render through another template."""

    renderer = CareerOpsStandardDocxRenderer()

    with pytest.raises(
        CVRenderingError,
        match="template identifier",
    ):
        renderer.render(version=build_version(template_id=("other-template")))

    with pytest.raises(
        CVRenderingError,
        match="template version",
    ):
        renderer.render(version=build_version(template_version=("2.0.0")))


def test_renderer_output_is_byte_deterministic() -> None:
    """An exact rendering retry should produce the same bytes."""

    renderer = CareerOpsStandardDocxRenderer()

    version = build_version()

    first = renderer.render(version=version)

    second = renderer.render(version=version)

    assert first.data == second.data

    with ZipFile(
        BytesIO(first.data),
        "r",
    ) as package:
        assert {info.date_time for info in package.infolist()} == {
            CANONICAL_ZIP_TIMESTAMP
        }
