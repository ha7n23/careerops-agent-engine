"""Tests for deterministic CV section parsing."""

from careerops_agent_engine.domain.enums import (
    CVSection,
)
from careerops_agent_engine.domain.models.document import (
    ExtractedDocumentText,
)
from careerops_agent_engine.infrastructure.documents.cv_section_parser import (
    DeterministicCVSectionParser,
)


def build_extracted_text(
    text: str,
) -> ExtractedDocumentText:
    """Create deterministic extracted CV text."""

    return ExtractedDocumentText(
        document_id="DOC-001",
        text=text,
        page_count=1,
        paragraph_count=None,
        warnings=[],
    )


def test_parser_maps_supported_heading_aliases() -> None:
    """Common CV headings should map to canonical sections."""

    document = build_extracted_text(
        """
        Candidate Name
        candidate@example.com

        Professional Summary
        Python engineer building AI applications.

        Technical Skills
        Python, FastAPI, Docker

        Work Experience
        Software Engineer at Example Ltd.

        Selected Projects
        CareerOps AI platform.

        Education
        BSc Computer Science

        Certifications
        Cloud certification
        """
    )

    result = DeterministicCVSectionParser().parse(document=document)

    assert result.preamble_text is not None
    assert "Candidate Name" in result.preamble_text
    assert "candidate@example.com" in result.preamble_text

    assert [section.section for section in result.sections] == [
        CVSection.PROFILE,
        CVSection.SKILLS,
        CVSection.EXPERIENCE,
        CVSection.PROJECTS,
        CVSection.EDUCATION,
        CVSection.CERTIFICATIONS,
    ]

    assert result.warnings == []


def test_heading_matching_is_case_insensitive() -> None:
    """Capitalisation and trailing colons should not affect headings."""

    document = build_extracted_text(
        """
        PROFESSIONAL EXPERIENCE:
        Built Python APIs.

        selected projects:
        Built CareerOps.
        """
    )

    result = DeterministicCVSectionParser().parse(document=document)

    assert [section.section for section in result.sections] == [
        CVSection.EXPERIENCE,
        CVSection.PROJECTS,
    ]


def test_heading_words_inside_sentences_are_not_headings() -> None:
    """Heading keywords within body sentences must not split text."""

    document = build_extracted_text(
        """
        Candidate Name
        Experience building Python applications with FastAPI.
        Projects included several AI engineering systems.
        """
    )

    result = DeterministicCVSectionParser().parse(document=document)

    assert result.sections == []

    assert result.preamble_text is not None

    assert "Experience building Python applications" in result.preamble_text

    assert any(
        "No recognised CV section headings" in warning for warning in result.warnings
    )


def test_repeated_section_types_preserve_source_order() -> None:
    """Repeated headings should remain separate ordered blocks."""

    document = build_extracted_text(
        """
        Projects
        CareerOps platform.

        Experience
        Software Engineer.

        Projects
        Computer vision classifier.
        """
    )

    result = DeterministicCVSectionParser().parse(document=document)

    assert [section.section for section in result.sections] == [
        CVSection.PROJECTS,
        CVSection.EXPERIENCE,
        CVSection.PROJECTS,
    ]

    assert [section.order_index for section in result.sections] == [
        0,
        1,
        2,
    ]

    assert result.sections[0].text == "CareerOps platform."

    assert result.sections[2].text == "Computer vision classifier."


def test_empty_recognised_section_is_omitted_with_warning() -> None:
    """A heading without content should not create an empty section."""

    document = build_extracted_text(
        """
        Experience

        Projects
        CareerOps platform.
        """
    )

    result = DeterministicCVSectionParser().parse(document=document)

    assert [section.section for section in result.sections] == [CVSection.PROJECTS]

    assert any("Experience" in warning for warning in result.warnings)


def test_unknown_heading_like_text_is_preserved() -> None:
    """Unsupported headings should remain content rather than vanish."""

    document = build_extracted_text(
        """
        Projects
        CareerOps

        Open Source Contributions
        Contributed to an internal utility.
        """
    )

    result = DeterministicCVSectionParser().parse(document=document)

    assert len(result.sections) == 1

    project_text = result.sections[0].text

    assert "Open Source Contributions" in project_text

    assert "Contributed to an internal utility." in project_text
