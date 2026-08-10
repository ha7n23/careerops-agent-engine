"""Tests for deterministic source-CV assembly."""

import pytest

from careerops_agent_engine.application.exceptions import (
    StructuredCVAssemblyError,
)
from careerops_agent_engine.application.services.structured_cv_assembly import (
    BaseStructuredCVAssembler,
    build_structured_cv_id,
)
from careerops_agent_engine.domain.enums import (
    CVSection,
)
from careerops_agent_engine.domain.models.document import (
    ParsedCVDocument,
    ParsedCVSection,
)


def build_parsed_document() -> ParsedCVDocument:
    """Create a representative parsed CV."""

    return ParsedCVDocument(
        document_id="DOC-001",
        preamble_text=("Example Candidate\ncandidate@example.com"),
        sections=[
            ParsedCVSection(
                section=CVSection.PROFILE,
                heading="Profile",
                text=("AI engineer building grounded LLM applications."),
                order_index=0,
            ),
            ParsedCVSection(
                section=CVSection.PROJECTS,
                heading="Projects",
                text=("CareerOps\nBuilt CareerOps using Python."),
                order_index=1,
            ),
            ParsedCVSection(
                section=CVSection.EDUCATION,
                heading="Education",
                text=("BSc Computer Science."),
                order_index=2,
            ),
        ],
        warnings=[],
    )


def test_assembler_preserves_source_content_and_order() -> None:
    """Base assembly must not rewrite recognised source content."""

    cv = BaseStructuredCVAssembler().assemble(document=build_parsed_document())

    assert cv.source_document_id == "DOC-001"

    assert cv.preamble_text == ("Example Candidate\ncandidate@example.com")

    assert [section.section for section in cv.sections] == [
        CVSection.PROFILE,
        CVSection.PROJECTS,
        CVSection.EDUCATION,
    ]

    assert cv.sections[1].heading == "Projects"

    assert cv.sections[1].free_text == ("CareerOps\nBuilt CareerOps using Python.")


def test_structured_cv_id_is_stable_for_source_document() -> None:
    """Retrying base assembly should produce the same CV-family ID."""

    first = build_structured_cv_id(document_id="DOC-001")

    second = build_structured_cv_id(document_id="DOC-001")

    other = build_structured_cv_id(document_id="DOC-OTHER")

    assert first == second
    assert first.startswith("CV-")
    assert first != other


def test_repeated_sections_are_merged_without_data_loss() -> None:
    """Repeated recognised sections should retain all source text."""

    document = ParsedCVDocument(
        document_id="DOC-001",
        sections=[
            ParsedCVSection(
                section=CVSection.PROJECTS,
                heading="Projects",
                text="CareerOps project.",
                order_index=0,
            ),
            ParsedCVSection(
                section=CVSection.EDUCATION,
                heading="Education",
                text="BSc Computer Science.",
                order_index=1,
            ),
            ParsedCVSection(
                section=CVSection.PROJECTS,
                heading="Selected Projects",
                text="Computer vision project.",
                order_index=2,
            ),
        ],
        warnings=[],
    )

    cv = BaseStructuredCVAssembler().assemble(document=document)

    assert len(cv.sections) == 2

    projects = cv.sections[0]

    assert projects.section is CVSection.PROJECTS
    assert projects.heading == "Projects"

    assert projects.free_text == ("CareerOps project.\nComputer vision project.")

    assert cv.sections[1].section is (CVSection.EDUCATION)


def test_assembler_allows_cv_without_preamble() -> None:
    """A CV may legitimately have no extracted preamble."""

    document = build_parsed_document().model_copy(
        update={
            "preamble_text": None,
        }
    )

    cv = BaseStructuredCVAssembler().assemble(document=document)

    assert cv.preamble_text is None


def test_assembler_rejects_document_without_recognised_sections() -> None:
    """Assembly must fail rather than fabricate missing CV structure."""

    document = ParsedCVDocument(
        document_id="DOC-001",
        preamble_text="Unstructured CV text.",
        sections=[],
        warnings=["No recognised CV section headings were found."],
    )

    with pytest.raises(
        StructuredCVAssemblyError,
        match="at least one recognised source section",
    ):
        BaseStructuredCVAssembler().assemble(document=document)
