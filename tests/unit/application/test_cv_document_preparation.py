"""Tests for CV document preparation orchestration."""

import pytest

from careerops_agent_engine.application.exceptions import (
    DocumentExtractionError,
)
from careerops_agent_engine.application.services.cv_document_preparation import (
    CVDocumentPreparationService,
    merge_warnings,
)
from careerops_agent_engine.domain.enums import (
    CareerDocumentFormat,
    CareerDocumentStatus,
    CVSection,
)
from careerops_agent_engine.domain.models.document import (
    CareerDocument,
    ExtractedDocumentText,
    ParsedCVDocument,
    ParsedCVSection,
)


class FakeExtractionService:
    """Return controlled native extraction output."""

    def __init__(
        self,
        result: ExtractedDocumentText,
    ) -> None:
        """Store deterministic extraction output."""

        self.result = result
        self.user_id: str | None = None
        self.document: CareerDocument | None = None

    def extract(
        self,
        *,
        user_id: str,
        document: CareerDocument,
    ) -> ExtractedDocumentText:
        """Capture arguments and return extracted text."""

        self.user_id = user_id
        self.document = document

        return self.result


class FakeSectionParser:
    """Return controlled deterministic section output."""

    def __init__(
        self,
        result: ParsedCVDocument,
    ) -> None:
        """Store deterministic parser output."""

        self.result = result
        self.document: ExtractedDocumentText | None = None

    def parse(
        self,
        *,
        document: ExtractedDocumentText,
    ) -> ParsedCVDocument:
        """Capture extracted text and return parsed CV."""

        self.document = document

        return self.result


def build_document() -> CareerDocument:
    """Create trusted career-document metadata."""

    return CareerDocument(
        document_id="DOC-001",
        original_filename="cv.pdf",
        document_format=CareerDocumentFormat.PDF,
        media_type="application/pdf",
        size_bytes=100,
        sha256_hex="a" * 64,
        storage_key=("documents/usr-test/DOC-001.pdf"),
        status=CareerDocumentStatus.UPLOADED,
    )


def test_prepare_runs_extraction_then_section_parsing() -> None:
    """Preparation should compose the existing trusted stages."""

    extracted = ExtractedDocumentText(
        document_id="DOC-001",
        text=("Profile\nPython engineer.\nSkills\nPython, FastAPI"),
        page_count=2,
        paragraph_count=None,
        warnings=["Page 2 contained no extractable native text."],
    )

    parsed = ParsedCVDocument(
        document_id="DOC-001",
        preamble_text=None,
        sections=[
            ParsedCVSection(
                section=CVSection.PROFILE,
                heading="Profile",
                text="Python engineer.",
                order_index=0,
            ),
            ParsedCVSection(
                section=CVSection.SKILLS,
                heading="Skills",
                text="Python, FastAPI",
                order_index=1,
            ),
        ],
        warnings=["Parser warning."],
    )

    extraction_service = FakeExtractionService(extracted)
    parser = FakeSectionParser(parsed)

    service = CVDocumentPreparationService(
        extraction_service=extraction_service,
        section_parser=parser,
    )

    document = build_document()

    result = service.prepare(
        user_id="USER-001",
        document=document,
    )

    assert extraction_service.user_id == "USER-001"
    assert extraction_service.document == document

    assert parser.document == extracted

    assert [section.section for section in result.sections] == [
        CVSection.PROFILE,
        CVSection.SKILLS,
    ]

    assert result.warnings == [
        "Page 2 contained no extractable native text.",
        "Parser warning.",
    ]


def test_merge_warnings_preserves_order_and_removes_duplicates() -> None:
    """Repeated warnings should appear only once downstream."""

    warnings = merge_warnings(
        [
            "Warning A",
            "Warning B",
        ],
        [
            "Warning B",
            "Warning C",
        ],
    )

    assert warnings == [
        "Warning A",
        "Warning B",
        "Warning C",
    ]


def test_prepare_rejects_excessive_extracted_text_before_parsing() -> None:
    """Oversized extracted text must not reach the section parser."""

    extracted = ExtractedDocumentText(
        document_id="DOC-001",
        text="A" * 11,
        page_count=1,
        paragraph_count=None,
        warnings=[],
    )

    parser = FakeSectionParser(
        ParsedCVDocument(
            document_id="DOC-001",
            sections=[],
        )
    )

    service = CVDocumentPreparationService(
        extraction_service=FakeExtractionService(extracted),
        section_parser=parser,
        max_extracted_characters=10,
    )

    with pytest.raises(
        DocumentExtractionError,
        match="maximum extracted text length",
    ):
        service.prepare(
            user_id="USER-001",
            document=build_document(),
        )

    assert parser.document is None


def test_prepare_rejects_too_many_sections() -> None:
    """Excessive parsed sections must not reach proposal generation."""

    extracted = ExtractedDocumentText(
        document_id="DOC-001",
        text="Projects\nProject A\nProjects\nProject B",
        page_count=1,
        paragraph_count=None,
        warnings=[],
    )

    parsed = ParsedCVDocument(
        document_id="DOC-001",
        sections=[
            ParsedCVSection(
                section=CVSection.PROJECTS,
                heading="Projects",
                text="Project A",
                order_index=0,
            ),
            ParsedCVSection(
                section=CVSection.PROJECTS,
                heading="Projects",
                text="Project B",
                order_index=1,
            ),
        ],
    )

    service = CVDocumentPreparationService(
        extraction_service=FakeExtractionService(extracted),
        section_parser=FakeSectionParser(parsed),
        max_sections=1,
    )

    with pytest.raises(
        DocumentExtractionError,
        match="too many recognised CV sections",
    ):
        service.prepare(
            user_id="USER-001",
            document=build_document(),
        )


def test_preparation_requires_positive_limits() -> None:
    """Application-layer document bounds must not be disabled."""

    extracted = ExtractedDocumentText(
        document_id="DOC-001",
        text="Projects\nCareerOps",
        page_count=1,
        paragraph_count=None,
        warnings=[],
    )

    parser = FakeSectionParser(
        ParsedCVDocument(
            document_id="DOC-001",
            sections=[],
        )
    )

    with pytest.raises(
        ValueError,
        match="character count must be positive",
    ):
        CVDocumentPreparationService(
            extraction_service=FakeExtractionService(extracted),
            section_parser=parser,
            max_extracted_characters=0,
        )

    with pytest.raises(
        ValueError,
        match="section count must be positive",
    ):
        CVDocumentPreparationService(
            extraction_service=FakeExtractionService(extracted),
            section_parser=parser,
            max_sections=0,
        )
