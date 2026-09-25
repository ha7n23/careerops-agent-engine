"""Application orchestration for extracted and parsed CV documents."""

from careerops_agent_engine.application.exceptions import (
    DocumentExtractionError,
)
from careerops_agent_engine.application.ports.career_document_extraction import (
    CareerDocumentExtraction,
)
from careerops_agent_engine.application.ports.cv_section_parser import (
    CVSectionParser,
)
from careerops_agent_engine.domain.enums import (
    CareerDocumentFormat,
    EvidenceSourceType,
)
from careerops_agent_engine.domain.models.document import (
    CareerDocument,
    ParsedCVDocument,
)

DEFAULT_MAX_EXTRACTED_CHARACTERS = 30_000
DEFAULT_MAX_SECTIONS = 20


class CVDocumentPreparationService:
    """Extract and deterministically structure one uploaded CV."""

    def __init__(
        self,
        *,
        extraction_service: CareerDocumentExtraction,
        section_parser: CVSectionParser,
        max_extracted_characters: int = DEFAULT_MAX_EXTRACTED_CHARACTERS,
        max_sections: int = DEFAULT_MAX_SECTIONS,
    ) -> None:
        """Store document-processing dependencies and safety bounds."""

        if max_extracted_characters < 1:
            raise ValueError("Maximum extracted character count must be positive.")

        if max_sections < 1:
            raise ValueError("Maximum section count must be positive.")

        self._extraction_service = extraction_service
        self._section_parser = section_parser
        self._max_extracted_characters = max_extracted_characters
        self._max_sections = max_sections

    def prepare(
        self,
        *,
        user_id: str,
        document: CareerDocument,
    ) -> ParsedCVDocument:
        """Extract native text and divide it into recognised sections."""

        extracted = self._extraction_service.extract(
            user_id=user_id,
            document=document,
        )

        if len(extracted.text) > self._max_extracted_characters:
            raise DocumentExtractionError(
                "The document exceeds the maximum extracted text length."
            )

        parsed = self._section_parser.parse(document=extracted)

        if len(parsed.sections) > self._max_sections:
            raise DocumentExtractionError(
                "The document contains too many recognised CV sections."
            )

        if parsed.document_id != document.document_id:
            raise RuntimeError(
                "Parsed CV document identifier does not match the source document."
            )

        warnings = merge_warnings(
            extracted.warnings,
            parsed.warnings,
        )

        return ParsedCVDocument(
            document_id=parsed.document_id,
            source_type=(
                EvidenceSourceType.MANUAL_ENTRY
                if document.document_format is CareerDocumentFormat.TEXT
                else EvidenceSourceType.UPLOADED_CV
            ),
            preamble_text=parsed.preamble_text,
            sections=list(parsed.sections),
            warnings=warnings,
        )


def merge_warnings(
    *warning_groups: list[str],
) -> list[str]:
    """Combine warnings in source order without duplicates."""

    merged: list[str] = []
    seen: set[str] = set()

    for warning_group in warning_groups:
        for warning in warning_group:
            if warning in seen:
                continue

            seen.add(warning)
            merged.append(warning)

    return merged
