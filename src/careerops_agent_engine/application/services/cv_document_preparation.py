"""Application orchestration for extracted and parsed CV documents."""

from careerops_agent_engine.application.ports.cv_section_parser import (
    CVSectionParser,
)
from careerops_agent_engine.application.services.cv_document_extraction import (
    CVDocumentExtractionService,
)
from careerops_agent_engine.domain.enums import (
    CareerDocumentFormat,
    EvidenceSourceType,
)
from careerops_agent_engine.domain.models.document import (
    CareerDocument,
    ParsedCVDocument,
)


class CVDocumentPreparationService:
    """Extract and deterministically structure one uploaded CV."""

    def __init__(
        self,
        *,
        extraction_service: CVDocumentExtractionService,
        section_parser: CVSectionParser,
    ) -> None:
        """Store document-processing dependencies."""

        self._extraction_service = extraction_service
        self._section_parser = section_parser

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

        parsed = self._section_parser.parse(document=extracted)

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
