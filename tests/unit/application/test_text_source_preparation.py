"""Tests for text sources using the shared CV preparation pipeline."""

from careerops_agent_engine.application.services.cv_document_extraction import (
    CVDocumentExtractionService,
)
from careerops_agent_engine.application.services.cv_document_preparation import (
    CVDocumentPreparationService,
)
from careerops_agent_engine.domain.enums import (
    CareerDocumentFormat,
    CareerDocumentStatus,
    CVSection,
    EvidenceSourceType,
)
from careerops_agent_engine.domain.models.document import (
    CareerDocument,
)
from careerops_agent_engine.infrastructure.documents.cv_section_parser import (
    DeterministicCVSectionParser,
)
from careerops_agent_engine.infrastructure.documents.native_document_extractor import (
    NativeDocumentExtractor,
)


class FakeTextStorage:
    """Return trusted UTF-8 bytes from the shared storage boundary."""

    def __init__(self, data: bytes) -> None:
        self._data = data

    def save(
        self,
        *,
        user_id: str,
        document_id: str,
        document_format: CareerDocumentFormat,
        data: bytes,
    ) -> str:
        """Unused storage operation."""

        del user_id, document_id, document_format, data
        raise NotImplementedError

    def read(
        self,
        *,
        user_id: str,
        storage_key: str,
    ) -> bytes:
        """Return the stored text inside its ownership boundary."""

        assert user_id == "USER-001"
        assert storage_key == "documents/usr-test/DOC-TEXT-001.txt"

        return self._data

    def delete(
        self,
        *,
        user_id: str,
        storage_key: str,
    ) -> None:
        """Unused storage operation."""

        del user_id, storage_key
        raise NotImplementedError


def test_text_source_uses_shared_preparation_pipeline() -> None:
    """Trusted text should become the same parsed structure as a CV file."""

    data = (
        b"Projects\n"
        b"  Built   CareerOps using Python and FastAPI.  \n"
        b"Deployed the service with Docker."
    )

    extraction_service = CVDocumentExtractionService(
        storage=FakeTextStorage(data),
        extractor=NativeDocumentExtractor(),
    )

    service = CVDocumentPreparationService(
        extraction_service=extraction_service,
        section_parser=DeterministicCVSectionParser(),
    )

    document = CareerDocument(
        document_id="DOC-TEXT-001",
        original_filename="pasted-evidence.txt",
        document_format=CareerDocumentFormat.TEXT,
        media_type="text/plain; charset=utf-8",
        size_bytes=len(data),
        sha256_hex="a" * 64,
        storage_key="documents/usr-test/DOC-TEXT-001.txt",
        status=CareerDocumentStatus.UPLOADED,
    )

    parsed = service.prepare(
        user_id="USER-001",
        document=document,
    )

    assert parsed.document_id == "DOC-TEXT-001"
    assert parsed.source_type is EvidenceSourceType.MANUAL_ENTRY
    assert len(parsed.sections) == 1
    assert parsed.sections[0].section is CVSection.PROJECTS
    assert parsed.sections[0].heading == "Projects"
    assert parsed.sections[0].text == (
        "Built CareerOps using Python and FastAPI.\nDeployed the service with Docker."
    )
    assert parsed.warnings == []
