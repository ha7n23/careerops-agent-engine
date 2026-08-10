"""Tests for trusted final-CV generation API."""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from careerops_agent_engine.api.dependencies import (
    get_cv_artifact_retrieval_service,
    get_final_cv_generation_service,
)
from careerops_agent_engine.application.exceptions import (
    CareerDocumentUnavailableError,
    CVArtifactRetrievalError,
    CVPDFConversionError,
    FinalCVGenerationError,
)
from careerops_agent_engine.application.services.cv_artifact_retrieval import (
    RetrievedCVArtifact,
)
from careerops_agent_engine.application.services.final_cv_generation import (
    FinalCVGenerationExecutionResult,
)
from careerops_agent_engine.domain.enums import (
    ApprovalStatus,
    CVArtifactFormat,
    CVArtifactVerificationStatus,
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
    RenderedCVArtifact,
)
from careerops_agent_engine.domain.models.structured_cv import (
    StructuredCV,
    StructuredCVSection,
)
from careerops_agent_engine.main import app


def build_verified_version() -> CVVersion:
    """Create one complete verified final CV."""

    return CVVersion(
        cv_version_id="CVV-API-001",
        version_number=1,
        status=CVVersionStatus.VERIFIED,
        structured_cv=StructuredCV(
            cv_id="CV-API-001",
            source_document_id="DOC-001",
            sections=[
                StructuredCVSection(
                    section=(CVSection.PROJECTS),
                    heading="Projects",
                    free_text=("Built CareerOps using Python and FastAPI."),
                )
            ],
        ),
        applied_changes=[
            AppliedCVChange(
                change_id="CHG-API-001",
                proposal_id="CVP-001",
                section=(CVSection.PROJECTS),
                application_mode=(CVChangeApplicationMode.ANCHORED_REPLACEMENT),
                source_anchor="Original",
                original_text="Original",
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
            template_id=("careerops-standard"),
            template_version="1.0.0",
            workflow_version=("final-cv-v1"),
        ),
        artifacts=[
            RenderedCVArtifact(
                artifact_id="ART-DOCX",
                artifact_format=(CVArtifactFormat.DOCX),
                storage_key=("private/docx"),
                sha256_hex="a" * 64,
                size_bytes=1_000,
                verification_status=(CVArtifactVerificationStatus.VERIFIED),
                verification_notes=[],
            ),
            RenderedCVArtifact(
                artifact_id="ART-PDF",
                artifact_format=(CVArtifactFormat.PDF),
                storage_key=("private/pdf"),
                sha256_hex="b" * 64,
                size_bytes=900,
                verification_status=(CVArtifactVerificationStatus.VERIFIED),
                verification_notes=[],
            ),
        ],
    )


class FakeFinalCVGenerationService:
    """Return or fail one configured final-CV generation."""

    def __init__(
        self,
        *,
        error: Exception | None = None,
        reused_existing: bool = False,
    ) -> None:
        self.error = error
        self.reused_existing = reused_existing
        self.call_count = 0

    def generate(
        self,
        *,
        user_id: str,
        thread_id: str,
        source_document_id: str,
    ) -> FinalCVGenerationExecutionResult:
        """Return deterministic final-CV state."""

        self.call_count += 1

        assert user_id == "USER-001"
        assert thread_id == "THR-001"
        assert source_document_id == "DOC-001"

        if self.error is not None:
            raise self.error

        return FinalCVGenerationExecutionResult(
            version=build_verified_version(),
            reused_existing_version=(self.reused_existing),
        )


class FakeCVArtifactRetrievalService:
    """Return safe metadata and verified bytes."""

    def __init__(
        self,
        *,
        error: Exception | None = None,
    ) -> None:
        self.error = error

    def get_version(
        self,
        *,
        user_id: str,
        cv_version_id: str,
    ) -> CVVersion:
        """Return one owned version."""

        assert user_id == "USER-001"
        assert cv_version_id == ("CVV-API-001")

        if self.error is not None:
            raise self.error

        return build_verified_version()

    def get_verified_artifact(
        self,
        *,
        user_id: str,
        cv_version_id: str,
        artifact_format: CVArtifactFormat,
    ) -> RetrievedCVArtifact:
        """Return one verified artifact payload."""

        version = self.get_version(
            user_id=user_id,
            cv_version_id=cv_version_id,
        )

        artifact = next(
            item
            for item in version.artifacts
            if item.artifact_format is artifact_format
        )

        return RetrievedCVArtifact(
            artifact=artifact,
            filename=(f"{cv_version_id}.{artifact_format.value}"),
            media_type=(
                "application/pdf"
                if artifact_format is CVArtifactFormat.PDF
                else (
                    "application/vnd.openxmlformats-officedocument."
                    "wordprocessingml.document"
                )
            ),
            data=(
                b"%PDF-test"
                if artifact_format is CVArtifactFormat.PDF
                else b"docx-test"
            ),
        )


@pytest.fixture
def generation_service() -> FakeFinalCVGenerationService:
    """Provide one successful final-CV service."""

    return FakeFinalCVGenerationService()


@pytest.fixture
def retrieval_service() -> FakeCVArtifactRetrievalService:
    """Provide one successful retrieval service."""

    return FakeCVArtifactRetrievalService()


@pytest.fixture
def client(
    generation_service: FakeFinalCVGenerationService,
    retrieval_service: FakeCVArtifactRetrievalService,
) -> Iterator[TestClient]:
    """Create API client with final-CV workflow override."""

    app.dependency_overrides[get_final_cv_generation_service] = lambda: (
        generation_service
    )

    app.dependency_overrides[get_cv_artifact_retrieval_service] = lambda: (
        retrieval_service
    )

    try:
        with TestClient(app) as test_client:
            yield test_client

    finally:
        app.dependency_overrides.pop(
            get_final_cv_generation_service,
            None,
        )

        app.dependency_overrides.pop(
            get_cv_artifact_retrieval_service,
            None,
        )


def request_payload() -> dict[str, str]:
    """Return one valid final-CV request."""

    return {
        "thread_id": "THR-001",
        "source_document_id": "DOC-001",
    }


def test_generate_final_cv_returns_safe_verified_metadata(
    client: TestClient,
    generation_service: FakeFinalCVGenerationService,
) -> None:
    """Successful generation should expose safe verified metadata."""

    response = client.post(
        "/api/v1/cv-versions",
        headers={"X-User-ID": "USER-001"},
        json=request_payload(),
    )

    assert response.status_code == 200

    body = response.json()

    assert body["cv_version_id"] == ("CVV-API-001")

    assert body["status"] == "verified"

    assert body["version_number"] == 1

    assert body["source_document_id"] == ("DOC-001")

    assert body["thread_id"] == "THR-001"

    assert body["reused_existing_version"] is False

    assert {artifact["artifact_format"] for artifact in body["artifacts"]} == {
        "docx",
        "pdf",
    }

    assert all(
        artifact["verification_status"] == "verified" for artifact in body["artifacts"]
    )

    serialized = response.text

    assert "storage_key" not in serialized
    assert "private/docx" not in serialized
    assert "private/pdf" not in serialized

    assert "EVD-001" not in serialized
    assert "REQ-001" not in serialized
    assert "CVP-001" not in serialized

    assert generation_service.call_count == 1


def test_exact_api_retry_can_report_reused_version(
    client: TestClient,
    generation_service: FakeFinalCVGenerationService,
) -> None:
    """Recovered generation should be explicit to the caller."""

    generation_service.reused_existing = True

    response = client.post(
        "/api/v1/cv-versions",
        headers={"X-User-ID": "USER-001"},
        json=request_payload(),
    )

    assert response.status_code == 200

    assert response.json()["reused_existing_version"] is True


def test_missing_user_header_returns_401(
    client: TestClient,
) -> None:
    """Final CV generation requires the authenticated user boundary."""

    response = client.post(
        "/api/v1/cv-versions",
        json=request_payload(),
    )

    assert response.status_code == 401


def test_missing_source_document_returns_404(
    client: TestClient,
    generation_service: FakeFinalCVGenerationService,
) -> None:
    """Unavailable user source documents should remain opaque."""

    generation_service.error = CareerDocumentUnavailableError(
        "The career document is unavailable."
    )

    response = client.post(
        "/api/v1/cv-versions",
        headers={"X-User-ID": "USER-001"},
        json=request_payload(),
    )

    assert response.status_code == 404


def test_ineligible_workflow_state_returns_409(
    client: TestClient,
    generation_service: FakeFinalCVGenerationService,
) -> None:
    """Valid requests cannot bypass final workflow eligibility."""

    generation_service.error = FinalCVGenerationError(
        "Final CV generation is not eligible."
    )

    response = client.post(
        "/api/v1/cv-versions",
        headers={"X-User-ID": "USER-001"},
        json=request_payload(),
    )

    assert response.status_code == 409


def test_pdf_infrastructure_failure_returns_502(
    client: TestClient,
    generation_service: FakeFinalCVGenerationService,
) -> None:
    """Converter failures should not be represented as client errors."""

    generation_service.error = CVPDFConversionError(
        "LibreOffice PDF conversion failed."
    )

    response = client.post(
        "/api/v1/cv-versions",
        headers={"X-User-ID": "USER-001"},
        json=request_payload(),
    )

    assert response.status_code == 502

    assert response.json()["detail"] == (
        "The final CV artifact pipeline could not complete successfully."
    )


def test_get_cv_version_returns_safe_metadata(
    client: TestClient,
) -> None:
    """Version retrieval should never expose storage or provenance internals."""

    response = client.get(
        "/api/v1/cv-versions/CVV-API-001",
        headers={"X-User-ID": "USER-001"},
    )

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "verified"
    assert body["cv_version_id"] == ("CVV-API-001")

    serialized = response.text

    assert "storage_key" not in serialized
    assert "private/docx" not in serialized
    assert "private/pdf" not in serialized

    assert "EVD-001" not in serialized
    assert "REQ-001" not in serialized
    assert "CVP-001" not in serialized


def test_download_verified_docx_returns_attachment(
    client: TestClient,
) -> None:
    """Verified DOCX should return private bytes as an attachment."""

    response = client.get(
        ("/api/v1/cv-versions/CVV-API-001/artifacts/docx"),
        headers={"X-User-ID": "USER-001"},
    )

    assert response.status_code == 200
    assert response.content == b"docx-test"

    assert response.headers["content-disposition"] == (
        'attachment; filename="CVV-API-001.docx"'
    )

    assert response.headers["x-content-type-options"] == "nosniff"


def test_download_verified_pdf_returns_attachment(
    client: TestClient,
) -> None:
    """Verified PDF should return private bytes as an attachment."""

    response = client.get(
        ("/api/v1/cv-versions/CVV-API-001/artifacts/pdf"),
        headers={"X-User-ID": "USER-001"},
    )

    assert response.status_code == 200
    assert response.content == (b"%PDF-test")

    assert response.headers["content-type"].startswith("application/pdf")


def test_unavailable_cv_version_returns_404(
    client: TestClient,
    retrieval_service: FakeCVArtifactRetrievalService,
) -> None:
    """Missing and cross-user versions remain indistinguishable."""

    retrieval_service.error = CVArtifactRetrievalError("The CV version is unavailable.")

    response = client.get(
        "/api/v1/cv-versions/CVV-API-001",
        headers={"X-User-ID": "USER-001"},
    )

    assert response.status_code == 404

    assert response.json()["detail"] == ("The CV version is unavailable.")


def test_unavailable_artifact_returns_404(
    client: TestClient,
    retrieval_service: FakeCVArtifactRetrievalService,
) -> None:
    """Unverified or unavailable files must never leak state."""

    retrieval_service.error = CVArtifactRetrievalError(
        "The requested CV artifact is not verified."
    )

    response = client.get(
        ("/api/v1/cv-versions/CVV-API-001/artifacts/pdf"),
        headers={"X-User-ID": "USER-001"},
    )

    assert response.status_code == 404

    assert response.json()["detail"] == ("The CV artifact is unavailable.")


def test_invalid_artifact_format_is_rejected(
    client: TestClient,
) -> None:
    """Only DOCX and PDF download formats are accepted."""

    response = client.get(
        ("/api/v1/cv-versions/CVV-API-001/artifacts/exe"),
        headers={"X-User-ID": "USER-001"},
    )

    assert response.status_code == 422
