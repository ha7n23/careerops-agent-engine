"""Endpoints for validated career-document ingestion."""

from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Query,
    UploadFile,
    status,
)

from careerops_agent_engine.api.dependencies import (
    get_authenticated_user_id,
    get_cv_document_ingestion_service,
    get_cv_evidence_history_service,
    get_cv_evidence_workflow_service,
)
from careerops_agent_engine.api.schemas.cv_documents import (
    CareerDocumentHistoryResponse,
    CareerDocumentUploadResponse,
    CVEvidenceReviewRunResponse,
)
from careerops_agent_engine.application.exceptions import (
    CareerDocumentUnavailableError,
    CVEvidenceProposalValidationError,
    DocumentExtractionError,
    DocumentUploadValidationError,
)
from careerops_agent_engine.application.services.cv_document_ingestion import (
    CVDocumentIngestionService,
)
from careerops_agent_engine.application.services.cv_evidence_history import (
    DEFAULT_HISTORY_LIMIT,
    CVEvidenceHistoryService,
)
from careerops_agent_engine.application.services.cv_evidence_workflow import (
    CVEvidenceWorkflowService,
)

router = APIRouter(
    prefix="/api/v1/cv-documents",
    tags=["CV Documents"],
)

CVDocumentIngestionServiceDependency = Annotated[
    CVDocumentIngestionService,
    Depends(get_cv_document_ingestion_service),
]

CVEvidenceWorkflowServiceDependency = Annotated[
    CVEvidenceWorkflowService,
    Depends(get_cv_evidence_workflow_service),
]

AuthenticatedUserIdDependency = Annotated[
    str,
    Depends(get_authenticated_user_id),
]

CVEvidenceHistoryServiceDependency = Annotated[
    CVEvidenceHistoryService,
    Depends(get_cv_evidence_history_service),
]


@router.get(
    "",
    response_model=CareerDocumentHistoryResponse,
    status_code=status.HTTP_200_OK,
    summary="List uploaded CV documents",
)
def list_cv_documents(
    service: CVEvidenceHistoryServiceDependency,
    user_id: AuthenticatedUserIdDependency,
    limit: Annotated[
        int,
        Query(ge=1, le=100),
    ] = DEFAULT_HISTORY_LIMIT,
) -> CareerDocumentHistoryResponse:
    """List the authenticated user's newest uploaded CV documents."""

    summaries = service.list_documents(
        user_id=user_id,
        limit=limit,
    )

    return CareerDocumentHistoryResponse.from_domain(
        summaries,
        limit=limit,
    )


@router.post(
    "",
    response_model=CareerDocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a CV document",
)
async def upload_cv_document(
    file: Annotated[
        UploadFile,
        File(description=("Candidate CV in PDF or DOCX format.")),
    ],
    service: CVDocumentIngestionServiceDependency,
    user_id: AuthenticatedUserIdDependency,
) -> CareerDocumentUploadResponse:
    """Validate and securely store one user-owned CV document."""

    try:
        # Read only enough bytes to determine whether the
        # configured limit has been exceeded.
        data = await file.read(service.max_upload_bytes + 1)

        document = service.ingest(
            user_id=user_id,
            original_filename=(file.filename or ""),
            declared_media_type=(file.content_type),
            data=data,
        )

    except DocumentUploadValidationError as exc:
        raise HTTPException(
            status_code=(status.HTTP_422_UNPROCESSABLE_CONTENT),
            detail=str(exc),
        ) from exc

    except FileExistsError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "The document could not be stored "
                "because its identifier already exists."
            ),
        ) from exc

    finally:
        await file.close()

    return CareerDocumentUploadResponse.from_domain(document)


@router.post(
    "/{document_id}/evidence-review",
    response_model=CVEvidenceReviewRunResponse,
    status_code=status.HTTP_200_OK,
    summary="Start or recover CV evidence review",
)
def start_cv_evidence_review(
    document_id: str,
    service: CVEvidenceWorkflowServiceDependency,
    user_id: AuthenticatedUserIdDependency,
) -> CVEvidenceReviewRunResponse:
    """Create or recover the durable evidence-review state."""

    try:
        snapshot = service.start_review(
            user_id=user_id,
            document_id=document_id,
        )

    except CareerDocumentUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    except DocumentExtractionError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc

    except CVEvidenceProposalValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=("The CV evidence extractor returned an invalid structured result."),
        ) from exc

    return CVEvidenceReviewRunResponse.from_domain(snapshot)
