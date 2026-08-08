"""Endpoints for validated career-document ingestion."""

from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    UploadFile,
    status,
)

from careerops_agent_engine.api.dependencies import (
    get_authenticated_user_id,
    get_cv_document_upload_service,
)
from careerops_agent_engine.api.schemas.cv_documents import (
    CareerDocumentUploadResponse,
)
from careerops_agent_engine.application.exceptions import (
    DocumentUploadValidationError,
)
from careerops_agent_engine.application.services.cv_document_upload import (
    CVDocumentUploadService,
)

router = APIRouter(
    prefix="/api/v1/cv-documents",
    tags=["CV Documents"],
)

CVDocumentUploadServiceDependency = Annotated[
    CVDocumentUploadService,
    Depends(get_cv_document_upload_service),
]

AuthenticatedUserIdDependency = Annotated[
    str,
    Depends(get_authenticated_user_id),
]


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
    service: CVDocumentUploadServiceDependency,
    user_id: AuthenticatedUserIdDependency,
) -> CareerDocumentUploadResponse:
    """Validate and securely store one user-owned CV document."""

    try:
        # Read only enough bytes to determine whether the
        # configured limit has been exceeded.
        data = await file.read(service.max_upload_bytes + 1)

        document = service.upload(
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
