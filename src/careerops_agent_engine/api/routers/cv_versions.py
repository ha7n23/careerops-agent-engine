"""Endpoints for trusted final-CV generation."""

from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Response,
    status,
)

from careerops_agent_engine.api.dependencies import (
    get_authenticated_user_id,
    get_cv_artifact_retrieval_service,
    get_final_cv_generation_service,
)
from careerops_agent_engine.api.schemas.cv_versions import (
    CVVersionResponse,
    FinalCVGenerationRequest,
    FinalCVVersionResponse,
)
from careerops_agent_engine.application.exceptions import (
    CareerDocumentUnavailableError,
    CVArtifactRetrievalError,
    CVArtifactVerificationError,
    CVPDFConversionError,
    CVRenderingError,
    FinalCVGenerationError,
    StructuredCVAssemblyError,
)
from careerops_agent_engine.application.services.cv_artifact_retrieval import (
    CVArtifactRetrievalService,
)
from careerops_agent_engine.application.services.final_cv_generation import (
    FinalCVGenerationService,
)
from careerops_agent_engine.domain.enums import (
    CVArtifactFormat,
)

router = APIRouter(
    prefix="/api/v1/cv-versions",
    tags=["CV Versions"],
)

FinalCVGenerationServiceDependency = Annotated[
    FinalCVGenerationService,
    Depends(get_final_cv_generation_service),
]

AuthenticatedUserIdDependency = Annotated[
    str,
    Depends(get_authenticated_user_id),
]

CVArtifactRetrievalServiceDependency = Annotated[
    CVArtifactRetrievalService,
    Depends(get_cv_artifact_retrieval_service),
]


@router.post(
    "",
    response_model=FinalCVVersionResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate a verified final CV",
)
def generate_final_cv(
    request: FinalCVGenerationRequest,
    service: FinalCVGenerationServiceDependency,
    user_id: AuthenticatedUserIdDependency,
) -> FinalCVVersionResponse:
    """Generate or recover verified DOCX and PDF artifacts."""

    try:
        execution = service.generate(
            user_id=user_id,
            thread_id=request.thread_id,
            source_document_id=(request.source_document_id),
        )

    except CareerDocumentUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    except (
        StructuredCVAssemblyError,
        FinalCVGenerationError,
    ) as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    except (
        CVRenderingError,
        CVArtifactVerificationError,
        CVPDFConversionError,
    ) as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=("The final CV artifact pipeline could not complete successfully."),
        ) from exc

    return FinalCVVersionResponse.from_execution(execution)


@router.get(
    "/{cv_version_id}",
    response_model=CVVersionResponse,
    summary="Get a generated CV version",
)
def get_cv_version(
    cv_version_id: str,
    service: CVArtifactRetrievalServiceDependency,
    user_id: AuthenticatedUserIdDependency,
) -> CVVersionResponse:
    """Return safe metadata for one user-owned CV version."""

    try:
        version = service.get_version(
            user_id=user_id,
            cv_version_id=cv_version_id,
        )

    except CVArtifactRetrievalError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="The CV version is unavailable.",
        ) from exc

    return CVVersionResponse.from_domain(version)


@router.get(
    "/{cv_version_id}/artifacts/{artifact_format}",
    summary="Download a verified generated CV artifact",
)
def download_cv_artifact(
    cv_version_id: str,
    artifact_format: CVArtifactFormat,
    service: CVArtifactRetrievalServiceDependency,
    user_id: AuthenticatedUserIdDependency,
) -> Response:
    """Return verified DOCX or PDF bytes without exposing storage paths."""

    try:
        retrieved = service.get_verified_artifact(
            user_id=user_id,
            cv_version_id=cv_version_id,
            artifact_format=artifact_format,
        )

    except CVArtifactRetrievalError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="The CV artifact is unavailable.",
        ) from exc

    return Response(
        content=retrieved.data,
        media_type=retrieved.media_type,
        headers={
            "Content-Disposition": (f'attachment; filename="{retrieved.filename}"'),
            "X-Content-Type-Options": "nosniff",
        },
    )
