"""Read-only endpoints for the approved Evidence Registry."""

from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    status,
)

from careerops_agent_engine.api.dependencies import (
    get_authenticated_user_id,
    get_evidence_registry_service,
)
from careerops_agent_engine.api.schemas.evidence import (
    CareerEvidenceEditRequest,
    CareerEvidenceResponse,
    EvidenceRegistryListResponse,
)
from careerops_agent_engine.application.exceptions import (
    CareerEvidenceEditValidationError,
    CareerEvidenceUnavailableError,
)
from careerops_agent_engine.application.services.evidence_registry import (
    EvidenceRegistryService,
)
from careerops_agent_engine.domain.enums import (
    EvidenceCategory,
    EvidenceLifecycleStatus,
)

router = APIRouter(
    prefix="/api/v1/evidence",
    tags=["Evidence Registry"],
)

EvidenceRegistryServiceDependency = Annotated[
    EvidenceRegistryService,
    Depends(get_evidence_registry_service),
]

AuthenticatedUserIdDependency = Annotated[
    str,
    Depends(get_authenticated_user_id),
]


@router.get(
    "",
    response_model=EvidenceRegistryListResponse,
    status_code=status.HTTP_200_OK,
    summary="List approved evidence",
)
def list_approved_evidence(
    service: EvidenceRegistryServiceDependency,
    user_id: AuthenticatedUserIdDependency,
    query: Annotated[
        str | None,
        Query(alias="q", min_length=1, max_length=200),
    ] = None,
    category: Annotated[
        EvidenceCategory | None,
        Query(),
    ] = None,
    lifecycle_status: Annotated[
        EvidenceLifecycleStatus,
        Query(),
    ] = EvidenceLifecycleStatus.ACTIVE,
    offset: Annotated[
        int,
        Query(ge=0, le=10_000),
    ] = 0,
    limit: Annotated[
        int,
        Query(ge=1, le=100),
    ] = 100,
) -> EvidenceRegistryListResponse:
    """Search, filter and page approved evidence for the authenticated user."""

    page = service.query_approved(
        user_id=user_id,
        query=query,
        category=category,
        lifecycle_status=lifecycle_status,
        offset=offset,
        limit=limit,
    )

    return EvidenceRegistryListResponse.from_domain(
        page,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/{evidence_id}",
    response_model=CareerEvidenceResponse,
    status_code=status.HTTP_200_OK,
    summary="Get approved evidence",
)
def get_approved_evidence(
    evidence_id: str,
    service: EvidenceRegistryServiceDependency,
    user_id: AuthenticatedUserIdDependency,
) -> CareerEvidenceResponse:
    """Retrieve one approved evidence record within the user boundary."""

    try:
        evidence = service.get_approved(
            user_id=user_id,
            evidence_id=evidence_id,
        )

    except CareerEvidenceUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    return CareerEvidenceResponse.from_domain(evidence)


@router.patch(
    "/{evidence_id}",
    response_model=CareerEvidenceResponse,
    status_code=status.HTTP_200_OK,
    summary="Edit approved evidence",
)
def edit_approved_evidence(
    evidence_id: str,
    request: CareerEvidenceEditRequest,
    service: EvidenceRegistryServiceDependency,
    user_id: AuthenticatedUserIdDependency,
) -> CareerEvidenceResponse:
    """Edit safe fields while preserving ownership and provenance."""

    try:
        evidence = service.edit_approved(
            user_id=user_id,
            evidence_id=evidence_id,
            edit=request.to_domain(),
        )
    except CareerEvidenceUnavailableError as exc:
        raise unavailable_http_error(exc) from exc
    except CareerEvidenceEditValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc

    return CareerEvidenceResponse.from_domain(evidence)


@router.post(
    "/{evidence_id}/archive",
    response_model=CareerEvidenceResponse,
    status_code=status.HTTP_200_OK,
    summary="Archive approved evidence",
)
def archive_approved_evidence(
    evidence_id: str,
    service: EvidenceRegistryServiceDependency,
    user_id: AuthenticatedUserIdDependency,
) -> CareerEvidenceResponse:
    """Recoverably remove evidence from active use."""

    try:
        evidence = service.archive_approved(
            user_id=user_id,
            evidence_id=evidence_id,
        )
    except CareerEvidenceUnavailableError as exc:
        raise unavailable_http_error(exc) from exc

    return CareerEvidenceResponse.from_domain(evidence)


@router.post(
    "/{evidence_id}/restore",
    response_model=CareerEvidenceResponse,
    status_code=status.HTTP_200_OK,
    summary="Restore approved evidence",
)
def restore_approved_evidence(
    evidence_id: str,
    service: EvidenceRegistryServiceDependency,
    user_id: AuthenticatedUserIdDependency,
) -> CareerEvidenceResponse:
    """Return archived evidence to active downstream use."""

    try:
        evidence = service.restore_approved(
            user_id=user_id,
            evidence_id=evidence_id,
        )
    except CareerEvidenceUnavailableError as exc:
        raise unavailable_http_error(exc) from exc

    return CareerEvidenceResponse.from_domain(evidence)


def unavailable_http_error(
    error: CareerEvidenceUnavailableError,
) -> HTTPException:
    """Map unknown and cross-user evidence to the same response."""

    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=str(error),
    )
