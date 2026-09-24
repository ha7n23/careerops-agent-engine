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
    CareerEvidenceResponse,
    EvidenceRegistryListResponse,
)
from careerops_agent_engine.application.exceptions import (
    CareerEvidenceUnavailableError,
)
from careerops_agent_engine.application.services.evidence_registry import (
    EvidenceRegistryService,
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
    limit: Annotated[
        int,
        Query(ge=1, le=100),
    ] = 100,
) -> EvidenceRegistryListResponse:
    """List approved evidence belonging to the authenticated user."""

    evidence = service.list_approved(
        user_id=user_id,
        limit=limit,
    )

    return EvidenceRegistryListResponse.from_domain(
        evidence,
        limit=limit,
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
