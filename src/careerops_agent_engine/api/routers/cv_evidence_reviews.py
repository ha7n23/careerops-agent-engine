"""Endpoints for durable human CV evidence review."""

from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
)

from careerops_agent_engine.api.dependencies import (
    get_authenticated_user_id,
    get_cv_evidence_workflow_service,
)
from careerops_agent_engine.api.schemas.cv_documents import (
    CVEvidenceReviewRunResponse,
)
from careerops_agent_engine.application.exceptions import (
    CVEvidenceReviewRunUnavailableError,
    CVEvidenceReviewValidationError,
)
from careerops_agent_engine.application.services.cv_evidence_workflow import (
    CVEvidenceWorkflowService,
)
from careerops_agent_engine.domain.models.evidence_review import (
    EvidenceReviewDecision,
)

router = APIRouter(
    prefix="/api/v1/cv-evidence-reviews",
    tags=["CV Evidence Reviews"],
)

CVEvidenceWorkflowServiceDependency = Annotated[
    CVEvidenceWorkflowService,
    Depends(get_cv_evidence_workflow_service),
]

AuthenticatedUserIdDependency = Annotated[
    str,
    Depends(get_authenticated_user_id),
]


@router.get(
    "/{review_run_id}",
    response_model=CVEvidenceReviewRunResponse,
    status_code=status.HTTP_200_OK,
    summary="Get CV evidence review",
)
def get_cv_evidence_review(
    review_run_id: str,
    service: CVEvidenceWorkflowServiceDependency,
    user_id: AuthenticatedUserIdDependency,
) -> CVEvidenceReviewRunResponse:
    """Recover one persisted user-owned review run."""

    try:
        snapshot = service.get_review(
            user_id=user_id,
            review_run_id=review_run_id,
        )

    except CVEvidenceReviewRunUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    return CVEvidenceReviewRunResponse.from_domain(snapshot)


@router.post(
    "/{review_run_id}/review",
    response_model=CVEvidenceReviewRunResponse,
    status_code=status.HTTP_200_OK,
    summary="Submit CV evidence review",
)
def submit_cv_evidence_review(
    review_run_id: str,
    decision: EvidenceReviewDecision,
    service: CVEvidenceWorkflowServiceDependency,
    user_id: AuthenticatedUserIdDependency,
) -> CVEvidenceReviewRunResponse:
    """Apply an explicit human decision to persisted evidence proposals."""

    try:
        snapshot = service.submit_review(
            user_id=user_id,
            review_run_id=review_run_id,
            decision=decision,
        )

    except CVEvidenceReviewValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc

    except CVEvidenceReviewRunUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    return CVEvidenceReviewRunResponse.from_domain(snapshot)
