"""Endpoints for durable human CV evidence review."""

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
    get_cv_evidence_history_service,
    get_cv_evidence_workflow_service,
)
from careerops_agent_engine.api.schemas.cv_documents import (
    CVEvidenceReviewHistoryResponse,
    CVEvidenceReviewRunResponse,
)
from careerops_agent_engine.application.exceptions import (
    CVEvidenceReviewRunUnavailableError,
    CVEvidenceReviewValidationError,
)
from careerops_agent_engine.application.services.cv_evidence_history import (
    DEFAULT_HISTORY_LIMIT,
    CVEvidenceHistoryService,
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

CVEvidenceHistoryServiceDependency = Annotated[
    CVEvidenceHistoryService,
    Depends(get_cv_evidence_history_service),
]


@router.get(
    "",
    response_model=CVEvidenceReviewHistoryResponse,
    status_code=status.HTTP_200_OK,
    summary="List CV evidence reviews",
)
def list_cv_evidence_reviews(
    service: CVEvidenceHistoryServiceDependency,
    user_id: AuthenticatedUserIdDependency,
    limit: Annotated[
        int,
        Query(ge=1, le=100),
    ] = DEFAULT_HISTORY_LIMIT,
) -> CVEvidenceReviewHistoryResponse:
    """List the authenticated user's newest evidence-review runs."""

    summaries = service.list_review_runs(
        user_id=user_id,
        limit=limit,
    )

    return CVEvidenceReviewHistoryResponse.from_domain(
        summaries,
        limit=limit,
    )


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
