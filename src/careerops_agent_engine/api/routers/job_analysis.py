"""Endpoints for durable CareerOps job analysis."""

from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
)

from careerops_agent_engine.api.dependencies import (
    get_authenticated_user_id,
    get_job_analysis_service,
)
from careerops_agent_engine.api.schemas.job_analysis import (
    AuditEventResponse,
    CVProposalReviewResponse,
    JobAnalysisAwaitingReviewResponse,
    JobAnalysisCompletedResponse,
    JobAnalysisRequest,
    JobAnalysisResponse,
)
from careerops_agent_engine.application.exceptions import (
    CVReviewValidationError,
    JobAnalysisThreadUnavailableError,
)
from careerops_agent_engine.application.services.job_analysis import (
    JobAnalysisExecutionResult,
    JobAnalysisService,
)
from careerops_agent_engine.domain.enums import (
    ApprovalStatus,
)
from careerops_agent_engine.domain.models.approval import (
    CVReviewDecision,
)
from careerops_agent_engine.domain.models.cv import (
    CVChangeProposal,
)
from careerops_agent_engine.domain.models.evidence import (
    EvidenceMatch,
)
from careerops_agent_engine.domain.models.job import (
    JobRequirement,
)
from careerops_agent_engine.domain.models.verification import (
    CVClaimVerificationReport,
)

router = APIRouter(
    prefix="/api/v1/job-analysis",
    tags=["Job Analysis"],
)

JobAnalysisServiceDependency = Annotated[
    JobAnalysisService,
    Depends(get_job_analysis_service),
]

AuthenticatedUserIdDependency = Annotated[
    str,
    Depends(get_authenticated_user_id),
]


def build_job_analysis_response(
    execution: JobAnalysisExecutionResult,
) -> JobAnalysisResponse:
    """Convert durable graph execution into an API response."""

    state = execution.state

    fit_score = state.get("fit_score")

    if fit_score is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=("The workflow did not calculate a fit score."),
        )

    requirements = [
        JobRequirement.model_validate(payload)
        for payload in state.get("requirements", [])
    ]

    evidence_matches = [
        EvidenceMatch.model_validate(payload)
        for payload in state.get("evidence_matches", [])
    ]

    cv_proposals = [
        CVChangeProposal.model_validate(payload)
        for payload in state.get("cv_proposals", [])
    ]

    verification_reports = [
        CVClaimVerificationReport.model_validate(payload)
        for payload in state.get(
            "claim_verification_reports",
            [],
        )
    ]

    audit_events = [
        AuditEventResponse.model_validate(event)
        for event in state.get("audit_events", [])
    ]

    if execution.awaiting_review:
        interrupt_payload = execution.interrupt_payload

        if interrupt_payload is None:
            raise HTTPException(
                status_code=(status.HTTP_500_INTERNAL_SERVER_ERROR),
                detail=(
                    "The workflow reported a review pause without a review payload."
                ),
            )

        review = CVProposalReviewResponse.model_validate(interrupt_payload)

        return JobAnalysisAwaitingReviewResponse(
            status="awaiting_review",
            thread_id=execution.thread_id,
            job_id=state["job_id"],
            role_title=state.get("role_title"),
            requirements=requirements,
            evidence_matches=evidence_matches,
            fit_score=fit_score,
            cv_proposals=cv_proposals,
            claim_verification_reports=(verification_reports),
            reviewable_proposal_ids=state.get(
                "reviewable_proposal_ids",
                [],
            ),
            blocked_proposal_ids=state.get(
                "blocked_proposal_ids",
                [],
            ),
            audit_events=audit_events,
            review=review,
        )

    if state.get("status") != "completed":
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "The workflow stopped without completing or requesting human review."
            ),
        )

    review_status_value = state.get("review_status")

    review_status = (
        ApprovalStatus(review_status_value) if review_status_value is not None else None
    )

    final_cv_proposals = [
        CVChangeProposal.model_validate(payload)
        for payload in state.get(
            "final_cv_proposals",
            [],
        )
    ]

    return JobAnalysisCompletedResponse(
        status="completed",
        thread_id=execution.thread_id,
        job_id=state["job_id"],
        role_title=state.get("role_title"),
        requirements=requirements,
        evidence_matches=evidence_matches,
        fit_score=fit_score,
        cv_proposals=cv_proposals,
        claim_verification_reports=verification_reports,
        reviewable_proposal_ids=state.get(
            "reviewable_proposal_ids",
            [],
        ),
        blocked_proposal_ids=state.get(
            "blocked_proposal_ids",
            [],
        ),
        audit_events=audit_events,
        review_status=review_status,
        final_cv_proposals=final_cv_proposals,
    )


@router.post(
    "",
    response_model=JobAnalysisResponse,
    summary="Start job analysis",
    status_code=status.HTTP_200_OK,
)
def analyse_job(
    request: JobAnalysisRequest,
    service: JobAnalysisServiceDependency,
    user_id: AuthenticatedUserIdDependency,
) -> JobAnalysisResponse:
    """Start an evidence-grounded durable job workflow."""

    try:
        execution = service.analyse(
            job_id=request.job_id,
            user_id=user_id,
            job_description=request.job_description,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc

    state = execution.state

    if state.get("status") == "invalid":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(state.get("validation_error") or "The job description is invalid."),
        )

    return build_job_analysis_response(execution)


@router.post(
    "/{thread_id}/review",
    response_model=JobAnalysisResponse,
    summary="Resume human CV review",
    status_code=status.HTTP_200_OK,
)
def review_job_analysis(
    thread_id: str,
    decision: CVReviewDecision,
    service: JobAnalysisServiceDependency,
    user_id: AuthenticatedUserIdDependency,
) -> JobAnalysisResponse:
    """Resume a paused workflow with a human review decision."""

    try:
        execution = service.resume_review(
            thread_id=thread_id,
            user_id=user_id,
            decision=decision,
        )
    except CVReviewValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc
    except JobAnalysisThreadUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    return build_job_analysis_response(execution)
