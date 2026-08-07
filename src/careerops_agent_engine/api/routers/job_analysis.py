"""Endpoints for CareerOps job analysis."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from careerops_agent_engine.api.dependencies import (
    get_authenticated_user_id,
    get_job_analysis_service,
)
from careerops_agent_engine.api.schemas.job_analysis import (
    AuditEventResponse,
    JobAnalysisRequest,
    JobAnalysisResponse,
)
from careerops_agent_engine.application.services.job_analysis import (
    JobAnalysisService,
)
from careerops_agent_engine.domain.models.cv import (
    CVChangeProposal,
)
from careerops_agent_engine.domain.models.evidence import EvidenceMatch
from careerops_agent_engine.domain.models.job import JobRequirement
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


@router.post(
    "",
    response_model=JobAnalysisResponse,
    summary="Analyse a job description",
    status_code=status.HTTP_200_OK,
)
def analyse_job(
    request: JobAnalysisRequest,
    service: JobAnalysisServiceDependency,
    user_id: AuthenticatedUserIdDependency,
) -> JobAnalysisResponse:
    """Analyse a job and produce evidence-verified CV proposals."""

    try:
        result = service.analyse(
            job_id=request.job_id,
            user_id=user_id,
            job_description=request.job_description,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc

    if result.get("status") == "invalid":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                result.get("validation_error") or "The job description is invalid."
            ),
        )

    fit_score = result.get("fit_score")

    if fit_score is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=("The workflow completed without calculating a fit score."),
        )

    requirements = [
        JobRequirement.model_validate(requirement)
        for requirement in result.get("requirements", [])
    ]

    evidence_matches = [
        EvidenceMatch.model_validate(match)
        for match in result.get("evidence_matches", [])
    ]

    cv_proposals = [
        CVChangeProposal.model_validate(proposal)
        for proposal in result.get("cv_proposals", [])
    ]

    claim_verification_reports = [
        CVClaimVerificationReport.model_validate(report)
        for report in result.get(
            "claim_verification_reports",
            [],
        )
    ]

    audit_events = [
        AuditEventResponse.model_validate(event)
        for event in result.get("audit_events", [])
    ]

    return JobAnalysisResponse(
        status="completed",
        job_id=result["job_id"],
        role_title=result.get("role_title"),
        requirements=requirements,
        evidence_matches=evidence_matches,
        fit_score=fit_score,
        cv_proposals=cv_proposals,
        claim_verification_reports=(claim_verification_reports),
        reviewable_proposal_ids=result.get(
            "reviewable_proposal_ids",
            [],
        ),
        blocked_proposal_ids=result.get(
            "blocked_proposal_ids",
            [],
        ),
        audit_events=audit_events,
    )
