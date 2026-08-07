"""Nodes for the CareerOps job-analysis graph."""

from collections.abc import Callable

from careerops_agent_engine.agents.states.job_analysis import (
    JobAnalysisState,
    JobAnalysisUpdate,
)
from careerops_agent_engine.application.ports.evidence_discovery import (
    EvidenceDiscoveryRunner,
)
from careerops_agent_engine.application.ports.requirement_extractor import (
    RequirementExtractor,
)
from careerops_agent_engine.application.services.cv_claim_verification import (
    CVClaimVerificationService,
)
from careerops_agent_engine.application.services.cv_proposals import (
    CVProposalGenerationService,
)
from careerops_agent_engine.application.services.fit_scoring import (
    calculate_evidence_weighted_fit,
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

MINIMUM_JOB_DESCRIPTION_LENGTH = 20


def validate_job_input(
    state: JobAnalysisState,
) -> JobAnalysisUpdate:
    """Validate and normalise the supplied job description."""

    normalised_description = state["job_description"].strip()

    if len(normalised_description) < MINIMUM_JOB_DESCRIPTION_LENGTH:
        return {
            "job_description": normalised_description,
            "status": "invalid",
            "validation_error": (
                "Job description must contain at least "
                f"{MINIMUM_JOB_DESCRIPTION_LENGTH} characters."
            ),
            "audit_events": [
                {
                    "node": "validate_job_input",
                    "event": "job_input_invalid",
                }
            ],
        }

    return {
        "job_description": normalised_description,
        "status": "validated",
        "validation_error": None,
        "audit_events": [
            {
                "node": "validate_job_input",
                "event": "job_input_validated",
            }
        ],
    }


def create_extract_requirements_node(
    extractor: RequirementExtractor,
) -> Callable[[JobAnalysisState], JobAnalysisUpdate]:
    """Create a graph node using the configured extraction adapter."""

    def extract_requirements(
        state: JobAnalysisState,
    ) -> JobAnalysisUpdate:
        extraction = extractor.extract(
            state["job_description"],
            job_id=state["job_id"],
        )

        return {
            "role_title": extraction.role_title,
            "requirements": [
                requirement.model_dump(mode="json")
                for requirement in extraction.requirements
            ],
            "audit_events": [
                {
                    "node": "extract_requirements",
                    "event": "requirements_extracted",
                }
            ],
        }

    return extract_requirements


def create_discover_evidence_node(
    evidence_discovery_runner: EvidenceDiscoveryRunner,
) -> Callable[[JobAnalysisState], JobAnalysisUpdate]:
    """Create a node that discovers evidence for every requirement."""

    def discover_evidence(
        state: JobAnalysisState,
    ) -> JobAnalysisUpdate:
        requirements = [
            JobRequirement.model_validate(payload)
            for payload in state.get("requirements", [])
        ]

        matches = [
            evidence_discovery_runner.discover(
                requirement,
                user_id=state["user_id"],
            )
            for requirement in requirements
        ]

        return {
            "evidence_matches": [match.model_dump(mode="json") for match in matches],
            "audit_events": [
                {
                    "node": "discover_evidence",
                    "event": "evidence_discovery_completed",
                }
            ],
        }

    return discover_evidence


def calculate_fit(
    state: JobAnalysisState,
) -> JobAnalysisUpdate:
    """Calculate a deterministic score from evidence matches."""

    requirements = [
        JobRequirement.model_validate(payload)
        for payload in state.get("requirements", [])
    ]

    evidence_matches = [
        EvidenceMatch.model_validate(payload)
        for payload in state.get("evidence_matches", [])
    ]

    fit_score = calculate_evidence_weighted_fit(
        requirements=requirements,
        evidence_matches=evidence_matches,
    )

    return {
        "fit_score": fit_score,
        "audit_events": [
            {
                "node": "calculate_fit",
                "event": "fit_score_calculated",
            }
        ],
    }


def create_generate_cv_proposals_node(
    proposal_service: CVProposalGenerationService,
) -> Callable[[JobAnalysisState], JobAnalysisUpdate]:
    """Generate proposals only for directly supported requirements."""

    def generate_cv_proposals(
        state: JobAnalysisState,
    ) -> JobAnalysisUpdate:
        requirements = [
            JobRequirement.model_validate(payload)
            for payload in state.get("requirements", [])
        ]

        evidence_matches = [
            EvidenceMatch.model_validate(payload)
            for payload in state.get("evidence_matches", [])
        ]

        matches_by_requirement = {
            match.requirement_id: match for match in evidence_matches
        }

        proposals: list[CVChangeProposal] = []

        for requirement in requirements:
            evidence_match = matches_by_requirement.get(requirement.requirement_id)

            if evidence_match is None:
                raise ValueError(
                    "Missing evidence match for requirement: "
                    f"{requirement.requirement_id}"
                )

            proposal = proposal_service.generate_for_requirement(
                job_id=state["job_id"],
                user_id=state["user_id"],
                requirement=requirement,
                evidence_match=evidence_match,
            )

            if proposal is not None:
                proposals.append(proposal)

        return {
            "cv_proposals": [
                proposal.model_dump(mode="json") for proposal in proposals
            ],
            "audit_events": [
                {
                    "node": "generate_cv_proposals",
                    "event": "cv_proposals_generated",
                }
            ],
        }

    return generate_cv_proposals


def create_verify_cv_proposals_node(
    verification_service: CVClaimVerificationService,
) -> Callable[[JobAnalysisState], JobAnalysisUpdate]:
    """Verify generated wording before human review is allowed."""

    def verify_cv_proposals(
        state: JobAnalysisState,
    ) -> JobAnalysisUpdate:
        proposals = [
            CVChangeProposal.model_validate(payload)
            for payload in state.get("cv_proposals", [])
        ]

        reports: list[dict[str, object]] = []
        reviewable_proposal_ids: list[str] = []
        blocked_proposal_ids: list[str] = []

        for proposal in proposals:
            report = verification_service.verify_proposal(
                user_id=state["user_id"],
                proposal=proposal,
            )

            reports.append(report.model_dump(mode="json"))

            if report.fully_supported:
                reviewable_proposal_ids.append(proposal.proposal_id)
            else:
                blocked_proposal_ids.append(proposal.proposal_id)

        return {
            "claim_verification_reports": reports,
            "reviewable_proposal_ids": (reviewable_proposal_ids),
            "blocked_proposal_ids": blocked_proposal_ids,
            "audit_events": [
                {
                    "node": "verify_cv_proposals",
                    "event": "cv_proposals_verified",
                }
            ],
        }

    return verify_cv_proposals


def complete_analysis(
    state: JobAnalysisState,
) -> JobAnalysisUpdate:
    """Mark a valid analysis as complete."""

    del state

    return {
        "status": "completed",
        "audit_events": [
            {
                "node": "complete_analysis",
                "event": "job_analysis_completed",
            }
        ],
    }


def mark_invalid(
    state: JobAnalysisState,
) -> JobAnalysisUpdate:
    """Finish invalid input without running later nodes."""

    del state

    return {
        "status": "invalid",
        "audit_events": [
            {
                "node": "mark_invalid",
                "event": "job_analysis_rejected",
            }
        ],
    }
