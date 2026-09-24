"""Nodes for the CareerOps job-analysis graph."""

from collections.abc import Callable

from langgraph.types import interrupt

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
from careerops_agent_engine.application.services.cv_review import (
    validate_review_decision,
)
from careerops_agent_engine.application.services.fit_scoring import (
    calculate_evidence_weighted_fit,
)
from careerops_agent_engine.domain.enums import (
    ApprovalStatus,
    ReviewAction,
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
    """Generate eligible proposals through one bounded batch call."""

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

        proposals = proposal_service.generate_for_requirements(
            job_id=state["job_id"],
            user_id=state["user_id"],
            requirements=requirements,
            evidence_matches=evidence_matches,
        )

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


def request_human_review(
    state: JobAnalysisState,
) -> JobAnalysisUpdate:
    """Pause until a human reviews verified CV proposals."""

    reviewable_ids = state.get(
        "reviewable_proposal_ids",
        [],
    )

    proposals = [
        CVChangeProposal.model_validate(payload)
        for payload in state.get("cv_proposals", [])
    ]

    reports = [
        CVClaimVerificationReport.model_validate(payload)
        for payload in state.get(
            "claim_verification_reports",
            [],
        )
    ]

    reviewable_id_set = set(reviewable_ids)

    reviewable_proposals = [
        proposal for proposal in proposals if proposal.proposal_id in reviewable_id_set
    ]

    reviewable_reports = [
        report for report in reports if report.proposal_id in reviewable_id_set
    ]

    raw_decision = interrupt(
        {
            "type": "cv_proposal_review",
            "proposals": [
                proposal.model_dump(mode="json") for proposal in reviewable_proposals
            ],
            "verification_reports": [
                report.model_dump(mode="json") for report in reviewable_reports
            ],
            "allowed_actions": [
                ReviewAction.APPROVE.value,
                ReviewAction.EDIT.value,
                ReviewAction.REGENERATE.value,
                ReviewAction.REJECT.value,
            ],
        }
    )

    decision = CVReviewDecision.model_validate(raw_decision)

    validate_review_decision(
        decision=decision,
        reviewable_proposal_ids=reviewable_ids,
        allowed_actions={
            ReviewAction.APPROVE,
            ReviewAction.EDIT,
            ReviewAction.REGENERATE,
            ReviewAction.REJECT,
        },
    )

    return {
        "review_decision": decision.model_dump(mode="json"),
        "review_target_proposal_ids": list(reviewable_ids),
        "audit_events": [
            {
                "node": "request_human_review",
                "event": "human_review_received",
            }
        ],
    }


def apply_human_edits(
    state: JobAnalysisState,
) -> JobAnalysisUpdate:
    """Apply human replacement text to proposals."""

    decision_payload = state.get("review_decision")

    if decision_payload is None:
        raise ValueError("Human edits require a review decision.")

    decision = CVReviewDecision.model_validate(decision_payload)

    if decision.action is not ReviewAction.EDIT:
        raise ValueError("Human-edit processing requires an edit decision.")

    target_ids = state.get(
        "review_target_proposal_ids",
        [],
    )

    validate_review_decision(
        decision=decision,
        reviewable_proposal_ids=target_ids,
        allowed_actions={
            ReviewAction.EDIT,
        },
    )

    edits_by_id = {edit.proposal_id: edit.edited_text for edit in decision.edits}

    proposals = [
        CVChangeProposal.model_validate(payload)
        for payload in state.get("cv_proposals", [])
    ]

    updated_proposals: list[CVChangeProposal] = []

    for proposal in proposals:
        edited_text = edits_by_id.get(proposal.proposal_id)

        if edited_text is None:
            updated_proposals.append(proposal)
            continue

        payload = proposal.model_dump(mode="python")
        payload["proposed_text"] = edited_text

        updated_proposals.append(CVChangeProposal.model_validate(payload))

    return {
        "cv_proposals": [
            proposal.model_dump(mode="json") for proposal in updated_proposals
        ],
        "edited_proposal_ids": sorted(edits_by_id),
        "regenerated_proposal_ids": [],
        "regeneration_verification_failed_ids": [],
        "audit_events": [
            {
                "node": "apply_human_edits",
                "event": "human_edits_applied",
            }
        ],
    }


def create_regenerate_cv_proposals_node(
    proposal_service: CVProposalGenerationService,
) -> Callable[[JobAnalysisState], JobAnalysisUpdate]:
    """Create a node that regenerates proposals from reviewer feedback."""

    def regenerate_cv_proposals(
        state: JobAnalysisState,
    ) -> JobAnalysisUpdate:
        decision_payload = state.get("review_decision")

        if decision_payload is None:
            raise ValueError("Proposal regeneration requires a human review decision.")

        decision = CVReviewDecision.model_validate(decision_payload)

        if decision.action is not ReviewAction.REGENERATE:
            raise ValueError("Proposal regeneration requires a regenerate decision.")

        target_ids = state.get(
            "review_target_proposal_ids",
            [],
        )

        validate_review_decision(
            decision=decision,
            reviewable_proposal_ids=target_ids,
            allowed_actions={
                ReviewAction.REGENERATE,
            },
        )

        reviewer_feedback = (decision.reviewer_comment or "").strip()

        if not reviewer_feedback:
            raise ValueError("Proposal regeneration requires reviewer feedback.")

        requirements = [
            JobRequirement.model_validate(payload)
            for payload in state.get(
                "requirements",
                [],
            )
        ]

        evidence_matches = [
            EvidenceMatch.model_validate(payload)
            for payload in state.get(
                "evidence_matches",
                [],
            )
        ]

        proposals = [
            CVChangeProposal.model_validate(payload)
            for payload in state.get(
                "cv_proposals",
                [],
            )
        ]

        requirements_by_id = {
            requirement.requirement_id: requirement for requirement in requirements
        }

        matches_by_requirement = {
            match.requirement_id: match for match in evidence_matches
        }

        target_id_set = set(target_ids)

        updated_proposals: list[CVChangeProposal] = []

        regenerated_ids: list[str] = []

        for proposal in proposals:
            if proposal.proposal_id not in target_id_set:
                updated_proposals.append(proposal)
                continue

            if len(proposal.requirement_ids) != 1:
                raise ValueError(
                    "Regeneration requires exactly one requirement per proposal."
                )

            requirement_id = proposal.requirement_ids[0]

            requirement = requirements_by_id.get(requirement_id)

            if requirement is None:
                raise ValueError(
                    "Regeneration requirement is "
                    "missing from workflow state: "
                    f"{requirement_id}"
                )

            evidence_match = matches_by_requirement.get(requirement_id)

            if evidence_match is None:
                raise ValueError(
                    "Regeneration evidence match is "
                    "missing from workflow state: "
                    f"{requirement_id}"
                )

            regenerated = proposal_service.regenerate_for_requirement(
                job_id=state["job_id"],
                user_id=state["user_id"],
                requirement=requirement,
                evidence_match=evidence_match,
                previous_proposal=proposal,
                reviewer_feedback=(reviewer_feedback),
            )

            updated_proposals.append(regenerated)
            regenerated_ids.append(regenerated.proposal_id)

        if set(regenerated_ids) != target_id_set:
            missing_ids = sorted(target_id_set - set(regenerated_ids))

            raise ValueError(
                "Regeneration targets were missing "
                "from workflow proposals: "
                f"{', '.join(missing_ids)}"
            )

        return {
            "cv_proposals": [
                proposal.model_dump(mode="json") for proposal in updated_proposals
            ],
            "regenerated_proposal_ids": sorted(regenerated_ids),
            "regeneration_verification_failed_ids": [],
            "regeneration_feedback": (reviewer_feedback),
            "edited_proposal_ids": [],
            "edit_verification_failed_ids": [],
            "audit_events": [
                {
                    "node": ("regenerate_cv_proposals"),
                    "event": ("cv_proposals_regenerated"),
                }
            ],
        }

    return regenerate_cv_proposals


def create_verify_regenerated_proposals_node(
    verification_service: CVClaimVerificationService,
) -> Callable[[JobAnalysisState], JobAnalysisUpdate]:
    """Verify regenerated proposals before another human review."""

    def verify_regenerated_proposals(
        state: JobAnalysisState,
    ) -> JobAnalysisUpdate:
        regenerated_ids = set(
            state.get(
                "regenerated_proposal_ids",
                [],
            )
        )

        if not regenerated_ids:
            raise ValueError(
                "Regeneration verification requires at least one regenerated proposal."
            )

        proposals = [
            CVChangeProposal.model_validate(payload)
            for payload in state.get(
                "cv_proposals",
                [],
            )
        ]

        existing_reports = [
            CVClaimVerificationReport.model_validate(payload)
            for payload in state.get(
                "claim_verification_reports",
                [],
            )
        ]

        proposals_by_id = {proposal.proposal_id: proposal for proposal in proposals}

        reports_by_id = {report.proposal_id: report for report in existing_reports}

        for proposal_id in regenerated_ids:
            proposal = proposals_by_id.get(proposal_id)

            if proposal is None:
                raise ValueError(
                    "Regenerated proposal is missing "
                    "from workflow state: "
                    f"{proposal_id}"
                )

            report = verification_service.verify_proposal(
                user_id=state["user_id"],
                proposal=proposal,
            )

            reports_by_id[proposal_id] = report

        target_ids = set(
            state.get(
                "review_target_proposal_ids",
                [],
            )
        )

        failed_ids = sorted(
            proposal_id
            for proposal_id in target_ids
            if (
                proposal_id not in reports_by_id
                or not reports_by_id[proposal_id].fully_supported
            )
        )

        passed_ids = sorted(target_ids - set(failed_ids))

        existing_non_target_blocked = (
            set(
                state.get(
                    "blocked_proposal_ids",
                    [],
                )
            )
            - target_ids
        )

        blocked_ids = sorted(existing_non_target_blocked | set(failed_ids))

        ordered_reports = [
            reports_by_id[proposal.proposal_id]
            for proposal in proposals
            if proposal.proposal_id in reports_by_id
        ]

        return {
            "claim_verification_reports": [
                report.model_dump(mode="json") for report in ordered_reports
            ],
            "reviewable_proposal_ids": (passed_ids),
            "blocked_proposal_ids": (blocked_ids),
            "regeneration_verification_failed_ids": (failed_ids),
            "edit_verification_failed_ids": [],
            "audit_events": [
                {
                    "node": ("verify_regenerated_proposals"),
                    "event": ("regenerated_proposals_verified"),
                }
            ],
        }

    return verify_regenerated_proposals


def create_reverify_human_edits_node(
    verification_service: CVClaimVerificationService,
) -> Callable[[JobAnalysisState], JobAnalysisUpdate]:
    """Reverify every proposal changed by the human."""

    def reverify_human_edits(
        state: JobAnalysisState,
    ) -> JobAnalysisUpdate:
        edited_ids = set(
            state.get(
                "edited_proposal_ids",
                [],
            )
        )

        if not edited_ids:
            raise ValueError(
                "Edited proposal verification requires at least one edited proposal."
            )

        proposals = [
            CVChangeProposal.model_validate(payload)
            for payload in state.get(
                "cv_proposals",
                [],
            )
        ]

        existing_reports = [
            CVClaimVerificationReport.model_validate(payload)
            for payload in state.get(
                "claim_verification_reports",
                [],
            )
        ]

        reports_by_id = {report.proposal_id: report for report in existing_reports}

        proposals_by_id = {proposal.proposal_id: proposal for proposal in proposals}

        for proposal_id in edited_ids:
            proposal = proposals_by_id.get(proposal_id)

            if proposal is None:
                raise ValueError(
                    f"Edited proposal is missing from workflow state: {proposal_id}"
                )

            report = verification_service.verify_proposal(
                user_id=state["user_id"],
                proposal=proposal,
            )

            reports_by_id[proposal_id] = report

        target_ids = set(
            state.get(
                "review_target_proposal_ids",
                [],
            )
        )

        failed_ids = sorted(
            proposal_id
            for proposal_id in target_ids
            if (
                proposal_id not in reports_by_id
                or not reports_by_id[proposal_id].fully_supported
            )
        )

        passed_ids = sorted(target_ids - set(failed_ids))

        existing_non_target_blocked = (
            set(
                state.get(
                    "blocked_proposal_ids",
                    [],
                )
            )
            - target_ids
        )

        blocked_ids = sorted(existing_non_target_blocked | set(failed_ids))

        ordered_reports = [
            reports_by_id[proposal.proposal_id]
            for proposal in proposals
            if proposal.proposal_id in reports_by_id
        ]

        return {
            "claim_verification_reports": [
                report.model_dump(mode="json") for report in ordered_reports
            ],
            "reviewable_proposal_ids": passed_ids,
            "blocked_proposal_ids": blocked_ids,
            "edit_verification_failed_ids": (failed_ids),
            "audit_events": [
                {
                    "node": "reverify_human_edits",
                    "event": "human_edits_reverified",
                }
            ],
        }

    return reverify_human_edits


def request_edit_rework(
    state: JobAnalysisState,
) -> JobAnalysisUpdate:
    """Pause again when revised wording fails verification."""

    target_ids = state.get(
        "review_target_proposal_ids",
        [],
    )
    target_id_set = set(target_ids)

    proposals = [
        CVChangeProposal.model_validate(payload)
        for payload in state.get("cv_proposals", [])
    ]

    reports = [
        CVClaimVerificationReport.model_validate(payload)
        for payload in state.get(
            "claim_verification_reports",
            [],
        )
    ]

    target_proposals = [
        proposal for proposal in proposals if proposal.proposal_id in target_id_set
    ]

    target_reports = [
        report for report in reports if report.proposal_id in target_id_set
    ]

    raw_decision = interrupt(
        {
            "type": "cv_proposal_review",
            "proposals": [
                proposal.model_dump(mode="json") for proposal in target_proposals
            ],
            "verification_reports": [
                report.model_dump(mode="json") for report in target_reports
            ],
            "allowed_actions": [
                ReviewAction.EDIT.value,
                ReviewAction.REGENERATE.value,
                ReviewAction.REJECT.value,
            ],
        }
    )

    decision = CVReviewDecision.model_validate(raw_decision)

    validate_review_decision(
        decision=decision,
        reviewable_proposal_ids=target_ids,
        allowed_actions={
            ReviewAction.EDIT,
            ReviewAction.REGENERATE,
            ReviewAction.REJECT,
        },
    )

    return {
        "review_decision": decision.model_dump(mode="json"),
        "audit_events": [
            {
                "node": "request_edit_rework",
                "event": "human_edit_review_received",
            }
        ],
    }


def finalize_human_review(
    state: JobAnalysisState,
) -> JobAnalysisUpdate:
    """Apply the final human review outcome."""

    decision_payload = state.get("review_decision")

    if decision_payload is None:
        raise ValueError("Human review cannot be finalized without a decision.")

    decision = CVReviewDecision.model_validate(decision_payload)

    proposals = [
        CVChangeProposal.model_validate(payload)
        for payload in state.get("cv_proposals", [])
    ]

    if decision.action is ReviewAction.APPROVE:
        approved_ids = set(decision.approved_proposal_ids)

        final_proposals = [
            proposal for proposal in proposals if proposal.proposal_id in approved_ids
        ]

        review_status = ApprovalStatus.APPROVED

    elif decision.action is ReviewAction.REJECT:
        final_proposals = []
        review_status = ApprovalStatus.REJECTED

    elif decision.action is ReviewAction.EDIT:
        target_ids = set(
            state.get(
                "review_target_proposal_ids",
                [],
            )
        )

        reports = [
            CVClaimVerificationReport.model_validate(payload)
            for payload in state.get(
                "claim_verification_reports",
                [],
            )
        ]

        reports_by_id = {report.proposal_id: report for report in reports}

        unverified_ids = [
            proposal_id
            for proposal_id in target_ids
            if (
                proposal_id not in reports_by_id
                or not reports_by_id[proposal_id].fully_supported
            )
        ]

        if unverified_ids:
            raise ValueError(
                "Human-edited proposals cannot be finalized "
                "until every edit passes claim verification."
            )

        final_proposals = [
            proposal for proposal in proposals if proposal.proposal_id in target_ids
        ]

        review_status = ApprovalStatus.EDITED

    else:
        raise ValueError("Unexpected review action reached finalization.")

    return {
        "review_status": review_status.value,
        "final_cv_proposals": [
            proposal.model_dump(mode="json") for proposal in final_proposals
        ],
        "audit_events": [
            {
                "node": "finalize_human_review",
                "event": "human_review_finalized",
            }
        ],
    }


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
