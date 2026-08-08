"""Human approval boundary for extracted CV evidence."""

from hashlib import sha256

from careerops_agent_engine.application.exceptions import (
    CVEvidenceProposalValidationError,
    CVEvidenceReviewValidationError,
)
from careerops_agent_engine.application.services.cv_evidence_proposals import (
    validate_claim_grounding,
    validate_string_list,
    validate_technology_grounding,
)
from careerops_agent_engine.domain.enums import (
    VerificationStatus,
)
from careerops_agent_engine.domain.models.evidence import (
    CareerEvidence,
    CareerEvidenceCandidate,
    CareerEvidenceOverlapFinding,
    CareerEvidenceProposal,
)
from careerops_agent_engine.domain.models.evidence_review import (
    EvidenceProposalEdit,
    EvidenceReviewDecision,
    EvidenceReviewResult,
)


class CVEvidenceReviewService:
    """Convert explicitly accepted proposals into approved evidence."""

    def review(
        self,
        *,
        proposals: list[CareerEvidenceProposal],
        overlap_findings: list[CareerEvidenceOverlapFinding],
        decision: EvidenceReviewDecision,
    ) -> EvidenceReviewResult:
        """Validate one complete human decision and apply it."""

        proposals_by_id = build_proposal_index(proposals)

        validate_review_coverage(
            proposals_by_id=proposals_by_id,
            decision=decision,
        )

        validate_overlap_acknowledgements(
            proposals_by_id=proposals_by_id,
            overlap_findings=overlap_findings,
            decision=decision,
        )

        approved_evidence: list[CareerEvidence] = []

        for proposal_id in decision.approved_proposal_ids:
            proposal = proposals_by_id[proposal_id]

            approved_evidence.append(
                approve_proposal(
                    proposal=proposal,
                    edit=None,
                )
            )

        for edit in decision.edits:
            proposal = proposals_by_id[edit.proposal_id]

            approved_evidence.append(
                approve_proposal(
                    proposal=proposal,
                    edit=edit,
                )
            )

        return EvidenceReviewResult(
            approved_proposal_ids=list(decision.approved_proposal_ids),
            edited_proposal_ids=[edit.proposal_id for edit in decision.edits],
            rejected_proposal_ids=list(decision.rejected_proposal_ids),
            acknowledged_overlap_proposal_ids=list(
                decision.acknowledged_overlap_proposal_ids
            ),
            approved_evidence=approved_evidence,
        )


def build_proposal_index(
    proposals: list[CareerEvidenceProposal],
) -> dict[str, CareerEvidenceProposal]:
    """Build a unique review lookup."""

    indexed = {proposal.proposal_id: proposal for proposal in proposals}

    if len(indexed) != len(proposals):
        raise CVEvidenceReviewValidationError(
            "Evidence review contains duplicate proposal identifiers."
        )

    if not indexed:
        raise CVEvidenceReviewValidationError(
            "Evidence review requires at least one pending proposal."
        )

    return indexed


def validate_review_coverage(
    *,
    proposals_by_id: dict[
        str,
        CareerEvidenceProposal,
    ],
    decision: EvidenceReviewDecision,
) -> None:
    """Require every proposal to receive exactly one decision."""

    expected_ids = set(proposals_by_id)

    approved_ids = set(decision.approved_proposal_ids)

    rejected_ids = set(decision.rejected_proposal_ids)

    edited_ids = {edit.proposal_id for edit in decision.edits}

    referenced_ids = approved_ids | rejected_ids | edited_ids

    unknown_ids = referenced_ids - expected_ids

    if unknown_ids:
        unknown_display = ", ".join(sorted(unknown_ids))

        raise CVEvidenceReviewValidationError(
            "Evidence review references proposals "
            "outside the current review set: "
            f"{unknown_display}"
        )

    missing_ids = expected_ids - referenced_ids

    if missing_ids:
        missing_display = ", ".join(sorted(missing_ids))

        raise CVEvidenceReviewValidationError(
            "Every evidence proposal must receive "
            "a human decision. Missing: "
            f"{missing_display}"
        )


def validate_overlap_acknowledgements(
    *,
    proposals_by_id: dict[
        str,
        CareerEvidenceProposal,
    ],
    overlap_findings: list[CareerEvidenceOverlapFinding],
    decision: EvidenceReviewDecision,
) -> None:
    """Require explicit acknowledgement before accepting overlaps."""

    expected_ids = set(proposals_by_id)

    finding_proposal_ids = {finding.proposal_id for finding in overlap_findings}

    unknown_findings = finding_proposal_ids - expected_ids

    if unknown_findings:
        raise CVEvidenceReviewValidationError(
            "Evidence overlap findings reference "
            "proposals outside the current review set."
        )

    acknowledged_ids = set(decision.acknowledged_overlap_proposal_ids)

    unknown_acknowledgements = acknowledged_ids - finding_proposal_ids

    if unknown_acknowledgements:
        raise CVEvidenceReviewValidationError(
            "Overlap acknowledgement references a proposal without an overlap finding."
        )

    accepted_ids = {
        *decision.approved_proposal_ids,
        *(edit.proposal_id for edit in decision.edits),
    }

    required_acknowledgements = finding_proposal_ids & accepted_ids

    missing_acknowledgements = required_acknowledgements - acknowledged_ids

    if missing_acknowledgements:
        missing_display = ", ".join(sorted(missing_acknowledgements))

        raise CVEvidenceReviewValidationError(
            "Accepting overlapping evidence requires "
            "explicit human acknowledgement: "
            f"{missing_display}"
        )


def approve_proposal(
    *,
    proposal: CareerEvidenceProposal,
    edit: EvidenceProposalEdit | None,
) -> CareerEvidence:
    """Apply optional human edits and cross the approval boundary."""

    title = (
        edit.title if edit is not None and edit.title is not None else proposal.title
    )

    technologies = (
        list(edit.technologies)
        if edit is not None and edit.technologies is not None
        else list(proposal.technologies)
    )

    capabilities = (
        list(edit.capabilities)
        if edit is not None and edit.capabilities is not None
        else list(proposal.capabilities)
    )

    claims = (
        list(edit.claims)
        if edit is not None and edit.claims is not None
        else list(proposal.claims)
    )

    validate_accepted_grounding(
        proposal=proposal,
        title=title,
        technologies=technologies,
        capabilities=capabilities,
        claims=claims,
    )

    return CareerEvidence(
        evidence_id=build_approved_evidence_id(proposal.proposal_id),
        category=proposal.category,
        title=title,
        verification_status=(VerificationStatus.APPROVED),
        technologies=technologies,
        capabilities=capabilities,
        approved_claims=claims,
        source_references=list(proposal.source_references),
    )


def validate_accepted_grounding(
    *,
    proposal: CareerEvidenceProposal,
    title: str,
    technologies: list[str],
    capabilities: list[str],
    claims: list[str],
) -> None:
    """Revalidate accepted factual content before approval."""

    excerpts = [
        reference.source_excerpt
        for reference in proposal.source_references
        if reference.source_excerpt
    ]

    if len(excerpts) != 1:
        raise CVEvidenceReviewValidationError(
            "CV evidence approval requires exactly one source excerpt."
        )

    candidate = CareerEvidenceCandidate(
        category=proposal.category,
        title=title,
        source_section_order_index=(proposal.source_section_order_index),
        source_excerpt=excerpts[0],
        technologies=technologies,
        capabilities=capabilities,
        claims=claims,
        warnings=[],
    )

    try:
        validate_string_list(
            values=candidate.technologies,
            field_name="technologies",
        )

        validate_string_list(
            values=candidate.capabilities,
            field_name="capabilities",
        )

        validate_string_list(
            values=candidate.claims,
            field_name="claims",
        )

        validate_claim_grounding(candidate=candidate)

        validate_technology_grounding(candidate=candidate)

    except CVEvidenceProposalValidationError as exc:
        raise CVEvidenceReviewValidationError(str(exc)) from exc


def build_approved_evidence_id(
    proposal_id: str,
) -> str:
    """Build a stable approved-evidence identifier."""

    digest = sha256(proposal_id.encode("utf-8")).hexdigest()[:16].upper()

    return f"EVD-{digest}"
