"""Application service for grounded CV proposal generation."""

from hashlib import sha256

from careerops_agent_engine.application.exceptions import (
    CVProposalValidationError,
)
from careerops_agent_engine.application.ports.cv_proposal_generator import (
    CVProposalGenerator,
)
from careerops_agent_engine.application.ports.evidence_repository import (
    EvidenceRepository,
)
from careerops_agent_engine.domain.enums import MatchStrength
from careerops_agent_engine.domain.models.cv import (
    CVChangeProposal,
)
from careerops_agent_engine.domain.models.evidence import (
    CareerEvidence,
    EvidenceMatch,
)
from careerops_agent_engine.domain.models.job import JobRequirement

ELIGIBLE_MATCH_STRENGTHS = {
    MatchStrength.STRONG,
    MatchStrength.PARTIAL,
}


class CVProposalGenerationService:
    """Generate proposals only from validated direct evidence."""

    def __init__(
        self,
        *,
        repository: EvidenceRepository,
        generator: CVProposalGenerator,
    ) -> None:
        """Store the trusted repository and generation adapter."""

        self._repository = repository
        self._generator = generator

    def generate_for_requirement(
        self,
        *,
        job_id: str,
        user_id: str,
        requirement: JobRequirement,
        evidence_match: EvidenceMatch,
    ) -> CVChangeProposal | None:
        """Generate one grounded proposal when direct evidence exists."""

        self._validate_requirement_match(
            requirement=requirement,
            evidence_match=evidence_match,
        )

        if evidence_match.match_strength not in ELIGIBLE_MATCH_STRENGTHS:
            return None

        direct_evidence = self._load_direct_evidence(
            user_id=user_id,
            evidence_match=evidence_match,
        )

        proposal_id = build_proposal_id(
            job_id=job_id,
            requirement_id=requirement.requirement_id,
        )

        proposal = self._generator.generate(
            proposal_id=proposal_id,
            job_id=job_id,
            requirement=requirement,
            evidence_match=evidence_match,
            approved_evidence=direct_evidence,
        )

        validate_cv_proposal(
            proposal=proposal,
            expected_proposal_id=proposal_id,
            requirement=requirement,
            allowed_evidence_ids={evidence.evidence_id for evidence in direct_evidence},
        )

        return proposal

    def regenerate_for_requirement(
        self,
        *,
        job_id: str,
        user_id: str,
        requirement: JobRequirement,
        evidence_match: EvidenceMatch,
        previous_proposal: CVChangeProposal,
        reviewer_feedback: str,
    ) -> CVChangeProposal:
        """Regenerate one proposal from evidence and human feedback."""

        self._validate_requirement_match(
            requirement=requirement,
            evidence_match=evidence_match,
        )

        if evidence_match.match_strength not in ELIGIBLE_MATCH_STRENGTHS:
            raise CVProposalValidationError(
                "Regeneration requires direct eligible evidence."
            )

        feedback = reviewer_feedback.strip()

        if not feedback:
            raise CVProposalValidationError("Regeneration requires reviewer feedback.")

        direct_evidence = self._load_direct_evidence(
            user_id=user_id,
            evidence_match=evidence_match,
        )

        proposal_id = build_proposal_id(
            job_id=job_id,
            requirement_id=requirement.requirement_id,
        )

        allowed_evidence_ids = {evidence.evidence_id for evidence in direct_evidence}

        validate_cv_proposal(
            proposal=previous_proposal,
            expected_proposal_id=proposal_id,
            requirement=requirement,
            allowed_evidence_ids=allowed_evidence_ids,
        )

        regenerated = self._generator.regenerate(
            proposal_id=proposal_id,
            job_id=job_id,
            requirement=requirement,
            evidence_match=evidence_match,
            approved_evidence=direct_evidence,
            previous_proposal=previous_proposal,
            reviewer_feedback=feedback,
        )

        validate_cv_proposal(
            proposal=regenerated,
            expected_proposal_id=proposal_id,
            requirement=requirement,
            allowed_evidence_ids=allowed_evidence_ids,
        )

        if regenerated.proposed_text == previous_proposal.proposed_text:
            raise CVProposalValidationError(
                "Regenerated proposal must meaningfully revise "
                "the previous proposal text."
            )

        return regenerated

    def _validate_requirement_match(
        self,
        *,
        requirement: JobRequirement,
        evidence_match: EvidenceMatch,
    ) -> None:
        """Ensure the evidence match belongs to the requirement."""

        if evidence_match.requirement_id != requirement.requirement_id:
            raise CVProposalValidationError(
                "Evidence match requirement identifier does not "
                "match the requested requirement."
            )

    def _load_direct_evidence(
        self,
        *,
        user_id: str,
        evidence_match: EvidenceMatch,
    ) -> list[CareerEvidence]:
        """Reload direct evidence through the trusted user boundary."""

        evidence: list[CareerEvidence] = []

        for evidence_id in evidence_match.direct_evidence_ids:
            record = self._repository.get_approved(
                user_id=user_id,
                evidence_id=evidence_id,
            )

            if record is None:
                raise CVProposalValidationError(
                    "Direct evidence is no longer approved or "
                    "accessible to the authenticated user: "
                    f"{evidence_id}"
                )

            evidence.append(record)

        if not evidence:
            raise CVProposalValidationError(
                "An eligible CV proposal requires direct evidence."
            )

        return evidence


def build_proposal_id(
    *,
    job_id: str,
    requirement_id: str,
) -> str:
    """Build a stable proposal identifier for retry-safe execution."""

    digest = sha256(f"{job_id}:{requirement_id}".encode()).hexdigest()[:16].upper()

    return f"CVP-{digest}"


def validate_cv_proposal(
    *,
    proposal: CVChangeProposal,
    expected_proposal_id: str,
    requirement: JobRequirement,
    allowed_evidence_ids: set[str],
) -> None:
    """Apply deterministic provenance checks to generated output."""

    if proposal.proposal_id != expected_proposal_id:
        raise CVProposalValidationError("Generated proposal identifier was modified.")

    if proposal.requirement_ids != [requirement.requirement_id]:
        raise CVProposalValidationError(
            "Generated proposal references unexpected requirements."
        )

    cited_evidence_ids = set(proposal.supporting_evidence_ids)

    unsupported_ids = cited_evidence_ids - allowed_evidence_ids

    if unsupported_ids:
        unsupported_display = ", ".join(sorted(unsupported_ids))

        raise CVProposalValidationError(
            "Generated proposal cited evidence outside the "
            "validated direct-evidence set: "
            f"{unsupported_display}"
        )

    if proposal.target_entry_id is not None:
        raise CVProposalValidationError(
            "Proposal cannot target a CV entry before CV "
            "ingestion has established that entry."
        )

    if proposal.current_text is not None:
        raise CVProposalValidationError(
            "Proposal cannot claim existing CV text before CV "
            "ingestion has established that text."
        )
