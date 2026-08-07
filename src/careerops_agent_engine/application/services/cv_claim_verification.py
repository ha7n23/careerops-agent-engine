"""Application service for grounded CV claim verification."""

from careerops_agent_engine.application.exceptions import (
    CVClaimVerificationValidationError,
)
from careerops_agent_engine.application.ports.cv_claim_verifier import (
    CVClaimVerifier,
)
from careerops_agent_engine.application.ports.evidence_repository import (
    EvidenceRepository,
)
from careerops_agent_engine.domain.models.cv import (
    CVChangeProposal,
)
from careerops_agent_engine.domain.models.evidence import (
    CareerEvidence,
)
from careerops_agent_engine.domain.models.verification import (
    CVClaimVerificationReport,
)


class CVClaimVerificationService:
    """Verify proposals using only approved user evidence."""

    def __init__(
        self,
        *,
        repository: EvidenceRepository,
        verifier: CVClaimVerifier,
    ) -> None:
        """Store trusted persistence and verification adapters."""

        self._repository = repository
        self._verifier = verifier

    def verify_proposal(
        self,
        *,
        user_id: str,
        proposal: CVChangeProposal,
    ) -> CVClaimVerificationReport:
        """Verify one proposal inside the authenticated boundary."""

        approved_evidence = self._load_supporting_evidence(
            user_id=user_id,
            proposal=proposal,
        )

        report = self._verifier.verify(
            proposal=proposal,
            approved_evidence=approved_evidence,
        )

        validate_claim_verification_report(
            proposal=proposal,
            report=report,
            allowed_evidence_ids={
                evidence.evidence_id for evidence in approved_evidence
            },
        )

        return report

    def _load_supporting_evidence(
        self,
        *,
        user_id: str,
        proposal: CVChangeProposal,
    ) -> list[CareerEvidence]:
        """Reload proposal evidence through the trusted repository."""

        evidence_records: list[CareerEvidence] = []

        for evidence_id in proposal.supporting_evidence_ids:
            evidence = self._repository.get_approved(
                user_id=user_id,
                evidence_id=evidence_id,
            )

            if evidence is None:
                raise CVClaimVerificationValidationError(
                    "Proposal evidence is no longer approved or "
                    "accessible to the authenticated user: "
                    f"{evidence_id}"
                )

            evidence_records.append(evidence)

        return evidence_records


def validate_claim_verification_report(
    *,
    proposal: CVChangeProposal,
    report: CVClaimVerificationReport,
    allowed_evidence_ids: set[str],
) -> None:
    """Apply deterministic provenance checks to verifier output."""

    if report.proposal_id != proposal.proposal_id:
        raise CVClaimVerificationValidationError(
            "Verification report references the wrong proposal."
        )

    for claim in report.claims:
        cited_ids = set(claim.supporting_evidence_ids)

        unexpected_ids = cited_ids - allowed_evidence_ids

        if unexpected_ids:
            unexpected_display = ", ".join(sorted(unexpected_ids))

            raise CVClaimVerificationValidationError(
                "Claim verification cited evidence outside the "
                "proposal's approved evidence set: "
                f"{unexpected_display}"
            )
