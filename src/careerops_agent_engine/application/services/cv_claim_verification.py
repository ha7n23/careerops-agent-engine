"""Application service for grounded CV claim verification."""

from collections.abc import Sequence

from careerops_agent_engine.application.exceptions import (
    CVClaimVerificationValidationError,
)
from careerops_agent_engine.application.ports.cv_claim_verifier import (
    CVClaimVerificationRequest,
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

    def verify_proposals(
        self,
        *,
        user_id: str,
        proposals: Sequence[CVChangeProposal],
    ) -> list[CVClaimVerificationReport]:
        """Verify all initial proposals through one batch call."""

        proposal_list = list(proposals)
        proposal_ids = [proposal.proposal_id for proposal in proposal_list]

        if len(proposal_ids) != len(set(proposal_ids)):
            raise CVClaimVerificationValidationError(
                "Batch claim verification requires unique proposal identifiers."
            )

        verification_requests: list[CVClaimVerificationRequest] = []
        allowed_evidence_ids_by_proposal: dict[
            str,
            set[str],
        ] = {}

        for proposal in proposal_list:
            approved_evidence = self._load_supporting_evidence(
                user_id=user_id,
                proposal=proposal,
            )

            verification_requests.append(
                CVClaimVerificationRequest(
                    proposal=proposal,
                    approved_evidence=tuple(approved_evidence),
                )
            )
            allowed_evidence_ids_by_proposal[proposal.proposal_id] = {
                evidence.evidence_id for evidence in approved_evidence
            }

        if not verification_requests:
            return []

        generated_reports = self._verifier.verify_batch(
            requests=verification_requests,
        )

        reports_by_proposal: dict[
            str,
            CVClaimVerificationReport,
        ] = {}

        for report in generated_reports:
            if report.proposal_id in reports_by_proposal:
                raise CVClaimVerificationValidationError(
                    "Batch claim verification returned duplicate proposal identifiers."
                )

            reports_by_proposal[report.proposal_id] = report

        expected_proposal_ids = set(proposal_ids)

        if set(reports_by_proposal) != expected_proposal_ids:
            raise CVClaimVerificationValidationError(
                "Batch claim verification must return exactly one report per proposal."
            )

        validated_reports: list[CVClaimVerificationReport] = []

        for proposal in proposal_list:
            report = reports_by_proposal[proposal.proposal_id]

            validate_claim_verification_report(
                proposal=proposal,
                report=report,
                allowed_evidence_ids=(
                    allowed_evidence_ids_by_proposal[proposal.proposal_id]
                ),
            )
            validated_reports.append(report)

        return validated_reports

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
