"""Domain models for CV claim verification."""

from typing import Self

from pydantic import Field, model_validator

from careerops_agent_engine.domain.models.base import DomainModel


class ClaimAssessment(DomainModel):
    """Verification outcome for one factual CV claim."""

    claim_text: str = Field(
        min_length=1,
        max_length=1_500,
    )
    supported: bool

    supporting_evidence_ids: list[str] = Field(default_factory=list)

    explanation: str = Field(
        min_length=1,
        max_length=1_500,
    )

    @model_validator(mode="after")
    def validate_claim_assessment(self) -> Self:
        """Reject internally inconsistent claim assessments."""

        if len(self.supporting_evidence_ids) != len(set(self.supporting_evidence_ids)):
            raise ValueError("Supporting evidence identifiers must be unique.")

        if self.supported and not self.supporting_evidence_ids:
            raise ValueError("A supported claim requires supporting evidence.")

        return self


class CVClaimVerificationReport(DomainModel):
    """Verification report for one generated CV proposal."""

    proposal_id: str = Field(
        min_length=1,
        max_length=64,
    )

    claims: list[ClaimAssessment] = Field(min_length=1)

    coverage_complete: bool
    coverage_notes: list[str] = Field(default_factory=list)

    fully_supported: bool

    unsupported_claims: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_report_consistency(self) -> Self:
        """Ensure summary fields agree with claim assessments."""

        claim_texts = [claim.claim_text for claim in self.claims]

        if len(claim_texts) != len(set(claim_texts)):
            raise ValueError("Claim assessments must contain unique claim text.")

        expected_unsupported = [
            claim.claim_text for claim in self.claims if not claim.supported
        ]

        if self.unsupported_claims != expected_unsupported:
            raise ValueError("Unsupported claims must match the claim assessments.")

        expected_fully_supported = self.coverage_complete and not expected_unsupported

        if self.fully_supported != expected_fully_supported:
            raise ValueError(
                "Fully-supported status is inconsistent with "
                "claim assessments or coverage."
            )

        if not self.coverage_complete and not self.coverage_notes:
            raise ValueError("Incomplete claim coverage requires a coverage note.")

        return self
