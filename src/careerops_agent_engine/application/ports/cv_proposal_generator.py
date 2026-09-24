"""Application port for evidence-grounded CV proposal generation."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from careerops_agent_engine.domain.models.cv import (
    CVChangeProposal,
)
from careerops_agent_engine.domain.models.evidence import (
    CareerEvidence,
    EvidenceMatch,
)
from careerops_agent_engine.domain.models.job import JobRequirement


@dataclass(frozen=True, slots=True)
class CVProposalGenerationRequest:
    """One validated requirement and its approved generation context."""

    proposal_id: str
    requirement: JobRequirement
    evidence_match: EvidenceMatch
    approved_evidence: tuple[CareerEvidence, ...]


class CVProposalGenerator(Protocol):
    """Generate and regenerate evidence-grounded CV proposals."""

    def generate(
        self,
        *,
        proposal_id: str,
        job_id: str,
        requirement: JobRequirement,
        evidence_match: EvidenceMatch,
        approved_evidence: Sequence[CareerEvidence],
    ) -> CVChangeProposal:
        """Return one evidence-grounded CV change proposal."""

        ...

    def generate_batch(
        self,
        *,
        job_id: str,
        requests: Sequence[CVProposalGenerationRequest],
    ) -> list[CVChangeProposal]:
        """Generate multiple proposals through one provider invocation."""

        ...

    def regenerate(
        self,
        *,
        proposal_id: str,
        job_id: str,
        requirement: JobRequirement,
        evidence_match: EvidenceMatch,
        approved_evidence: Sequence[CareerEvidence],
        previous_proposal: CVChangeProposal,
        reviewer_feedback: str,
    ) -> CVChangeProposal:
        """Return a revised proposal using human feedback."""

        ...
