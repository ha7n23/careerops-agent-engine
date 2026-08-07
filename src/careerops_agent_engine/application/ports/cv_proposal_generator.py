"""Application port for evidence-grounded CV proposal generation."""

from collections.abc import Sequence
from typing import Protocol

from careerops_agent_engine.domain.models.cv import (
    CVChangeProposal,
)
from careerops_agent_engine.domain.models.evidence import (
    CareerEvidence,
    EvidenceMatch,
)
from careerops_agent_engine.domain.models.job import JobRequirement


class CVProposalGenerator(Protocol):
    """Generate one CV proposal from validated direct evidence."""

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
