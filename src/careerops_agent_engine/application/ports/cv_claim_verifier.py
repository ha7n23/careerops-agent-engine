"""Application port for CV claim verification."""

from collections.abc import Sequence
from typing import Protocol

from careerops_agent_engine.domain.models.cv import (
    CVChangeProposal,
)
from careerops_agent_engine.domain.models.evidence import (
    CareerEvidence,
)
from careerops_agent_engine.domain.models.verification import (
    CVClaimVerificationReport,
)


class CVClaimVerifier(Protocol):
    """Verify factual CV claims against approved evidence."""

    def verify(
        self,
        *,
        proposal: CVChangeProposal,
        approved_evidence: Sequence[CareerEvidence],
    ) -> CVClaimVerificationReport:
        """Return a claim-level verification report."""

        ...
