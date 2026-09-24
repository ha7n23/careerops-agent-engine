"""Application port for CV claim verification."""

from collections.abc import Sequence
from dataclasses import dataclass
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


@dataclass(frozen=True, slots=True)
class CVClaimVerificationRequest:
    """One proposal and its approved verification context."""

    proposal: CVChangeProposal
    approved_evidence: tuple[CareerEvidence, ...]


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

    def verify_batch(
        self,
        *,
        requests: Sequence[CVClaimVerificationRequest],
    ) -> list[CVClaimVerificationReport]:
        """Verify multiple proposals through one provider invocation."""

        ...
