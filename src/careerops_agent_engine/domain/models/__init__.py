"""Public CareerOps domain models."""

from careerops_agent_engine.domain.models.approval import (
    CVReviewDecision,
    CVReviewRecord,
    ProposalEdit,
)
from careerops_agent_engine.domain.models.cv import CVChangeProposal
from careerops_agent_engine.domain.models.evidence import (
    CareerEvidence,
    EvidenceMatch,
    SourceReference,
)
from careerops_agent_engine.domain.models.job import (
    JobRequirement,
    JobRequirementExtraction,
)

__all__ = [
    "CVChangeProposal",
    "CVReviewDecision",
    "CVReviewRecord",
    "CareerEvidence",
    "EvidenceMatch",
    "JobRequirement",
    "JobRequirementExtraction",
    "ProposalEdit",
    "SourceReference",
]
