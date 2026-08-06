"""Deterministic validation for evidence-discovery agent results."""

from collections.abc import Collection

from careerops_agent_engine.application.exceptions import (
    EvidenceDiscoveryValidationError,
)
from careerops_agent_engine.application.ports.evidence_repository import (
    EvidenceRepository,
)
from careerops_agent_engine.domain.models.evidence import EvidenceMatch
from careerops_agent_engine.domain.models.job import JobRequirement

REQUIRED_SEARCH_TOOL = "search_approved_evidence"


def validate_evidence_discovery_result(
    *,
    requirement: JobRequirement,
    match: EvidenceMatch,
    called_tools: Collection[str],
    observed_evidence_ids: Collection[str],
    repository: EvidenceRepository,
    user_id: str,
) -> None:
    """Validate provenance, trajectory and user isolation.

    The language model proposes the match, but deterministic application
    logic verifies that the referenced evidence was genuinely retrieved
    and belongs to the authenticated user.
    """

    if match.requirement_id != requirement.requirement_id:
        raise EvidenceDiscoveryValidationError(
            "Evidence match requirement identifier does not match "
            "the requested requirement."
        )

    if REQUIRED_SEARCH_TOOL not in called_tools:
        raise EvidenceDiscoveryValidationError(
            "Evidence discovery must search approved evidence before returning a match."
        )

    cited_ids = {
        *match.direct_evidence_ids,
        *match.related_evidence_ids,
    }
    observed_ids = set(observed_evidence_ids)

    unobserved_ids = cited_ids - observed_ids

    if unobserved_ids:
        unobserved_display = ", ".join(sorted(unobserved_ids))
        raise EvidenceDiscoveryValidationError(
            "The agent cited evidence that was not returned by its "
            f"tools: {unobserved_display}"
        )

    inaccessible_ids = {
        evidence_id
        for evidence_id in cited_ids
        if repository.get_approved(
            user_id=user_id,
            evidence_id=evidence_id,
        )
        is None
    }

    if inaccessible_ids:
        inaccessible_display = ", ".join(sorted(inaccessible_ids))
        raise EvidenceDiscoveryValidationError(
            "The agent cited evidence that is not approved or does "
            f"not belong to the authenticated user: {inaccessible_display}"
        )
