"""Deterministic validation for evidence-discovery agent results."""

from collections.abc import Collection, Sequence

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


def validate_evidence_discovery_batch(
    *,
    requirements: Sequence[JobRequirement],
    matches: Sequence[EvidenceMatch],
    called_tools: Collection[str],
    observed_evidence_ids: Collection[str],
    repository: EvidenceRepository,
    user_id: str,
) -> list[EvidenceMatch]:
    """Validate complete batch coverage and restore request order."""

    requirement_list = list(requirements)
    match_list = list(matches)

    requirement_ids = [requirement.requirement_id for requirement in requirement_list]

    if len(requirement_ids) != len(set(requirement_ids)):
        raise EvidenceDiscoveryValidationError(
            "Evidence discovery received duplicate requirement identifiers."
        )

    matches_by_requirement_id: dict[str, EvidenceMatch] = {}

    for match in match_list:
        if match.requirement_id in matches_by_requirement_id:
            raise EvidenceDiscoveryValidationError(
                "Evidence discovery returned duplicate matches "
                f"for requirement: {match.requirement_id}"
            )

        matches_by_requirement_id[match.requirement_id] = match

    expected_ids = set(requirement_ids)
    returned_ids = set(matches_by_requirement_id)

    unknown_ids = returned_ids - expected_ids

    if unknown_ids:
        unknown_display = ", ".join(sorted(unknown_ids))

        raise EvidenceDiscoveryValidationError(
            "Evidence discovery returned matches outside "
            f"the requested requirement set: {unknown_display}"
        )

    missing_ids = expected_ids - returned_ids

    if missing_ids:
        missing_display = ", ".join(sorted(missing_ids))

        raise EvidenceDiscoveryValidationError(
            "Evidence discovery did not return matches "
            f"for every requirement. Missing: {missing_display}"
        )

    ordered_matches: list[EvidenceMatch] = []

    for requirement in requirement_list:
        match = matches_by_requirement_id[requirement.requirement_id]

        validate_evidence_discovery_result(
            requirement=requirement,
            match=match,
            called_tools=called_tools,
            observed_evidence_ids=observed_evidence_ids,
            repository=repository,
            user_id=user_id,
        )

        ordered_matches.append(match)

    return ordered_matches
