"""Deterministic job-fit scoring services."""

from collections.abc import Collection, Sequence

from careerops_agent_engine.domain.enums import MatchStrength
from careerops_agent_engine.domain.models.evidence import EvidenceMatch
from careerops_agent_engine.domain.models.job import JobRequirement

MATCH_FACTORS: dict[MatchStrength, float] = {
    MatchStrength.STRONG: 1.0,
    MatchStrength.PARTIAL: 0.5,
    MatchStrength.RELATED: 0.0,
    MatchStrength.NONE: 0.0,
}


def calculate_weighted_fit(
    requirements: Sequence[JobRequirement],
    matched_requirement_ids: Collection[str],
) -> float:
    """Calculate the percentage of requirement weight directly matched.

    This function is retained for the earlier deterministic workflow
    tests. The evidence-aware graph uses
    ``calculate_evidence_weighted_fit``.
    """

    if not requirements:
        raise ValueError("At least one requirement is required.")

    known_ids = {requirement.requirement_id for requirement in requirements}
    matched_ids = set(matched_requirement_ids)
    unknown_ids = matched_ids - known_ids

    if unknown_ids:
        unknown_display = ", ".join(sorted(unknown_ids))
        raise ValueError(f"Unknown matched requirement identifiers: {unknown_display}")

    total_weight = sum(requirement.importance_score for requirement in requirements)
    matched_weight = sum(
        requirement.importance_score
        for requirement in requirements
        if requirement.requirement_id in matched_ids
    )

    return round((matched_weight / total_weight) * 100, 2)


def calculate_evidence_weighted_fit(
    requirements: Sequence[JobRequirement],
    evidence_matches: Sequence[EvidenceMatch],
) -> float:
    """Calculate fit from validated evidence-match classifications.

    Scoring policy:
    - Strong direct match: 100 percent of the requirement weight.
    - Partial direct match: 50 percent of the requirement weight.
    - Related evidence: no direct-fit credit.
    - No evidence: no credit.

    Related evidence remains useful in the gap report but must not
    silently become direct experience.
    """

    if not requirements:
        raise ValueError("At least one requirement is required.")

    requirement_ids = [requirement.requirement_id for requirement in requirements]
    match_ids = [match.requirement_id for match in evidence_matches]

    if len(match_ids) != len(set(match_ids)):
        raise ValueError(
            "Evidence matches must contain unique requirement identifiers."
        )

    required_id_set = set(requirement_ids)
    match_id_set = set(match_ids)

    missing_ids = required_id_set - match_id_set
    unknown_ids = match_id_set - required_id_set

    if missing_ids:
        missing_display = ", ".join(sorted(missing_ids))
        raise ValueError(
            f"Evidence matches are missing requirements: {missing_display}"
        )

    if unknown_ids:
        unknown_display = ", ".join(sorted(unknown_ids))
        raise ValueError(
            f"Evidence matches contain unknown requirements: {unknown_display}"
        )

    matches_by_requirement = {match.requirement_id: match for match in evidence_matches}

    total_weight = sum(requirement.importance_score for requirement in requirements)

    earned_weight = sum(
        requirement.importance_score
        * MATCH_FACTORS[
            matches_by_requirement[requirement.requirement_id].match_strength
        ]
        for requirement in requirements
    )

    return round((earned_weight / total_weight) * 100, 2)
