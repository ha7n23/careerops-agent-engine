"""Deterministic job-fit scoring services."""

from collections.abc import Collection, Sequence

from careerops_agent_engine.domain.models.job import JobRequirement


def calculate_weighted_fit(
    requirements: Sequence[JobRequirement],
    matched_requirement_ids: Collection[str],
) -> float:
    """Calculate the percentage of requirement weight directly matched.

    Args:
        requirements: Validated job requirements with importance scores.
        matched_requirement_ids: Requirements supported by direct evidence.

    Returns:
        A percentage rounded to two decimal places.

    Raises:
        ValueError: If requirements are empty or an unknown identifier is used.
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
