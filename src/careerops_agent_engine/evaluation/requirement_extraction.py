"""Deterministic evaluation of requirement-extraction outputs."""

import re

from careerops_agent_engine.evaluation.models import (
    ExpectedRequirement,
    RequirementExpectationEvaluation,
    RequirementExtractionEvalCase,
    RequirementExtractionEvaluation,
)
from careerops_agent_engine.infrastructure.llm.schemas import (
    ExtractedRequirement,
    ExtractedRequirementSet,
)


def evaluate_requirement_extraction(
    *,
    case: RequirementExtractionEvalCase,
    actual: ExtractedRequirementSet,
) -> RequirementExtractionEvaluation:
    """Evaluate one structured extraction against its gold case."""

    role_title_correct = _role_title_matches(
        expected=case.expected_role_title,
        actual=actual.role_title,
    )

    requirement_count_correct = (
        len(actual.requirements) == case.expected_requirement_count
    )

    forbidden_terms_absent = _forbidden_terms_are_absent(
        case=case,
        actual=actual,
    )

    source_grounding_correct = _all_source_text_is_grounded(
        job_description=(case.job_description),
        requirements=(actual.requirements),
    )

    used_requirement_indexes: set[int] = set()

    expectation_results: list[RequirementExpectationEvaluation] = []

    for expectation in case.expected_requirements:
        result = _evaluate_expectation(
            expectation=expectation,
            requirements=(actual.requirements),
            used_requirement_indexes=(used_requirement_indexes),
        )

        expectation_results.append(result)

    fully_correct_expectations = sum(
        1
        for result in expectation_results
        if (
            result.matched
            and result.category_correct
            and result.match_terms_present
            and result.source_terms_present
        )
    )

    expectation_accuracy = fully_correct_expectations / len(expectation_results)

    score = (
        float(role_title_correct)
        + float(requirement_count_correct)
        + float(forbidden_terms_absent)
        + float(source_grounding_correct)
        + expectation_accuracy
    ) / 5.0

    passed = (
        role_title_correct
        and requirement_count_correct
        and forbidden_terms_absent
        and source_grounding_correct
        and (fully_correct_expectations == len(expectation_results))
    )

    return RequirementExtractionEvaluation(
        case_id=case.case_id,
        role_title_correct=(role_title_correct),
        requirement_count_correct=(requirement_count_correct),
        forbidden_terms_absent=(forbidden_terms_absent),
        source_grounding_correct=(source_grounding_correct),
        expectation_results=(expectation_results),
        score=score,
        passed=passed,
    )


def _evaluate_expectation(
    *,
    expectation: ExpectedRequirement,
    requirements: list[ExtractedRequirement],
    used_requirement_indexes: set[int],
) -> RequirementExpectationEvaluation:
    """Match and score one expected requirement."""

    matched = _find_requirement_match(
        expectation=expectation,
        requirements=requirements,
        used_requirement_indexes=(used_requirement_indexes),
    )

    if matched is None:
        return RequirementExpectationEvaluation(
            expectation_id=(expectation.expectation_id),
            matched=False,
            actual_requirement_name=None,
            category_correct=False,
            match_terms_present=False,
            source_terms_present=False,
        )

    requirement_index, requirement = matched

    used_requirement_indexes.add(requirement_index)

    source_text = _normalize_text(requirement.source_text)

    source_terms_present = all(
        _normalize_text(term) in source_text for term in expectation.source_terms
    )

    return RequirementExpectationEvaluation(
        expectation_id=(expectation.expectation_id),
        matched=True,
        actual_requirement_name=(requirement.name),
        category_correct=(requirement.category == expectation.category),
        match_terms_present=True,
        source_terms_present=(source_terms_present),
    )


def _find_requirement_match(
    *,
    expectation: ExpectedRequirement,
    requirements: list[ExtractedRequirement],
    used_requirement_indexes: set[int],
) -> (
    tuple[
        int,
        ExtractedRequirement,
    ]
    | None
):
    """Find one unused output containing all expected match terms."""

    candidates: list[
        tuple[
            int,
            ExtractedRequirement,
        ]
    ] = []

    for index, requirement in enumerate(requirements):
        if index in used_requirement_indexes:
            continue

        searchable_text = _normalize_text(
            " ".join(
                [
                    requirement.name,
                    requirement.evidence_expected,
                ]
            )
        )

        if all(
            _normalize_text(term) in searchable_text for term in expectation.match_terms
        ):
            candidates.append(
                (
                    index,
                    requirement,
                )
            )

    if not candidates:
        return None

    # Prefer the correctly classified candidate where duplicate
    # or overlapping model outputs exist. Count/deduplication is
    # evaluated separately.
    for candidate in candidates:
        if candidate[1].category == expectation.category:
            return candidate

    return candidates[0]


def _role_title_matches(
    *,
    expected: str | None,
    actual: str | None,
) -> bool:
    """Compare role titles without making wording fuzzy."""

    if expected is None:
        return actual is None

    if actual is None:
        return False

    return _normalize_text(expected) == _normalize_text(actual)


def _forbidden_terms_are_absent(
    *,
    case: RequirementExtractionEvalCase,
    actual: ExtractedRequirementSet,
) -> bool:
    """Ensure prohibited concepts were not extracted."""

    output_text_parts: list[str] = []

    if actual.role_title is not None:
        output_text_parts.append(actual.role_title)

    for requirement in actual.requirements:
        output_text_parts.extend(
            [
                requirement.name,
                requirement.evidence_expected,
                requirement.source_text,
            ]
        )

    normalized_output = _normalize_text(" ".join(output_text_parts))

    return all(
        _normalize_text(forbidden_term) not in normalized_output
        for forbidden_term in case.forbidden_terms
    )


def _all_source_text_is_grounded(
    *,
    job_description: str,
    requirements: list[ExtractedRequirement],
) -> bool:
    """Require every cited source fragment to be grounded in the posting."""

    return all(
        _source_text_is_grounded(
            source_text=requirement.source_text,
            job_description=job_description,
        )
        for requirement in requirements
    )


def _source_text_is_grounded(
    *,
    source_text: str,
    job_description: str,
) -> bool:
    """Accept one contiguous excerpt or multiple individually grounded excerpts."""

    normalized_description = _normalize_text(job_description)

    normalized_source = _normalize_text(source_text)

    if normalized_source in normalized_description:
        return True

    source_segments = [
        segment.strip()
        for segment in re.split(
            r"(?<=[.!?])\s+",
            source_text,
        )
        if segment.strip()
    ]

    if len(source_segments) <= 1:
        return False

    return all(
        _normalize_text(segment) in normalized_description
        for segment in source_segments
    )


def _normalize_text(
    value: str,
) -> str:
    """Normalize text for deterministic case-insensitive comparison."""

    return " ".join(value.casefold().split())
