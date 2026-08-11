"""Run deterministic CareerOps evaluation datasets."""

from collections.abc import Mapping

from careerops_agent_engine.evaluation.models import (
    RequirementExtractionEvalDataset,
    RequirementExtractionEvaluationReport,
)
from careerops_agent_engine.evaluation.requirement_extraction import (
    evaluate_requirement_extraction,
)
from careerops_agent_engine.infrastructure.llm.schemas import (
    ExtractedRequirementSet,
)


def evaluate_requirement_extraction_dataset(
    *,
    dataset: RequirementExtractionEvalDataset,
    outputs: Mapping[
        str,
        ExtractedRequirementSet,
    ],
) -> RequirementExtractionEvaluationReport:
    """Evaluate outputs for every case in one benchmark dataset."""

    expected_case_ids = {case.case_id for case in dataset.cases}

    supplied_case_ids = set(outputs)

    missing_case_ids = expected_case_ids - supplied_case_ids

    if missing_case_ids:
        raise ValueError(
            "Missing evaluation outputs for cases: "
            + ", ".join(sorted(missing_case_ids))
        )

    unexpected_case_ids = supplied_case_ids - expected_case_ids

    if unexpected_case_ids:
        raise ValueError(
            "Unexpected evaluation outputs for cases: "
            + ", ".join(sorted(unexpected_case_ids))
        )

    results = [
        evaluate_requirement_extraction(
            case=case,
            actual=outputs[case.case_id],
        )
        for case in dataset.cases
    ]

    passed_cases = sum(1 for result in results if result.passed)

    failed_cases = len(results) - passed_cases

    average_score = sum(result.score for result in results) / len(results)

    pass_rate = passed_cases / len(results)

    return RequirementExtractionEvaluationReport(
        dataset_name=(dataset.dataset_name),
        dataset_version=(dataset.version),
        total_cases=len(results),
        passed_cases=(passed_cases),
        failed_cases=(failed_cases),
        pass_rate=(pass_rate),
        average_score=(average_score),
        results=results,
    )
