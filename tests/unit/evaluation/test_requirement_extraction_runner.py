"""Tests for deterministic CareerOps evaluation runs."""

import json
from pathlib import Path

import pytest

from careerops_agent_engine.domain.enums import (
    RequirementCategory,
)
from careerops_agent_engine.evaluation.datasets import (
    load_requirement_extraction_dataset,
)
from careerops_agent_engine.evaluation.reporting import (
    write_requirement_extraction_report,
)
from careerops_agent_engine.evaluation.runner import (
    evaluate_requirement_extraction_dataset,
)
from careerops_agent_engine.infrastructure.llm.schemas import (
    ExtractedRequirement,
    ExtractedRequirementSet,
)

DATASET_PATH = (
    Path(__file__).resolve().parents[3] / "evals" / "requirement_extraction" / "v1.json"
)


def _build_reference_outputs() -> dict[
    str,
    ExtractedRequirementSet,
]:
    """Build deterministic outputs satisfying the committed benchmark."""

    return {
        "REQ-EVAL-001": (
            ExtractedRequirementSet(
                role_title=("Junior AI Engineer"),
                requirements=[
                    ExtractedRequirement(
                        name="FastAPI",
                        category=(RequirementCategory.ESSENTIAL),
                        evidence_expected=(
                            "Hands-on experience building APIs with FastAPI."
                        ),
                        importance_score=5,
                        source_text=("Hands-on experience building APIs with FastAPI."),
                    )
                ],
            )
        ),
        "REQ-EVAL-002": (
            ExtractedRequirementSet(
                role_title=("Machine Learning Engineer"),
                requirements=[
                    ExtractedRequirement(
                        name="Python",
                        category=(RequirementCategory.ESSENTIAL),
                        evidence_expected=(
                            "Strong Python software engineering skills."
                        ),
                        importance_score=5,
                        source_text=("Strong Python software engineering skills."),
                    ),
                    ExtractedRequirement(
                        name="FastAPI",
                        category=(RequirementCategory.ESSENTIAL),
                        evidence_expected=(
                            "Experience building REST APIs with FastAPI."
                        ),
                        importance_score=5,
                        source_text=("Experience building REST APIs with FastAPI."),
                    ),
                    ExtractedRequirement(
                        name="Kubernetes",
                        category=(RequirementCategory.DESIRABLE),
                        evidence_expected=("Exposure to Kubernetes."),
                        importance_score=2,
                        source_text=("Exposure to Kubernetes is beneficial."),
                    ),
                ],
            )
        ),
        "REQ-EVAL-003": (
            ExtractedRequirementSet(
                role_title=("AI Platform Engineer"),
                requirements=[
                    ExtractedRequirement(
                        name="Docker",
                        category=(RequirementCategory.ESSENTIAL),
                        evidence_expected=(
                            "Experience containerising applications with Docker."
                        ),
                        importance_score=5,
                        source_text=(
                            "Experience containerising applications with Docker."
                        ),
                    ),
                    ExtractedRequirement(
                        name="PostgreSQL",
                        category=(RequirementCategory.ESSENTIAL),
                        evidence_expected=(
                            "Experience using PostgreSQL for application persistence."
                        ),
                        importance_score=5,
                        source_text=(
                            "Experience using PostgreSQL for application persistence."
                        ),
                    ),
                ],
            )
        ),
        "REQ-EVAL-004": (
            ExtractedRequirementSet(
                role_title=("Applied AI Engineer"),
                requirements=[
                    ExtractedRequirement(
                        name="Python",
                        category=(RequirementCategory.ESSENTIAL),
                        evidence_expected=("Strong Python programming skills."),
                        importance_score=5,
                        source_text=("Strong Python programming skills."),
                    )
                ],
            )
        ),
        "REQ-EVAL-005": (
            ExtractedRequirementSet(
                role_title=("Python Engineer"),
                requirements=[
                    ExtractedRequirement(
                        name="Python",
                        category=(RequirementCategory.ESSENTIAL),
                        evidence_expected=("Strong Python programming skills."),
                        importance_score=5,
                        source_text=("Strong Python programming skills are required."),
                    )
                ],
            )
        ),
        "REQ-EVAL-006": (
            ExtractedRequirementSet(
                role_title=None,
                requirements=[
                    ExtractedRequirement(
                        name="FastAPI",
                        category=(RequirementCategory.ESSENTIAL),
                        evidence_expected=(
                            "Hands-on experience building REST APIs with FastAPI."
                        ),
                        importance_score=5,
                        source_text=(
                            "Hands-on experience building REST APIs with FastAPI."
                        ),
                    ),
                    ExtractedRequirement(
                        name="Docker",
                        category=(RequirementCategory.DESIRABLE),
                        evidence_expected=("Familiarity with Docker."),
                        importance_score=2,
                        source_text=("Familiarity with Docker."),
                    ),
                ],
            )
        ),
    }


def test_reference_outputs_produce_perfect_report() -> None:
    """A fully correct benchmark run should score perfectly."""

    dataset = load_requirement_extraction_dataset(DATASET_PATH)

    report = evaluate_requirement_extraction_dataset(
        dataset=dataset,
        outputs=(_build_reference_outputs()),
    )

    assert report.total_cases == 6
    assert report.passed_cases == 6
    assert report.failed_cases == 0
    assert report.pass_rate == 1.0
    assert report.average_score == 1.0

    assert all(result.passed for result in report.results)


def test_failed_case_is_reflected_in_report() -> None:
    """One bad output should reduce aggregate benchmark metrics."""

    dataset = load_requirement_extraction_dataset(DATASET_PATH)

    outputs = _build_reference_outputs()

    outputs["REQ-EVAL-001"] = ExtractedRequirementSet(
        role_title=("Junior AI Engineer"),
        requirements=[
            ExtractedRequirement(
                name="FastAPI",
                category=(RequirementCategory.DESIRABLE),
                evidence_expected=("Hands-on experience building APIs with FastAPI."),
                importance_score=2,
                source_text=("Hands-on experience building APIs with FastAPI."),
            )
        ],
    )

    report = evaluate_requirement_extraction_dataset(
        dataset=dataset,
        outputs=outputs,
    )

    assert report.total_cases == 6
    assert report.passed_cases == 5
    assert report.failed_cases == 1
    assert report.pass_rate == pytest.approx(5 / 6)
    assert report.average_score < 1.0


def test_missing_case_output_is_rejected() -> None:
    """Incomplete benchmark runs must not look valid."""

    dataset = load_requirement_extraction_dataset(DATASET_PATH)

    outputs = _build_reference_outputs()

    del outputs["REQ-EVAL-006"]

    with pytest.raises(
        ValueError,
        match=("Missing evaluation outputs for cases: REQ-EVAL-006"),
    ):
        evaluate_requirement_extraction_dataset(
            dataset=dataset,
            outputs=outputs,
        )


def test_unexpected_case_output_is_rejected() -> None:
    """Unknown case identifiers must not enter the report."""

    dataset = load_requirement_extraction_dataset(DATASET_PATH)

    outputs = _build_reference_outputs()

    outputs["REQ-EVAL-999"] = outputs["REQ-EVAL-001"]

    with pytest.raises(
        ValueError,
        match=("Unexpected evaluation outputs for cases: REQ-EVAL-999"),
    ):
        evaluate_requirement_extraction_dataset(
            dataset=dataset,
            outputs=outputs,
        )


def test_report_can_be_written_as_json(
    tmp_path: Path,
) -> None:
    """Evaluation reports should be portable outside Python."""

    dataset = load_requirement_extraction_dataset(DATASET_PATH)

    report = evaluate_requirement_extraction_dataset(
        dataset=dataset,
        outputs=(_build_reference_outputs()),
    )

    report_path = tmp_path / "requirement-extraction-report.json"

    write_requirement_extraction_report(
        report=report,
        path=report_path,
    )

    payload = json.loads(
        report_path.read_text(
            encoding="utf-8",
        )
    )

    assert payload["dataset_name"] == ("careerops-requirement-extraction")

    assert payload["dataset_version"] == "1.0.0"

    assert payload["passed_cases"] == 6

    assert len(payload["results"]) == 6
