"""Tests for the CareerOps requirement-extraction benchmark."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from careerops_agent_engine.evaluation.datasets import (
    load_requirement_extraction_dataset,
)
from careerops_agent_engine.evaluation.models import (
    RequirementExtractionEvalCase,
    RequirementExtractionEvalDataset,
)

DATASET_PATH = (
    Path(__file__).resolve().parents[3] / "evals" / "requirement_extraction" / "v1.json"
)


def test_requirement_extraction_dataset_loads() -> None:
    """The committed gold dataset should remain valid."""

    dataset = load_requirement_extraction_dataset(DATASET_PATH)

    assert dataset.dataset_name == "careerops-requirement-extraction"
    assert dataset.version == "1.0.0"
    assert len(dataset.cases) == 6


def test_security_case_preserves_injection_boundary() -> None:
    """The benchmark must explicitly test prompt injection."""

    dataset = load_requirement_extraction_dataset(DATASET_PATH)

    case = next(case for case in dataset.cases if case.case_id == "REQ-EVAL-004")

    assert case.expected_requirement_count == 1
    assert case.expected_requirements[0].match_terms == ["Python"]
    assert "AWS" in case.forbidden_terms
    assert "AWS" in case.job_description


def test_duplicate_case_ids_are_rejected() -> None:
    """Stable evaluation identifiers must be unique."""

    dataset = load_requirement_extraction_dataset(DATASET_PATH)

    with pytest.raises(
        ValidationError,
        match=("Evaluation case identifiers must be unique"),
    ):
        RequirementExtractionEvalDataset(
            dataset_name=("careerops-requirement-extraction"),
            version="1.0.0",
            cases=[
                dataset.cases[0],
                dataset.cases[0],
            ],
        )


def test_requirement_count_mismatch_is_rejected() -> None:
    """Gold counts must agree with gold requirement records."""

    dataset = load_requirement_extraction_dataset(DATASET_PATH)

    payload = dataset.cases[0].model_dump()
    payload["expected_requirement_count"] = 99

    with pytest.raises(
        ValidationError,
        match=("Expected requirement count must equal"),
    ):
        RequirementExtractionEvalCase.model_validate(payload)


def test_missing_source_term_is_rejected() -> None:
    """Gold grounding terms must exist in the source posting."""

    dataset = load_requirement_extraction_dataset(DATASET_PATH)

    payload = dataset.cases[0].model_dump()

    payload["expected_requirements"][0]["source_terms"] = ["Kubernetes"]

    with pytest.raises(
        ValidationError,
        match=("Gold source terms must exist in the job description"),
    ):
        RequirementExtractionEvalCase.model_validate(payload)
