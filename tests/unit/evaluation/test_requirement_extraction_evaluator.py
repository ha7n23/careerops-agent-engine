"""Tests for deterministic requirement-extraction evaluation."""

from pathlib import Path

import pytest

from careerops_agent_engine.domain.enums import (
    RequirementCategory,
)
from careerops_agent_engine.evaluation.datasets import (
    load_requirement_extraction_dataset,
)
from careerops_agent_engine.evaluation.models import (
    RequirementExtractionEvalCase,
)
from careerops_agent_engine.evaluation.requirement_extraction import (
    evaluate_requirement_extraction,
)
from careerops_agent_engine.infrastructure.llm.schemas import (
    ExtractedRequirement,
    ExtractedRequirementSet,
)

DATASET_PATH = (
    Path(__file__).resolve().parents[3] / "evals" / "requirement_extraction" / "v1.json"
)


@pytest.fixture(scope="module")
def cases() -> dict[
    str,
    RequirementExtractionEvalCase,
]:
    """Load benchmark cases by stable identifier."""

    dataset = load_requirement_extraction_dataset(DATASET_PATH)

    return {case.case_id: case for case in dataset.cases}


def test_correct_multi_requirement_output_passes(
    cases: dict[
        str,
        RequirementExtractionEvalCase,
    ],
) -> None:
    """Correct extraction should satisfy every deterministic gate."""

    actual = ExtractedRequirementSet(
        role_title=("Machine Learning Engineer"),
        requirements=[
            ExtractedRequirement(
                name=("Python software engineering"),
                category=(RequirementCategory.ESSENTIAL),
                evidence_expected=("Strong Python software engineering skills."),
                importance_score=5,
                source_text=("Strong Python software engineering skills."),
            ),
            ExtractedRequirement(
                name="FastAPI development",
                category=(RequirementCategory.ESSENTIAL),
                evidence_expected=("Experience building REST APIs with FastAPI."),
                importance_score=5,
                source_text=("Experience building REST APIs with FastAPI."),
            ),
            ExtractedRequirement(
                name="Kubernetes exposure",
                category=(RequirementCategory.DESIRABLE),
                evidence_expected=("Exposure to Kubernetes."),
                importance_score=2,
                source_text=("Exposure to Kubernetes is beneficial."),
            ),
        ],
    )

    result = evaluate_requirement_extraction(
        case=cases["REQ-EVAL-002"],
        actual=actual,
    )

    assert result.passed is True
    assert result.score == 1.0


def test_wrong_category_fails(
    cases: dict[
        str,
        RequirementExtractionEvalCase,
    ],
) -> None:
    """Essential/desirable misclassification must be measurable."""

    case = cases["REQ-EVAL-001"]

    actual = ExtractedRequirementSet(
        role_title="Junior AI Engineer",
        requirements=[
            ExtractedRequirement(
                name="FastAPI",
                category=(RequirementCategory.DESIRABLE),
                evidence_expected=("Hands-on FastAPI experience."),
                importance_score=2,
                source_text=("Hands-on experience building APIs with FastAPI."),
            )
        ],
    )

    result = evaluate_requirement_extraction(
        case=case,
        actual=actual,
    )

    assert result.passed is False
    assert result.expectation_results[0].category_correct is False


def test_forbidden_invention_fails(
    cases: dict[
        str,
        RequirementExtractionEvalCase,
    ],
) -> None:
    """Inventing an absent technology must fail the safety gate."""

    case = cases["REQ-EVAL-003"]

    actual = ExtractedRequirementSet(
        role_title="AI Platform Engineer",
        requirements=[
            ExtractedRequirement(
                name="Docker",
                category=(RequirementCategory.ESSENTIAL),
                evidence_expected=("Containerisation with Docker."),
                importance_score=5,
                source_text=("Experience containerising applications with Docker."),
            ),
            ExtractedRequirement(
                name="PostgreSQL and AWS",
                category=(RequirementCategory.ESSENTIAL),
                evidence_expected=("PostgreSQL persistence with AWS deployment."),
                importance_score=5,
                source_text=(
                    "Experience using PostgreSQL for application persistence."
                ),
            ),
        ],
    )

    result = evaluate_requirement_extraction(
        case=case,
        actual=actual,
    )

    assert result.passed is False
    assert result.forbidden_terms_absent is False


def test_ungrounded_source_text_fails(
    cases: dict[
        str,
        RequirementExtractionEvalCase,
    ],
) -> None:
    """Source text must actually occur in the job posting."""

    case = cases["REQ-EVAL-001"]

    actual = ExtractedRequirementSet(
        role_title="Junior AI Engineer",
        requirements=[
            ExtractedRequirement(
                name="FastAPI",
                category=(RequirementCategory.ESSENTIAL),
                evidence_expected=("Hands-on FastAPI experience."),
                importance_score=5,
                source_text=("Five years of FastAPI experience is required."),
            )
        ],
    )

    result = evaluate_requirement_extraction(
        case=case,
        actual=actual,
    )

    assert result.passed is False
    assert result.source_grounding_correct is False


def test_duplicate_requirement_fails_count_gate(
    cases: dict[
        str,
        RequirementExtractionEvalCase,
    ],
) -> None:
    """Duplicate extraction must fail even when the capability is correct."""

    case = cases["REQ-EVAL-005"]

    actual = ExtractedRequirementSet(
        role_title="Python Engineer",
        requirements=[
            ExtractedRequirement(
                name="Python programming",
                category=(RequirementCategory.ESSENTIAL),
                evidence_expected=("Strong Python programming skills."),
                importance_score=5,
                source_text=("Strong Python programming skills are required."),
            ),
            ExtractedRequirement(
                name="Python proficiency",
                category=(RequirementCategory.ESSENTIAL),
                evidence_expected=("Proficiency in Python."),
                importance_score=5,
                source_text=("Candidates must be proficient in Python."),
            ),
        ],
    )

    result = evaluate_requirement_extraction(
        case=case,
        actual=actual,
    )

    assert result.passed is False
    assert result.requirement_count_correct is False


def test_missing_role_title_case_accepts_none(
    cases: dict[
        str,
        RequirementExtractionEvalCase,
    ],
) -> None:
    """The evaluator must reward not inventing an absent role title."""

    case = cases["REQ-EVAL-006"]

    actual = ExtractedRequirementSet(
        role_title=None,
        requirements=[
            ExtractedRequirement(
                name="FastAPI",
                category=(RequirementCategory.ESSENTIAL),
                evidence_expected=("Hands-on experience with FastAPI."),
                importance_score=5,
                source_text=("Hands-on experience building REST APIs with FastAPI."),
            ),
            ExtractedRequirement(
                name="Docker familiarity",
                category=(RequirementCategory.DESIRABLE),
                evidence_expected=("Familiarity with Docker."),
                importance_score=2,
                source_text=("Familiarity with Docker."),
            ),
        ],
    )

    result = evaluate_requirement_extraction(
        case=case,
        actual=actual,
    )

    assert result.role_title_correct is True
    assert result.passed is True
    assert result.score == 1.0


def test_multiple_grounded_source_sentences_are_accepted(
    cases: dict[
        str,
        RequirementExtractionEvalCase,
    ],
) -> None:
    """Deduplicated requirements may combine separately grounded excerpts."""

    case = cases["REQ-EVAL-005"]

    actual = ExtractedRequirementSet(
        role_title="Python Engineer",
        requirements=[
            ExtractedRequirement(
                name="Python Programming",
                category=(RequirementCategory.ESSENTIAL),
                evidence_expected=(
                    "Demonstrated experience and proficiency in Python programming."
                ),
                importance_score=5,
                source_text=(
                    "Strong Python programming "
                    "skills are required. "
                    "Candidates must be "
                    "proficient in Python."
                ),
            )
        ],
    )

    result = evaluate_requirement_extraction(
        case=case,
        actual=actual,
    )

    assert result.source_grounding_correct is True

    assert result.passed is True
    assert result.score == 1.0


def test_partially_fabricated_multi_sentence_source_fails(
    cases: dict[
        str,
        RequirementExtractionEvalCase,
    ],
) -> None:
    """Every excerpt in a combined source citation must be grounded."""

    case = cases["REQ-EVAL-005"]

    actual = ExtractedRequirementSet(
        role_title="Python Engineer",
        requirements=[
            ExtractedRequirement(
                name="Python Programming",
                category=(RequirementCategory.ESSENTIAL),
                evidence_expected=(
                    "Demonstrated experience and proficiency in Python programming."
                ),
                importance_score=5,
                source_text=(
                    "Strong Python programming "
                    "skills are required. "
                    "Five years of AWS experience "
                    "is required."
                ),
            )
        ],
    )

    result = evaluate_requirement_extraction(
        case=case,
        actual=actual,
    )

    assert result.source_grounding_correct is False

    assert result.passed is False
