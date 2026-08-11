"""Typed contracts for CareerOps offline evaluation datasets."""

from typing import Self

from pydantic import Field, model_validator

from careerops_agent_engine.domain.enums import (
    RequirementCategory,
)
from careerops_agent_engine.domain.models.base import (
    DomainModel,
)
from careerops_agent_engine.domain.models.job import (
    JobRequirementExtraction,
)


class ExpectedRequirement(DomainModel):
    """Gold expectation for one extracted job requirement."""

    expectation_id: str = Field(
        min_length=1,
        max_length=64,
    )
    category: RequirementCategory
    match_terms: list[str] = Field(
        min_length=1,
    )
    source_terms: list[str] = Field(
        min_length=1,
    )

    @model_validator(mode="after")
    def validate_terms(self) -> Self:
        """Require useful and non-duplicated comparison terms."""

        for field_name, terms in (
            (
                "match_terms",
                self.match_terms,
            ),
            (
                "source_terms",
                self.source_terms,
            ),
        ):
            normalized_terms = [term.strip().casefold() for term in terms]

            if any(not term for term in normalized_terms):
                raise ValueError(f"{field_name} cannot contain blank terms.")

            if len(normalized_terms) != len(set(normalized_terms)):
                raise ValueError(f"{field_name} must contain unique terms.")

        return self


class RequirementExtractionEvalCase(DomainModel):
    """One deterministic job-requirement benchmark case."""

    case_id: str = Field(
        min_length=1,
        max_length=64,
    )
    purpose: str = Field(
        min_length=1,
        max_length=500,
    )
    tags: list[str] = Field(
        min_length=1,
    )
    job_description: str = Field(
        min_length=1,
        max_length=20_000,
    )
    expected_role_title: str | None = Field(
        default=None,
        max_length=200,
    )
    expected_requirement_count: int = Field(
        ge=1,
    )
    expected_requirements: list[ExpectedRequirement] = Field(
        min_length=1,
    )
    forbidden_terms: list[str] = Field(
        default_factory=list,
    )

    @model_validator(mode="after")
    def validate_case_contract(self) -> Self:
        """Reject internally inconsistent gold cases."""

        if self.expected_requirement_count != len(self.expected_requirements):
            raise ValueError(
                "Expected requirement count must equal the number of gold requirements."
            )

        expectation_ids = [
            expectation.expectation_id for expectation in self.expected_requirements
        ]

        if len(expectation_ids) != len(set(expectation_ids)):
            raise ValueError("Expectation identifiers must be unique within a case.")

        normalized_description = self.job_description.casefold()

        for expectation in self.expected_requirements:
            missing_source_terms = [
                term
                for term in expectation.source_terms
                if (term.casefold() not in normalized_description)
            ]

            if missing_source_terms:
                raise ValueError("Gold source terms must exist in the job description.")

        return self


class RequirementExtractionEvalDataset(DomainModel):
    """Versioned CareerOps requirement-extraction benchmark."""

    dataset_name: str = Field(
        min_length=1,
        max_length=100,
    )
    version: str = Field(
        pattern=r"^\d+\.\d+\.\d+$",
    )
    cases: list[RequirementExtractionEvalCase] = Field(
        min_length=1,
    )

    @model_validator(mode="after")
    def validate_unique_case_ids(self) -> Self:
        """Require stable unique case identifiers."""

        case_ids = [case.case_id for case in self.cases]

        if len(case_ids) != len(set(case_ids)):
            raise ValueError("Evaluation case identifiers must be unique.")

        return self


class RequirementExpectationEvaluation(DomainModel):
    """Evaluation result for one expected requirement."""

    expectation_id: str = Field(
        min_length=1,
        max_length=64,
    )
    matched: bool
    actual_requirement_name: str | None = Field(
        default=None,
        max_length=200,
    )
    category_correct: bool
    match_terms_present: bool
    source_terms_present: bool


class RequirementExtractionEvaluation(DomainModel):
    """Deterministic score for one requirement-extraction case."""

    case_id: str = Field(
        min_length=1,
        max_length=64,
    )
    role_title_correct: bool
    requirement_count_correct: bool
    forbidden_terms_absent: bool
    source_grounding_correct: bool

    expectation_results: list[RequirementExpectationEvaluation] = Field(
        min_length=1,
    )

    score: float = Field(
        ge=0.0,
        le=1.0,
    )
    passed: bool


class RequirementExtractionEvaluationReport(DomainModel):
    """Aggregate deterministic requirement-extraction benchmark report."""

    dataset_name: str = Field(
        min_length=1,
        max_length=100,
    )
    dataset_version: str = Field(
        pattern=r"^\d+\.\d+\.\d+$",
    )
    total_cases: int = Field(
        ge=1,
    )
    passed_cases: int = Field(
        ge=0,
    )
    failed_cases: int = Field(
        ge=0,
    )
    pass_rate: float = Field(
        ge=0.0,
        le=1.0,
    )
    average_score: float = Field(
        ge=0.0,
        le=1.0,
    )
    results: list[RequirementExtractionEvaluation] = Field(
        min_length=1,
    )

    @model_validator(mode="after")
    def validate_report_totals(self) -> Self:
        """Require internally consistent aggregate metrics."""

        if self.total_cases != len(self.results):
            raise ValueError("Report total must equal the number of case results.")

        if self.passed_cases + self.failed_cases != self.total_cases:
            raise ValueError(
                "Passed and failed case totals must equal the report total."
            )

        return self


class RequirementExtractionBenchmarkIdentity(DomainModel):
    """Immutable provenance for one real-model benchmark run."""

    dataset_name: str = Field(
        min_length=1,
        max_length=100,
    )
    dataset_version: str = Field(
        pattern=r"^\d+\.\d+\.\d+$",
    )
    dataset_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )
    provider: str = Field(
        min_length=1,
        max_length=50,
    )
    model_name: str = Field(
        min_length=1,
        max_length=200,
    )
    prompt_version: str = Field(
        min_length=1,
        max_length=100,
    )
    temperature: float = Field(
        ge=0.0,
        le=2.0,
    )


class RequirementExtractionBenchmarkProgress(DomainModel):
    """Resumable raw outputs for one benchmark identity."""

    identity: RequirementExtractionBenchmarkIdentity
    outputs: dict[
        str,
        JobRequirementExtraction,
    ] = Field(
        default_factory=dict,
    )


class RequirementExtractionBenchmarkResult(DomainModel):
    """Completed real-model benchmark with provenance and outputs."""

    identity: RequirementExtractionBenchmarkIdentity
    outputs: dict[
        str,
        JobRequirementExtraction,
    ]
    report: RequirementExtractionEvaluationReport

    @model_validator(mode="after")
    def validate_result_contract(self) -> Self:
        """Require the result, report, and provenance to agree."""

        if self.report.dataset_name != self.identity.dataset_name:
            raise ValueError(
                "Benchmark report dataset name must match benchmark identity."
            )

        if self.report.dataset_version != self.identity.dataset_version:
            raise ValueError(
                "Benchmark report dataset version must match benchmark identity."
            )

        result_case_ids = {result.case_id for result in self.report.results}

        if set(self.outputs) != result_case_ids:
            raise ValueError(
                "Benchmark outputs must match the evaluated case identifiers."
            )

        return self
