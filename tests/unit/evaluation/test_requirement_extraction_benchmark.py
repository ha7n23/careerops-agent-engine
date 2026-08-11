"""Tests for resumable real-model requirement benchmarks."""

import json
from pathlib import Path

import pytest

from careerops_agent_engine.domain.models.job import (
    JobRequirement,
    JobRequirementExtraction,
)
from careerops_agent_engine.evaluation.benchmark import (
    build_requirement_extraction_benchmark_identity,
    run_requirement_extraction_benchmark,
)
from careerops_agent_engine.evaluation.datasets import (
    load_requirement_extraction_dataset,
)
from careerops_agent_engine.evaluation.models import (
    RequirementExtractionBenchmarkProgress,
    RequirementExtractionEvalCase,
)
from careerops_agent_engine.evaluation.reporting import (
    write_requirement_extraction_benchmark_result,
)

DATASET_PATH = (
    Path(__file__).resolve().parents[3] / "evals" / "requirement_extraction" / "v1.json"
)


def _reference_extraction(
    case: RequirementExtractionEvalCase,
) -> JobRequirementExtraction:
    """Build one deterministic output satisfying a benchmark case."""

    return JobRequirementExtraction(
        role_title=case.expected_role_title,
        requirements=[
            JobRequirement(
                requirement_id=(f"REQ-{index:03d}"),
                name=" ".join(expectation.match_terms),
                category=(expectation.category),
                evidence_expected=" ".join(expectation.match_terms),
                importance_score=(
                    5 if (expectation.category.value == "essential") else 2
                ),
                source_text=" ".join(expectation.source_terms),
            )
            for index, expectation in enumerate(
                case.expected_requirements,
                start=1,
            )
        ],
    )


class FakeRequirementExtractor:
    """Deterministic test double for benchmark execution."""

    def __init__(
        self,
        *,
        cases: list[RequirementExtractionEvalCase],
        fail_on_case_id: str | None = None,
    ) -> None:
        """Prepare reference outputs and optional failure."""

        self.calls: list[str] = []
        self._fail_on_case_id = fail_on_case_id
        self._outputs = {case.case_id: (_reference_extraction(case)) for case in cases}

    def extract(
        self,
        job_description: str,
        *,
        job_id: str,
    ) -> JobRequirementExtraction:
        """Return the reference output for one synthetic case."""

        del job_description

        case_id = job_id.removeprefix("JOB-")

        self.calls.append(case_id)

        if case_id == self._fail_on_case_id:
            raise RuntimeError("synthetic provider failure")

        return self._outputs[case_id]


def _identity():
    dataset = load_requirement_extraction_dataset(DATASET_PATH)

    return build_requirement_extraction_benchmark_identity(
        dataset=dataset,
        dataset_path=DATASET_PATH,
        provider="google",
        model_name="test-model",
        prompt_version="test-prompt-v1",
        temperature=1.0,
    )


def test_benchmark_completes_and_persists_outputs(
    tmp_path: Path,
) -> None:
    """A complete run should persist every case and score perfectly."""

    dataset = load_requirement_extraction_dataset(DATASET_PATH)

    extractor = FakeRequirementExtractor(cases=dataset.cases)

    progress_path = tmp_path / "progress.json"

    result = run_requirement_extraction_benchmark(
        dataset=dataset,
        extractor=extractor,
        identity=_identity(),
        progress_path=progress_path,
    )

    assert len(extractor.calls) == 6

    assert result.report.passed_cases == 6
    assert result.report.pass_rate == 1.0

    progress = RequirementExtractionBenchmarkProgress.model_validate_json(
        progress_path.read_text(
            encoding="utf-8",
        )
    )

    assert len(progress.outputs) == 6


def test_completed_progress_is_reused(
    tmp_path: Path,
) -> None:
    """A rerun should make zero model calls for completed cases."""

    dataset = load_requirement_extraction_dataset(DATASET_PATH)

    progress_path = tmp_path / "progress.json"

    first_extractor = FakeRequirementExtractor(cases=dataset.cases)

    run_requirement_extraction_benchmark(
        dataset=dataset,
        extractor=first_extractor,
        identity=_identity(),
        progress_path=progress_path,
    )

    second_extractor = FakeRequirementExtractor(cases=dataset.cases)

    result = run_requirement_extraction_benchmark(
        dataset=dataset,
        extractor=second_extractor,
        identity=_identity(),
        progress_path=progress_path,
    )

    assert second_extractor.calls == []
    assert result.report.pass_rate == 1.0


def test_partial_progress_survives_failure(
    tmp_path: Path,
) -> None:
    """Completed cases must survive a later provider failure."""

    dataset = load_requirement_extraction_dataset(DATASET_PATH)

    extractor = FakeRequirementExtractor(
        cases=dataset.cases,
        fail_on_case_id="REQ-EVAL-002",
    )

    progress_path = tmp_path / "progress.json"

    with pytest.raises(
        RuntimeError,
        match="synthetic provider failure",
    ):
        run_requirement_extraction_benchmark(
            dataset=dataset,
            extractor=extractor,
            identity=_identity(),
            progress_path=progress_path,
        )

    progress = RequirementExtractionBenchmarkProgress.model_validate_json(
        progress_path.read_text(
            encoding="utf-8",
        )
    )

    assert set(progress.outputs) == {"REQ-EVAL-001"}


def test_different_benchmark_identity_is_rejected(
    tmp_path: Path,
) -> None:
    """Cached output must not cross model or prompt configurations."""

    dataset = load_requirement_extraction_dataset(DATASET_PATH)

    progress_path = tmp_path / "progress.json"

    extractor = FakeRequirementExtractor(cases=dataset.cases)

    identity = _identity()

    run_requirement_extraction_benchmark(
        dataset=dataset,
        extractor=extractor,
        identity=identity,
        progress_path=progress_path,
    )

    changed_identity = identity.model_copy(update={"model_name": ("different-model")})

    with pytest.raises(
        ValueError,
        match=("different dataset/model/prompt"),
    ):
        run_requirement_extraction_benchmark(
            dataset=dataset,
            extractor=FakeRequirementExtractor(cases=dataset.cases),
            identity=changed_identity,
            progress_path=progress_path,
        )


def test_benchmark_result_is_portable_json(
    tmp_path: Path,
) -> None:
    """Completed benchmark output should retain provenance and scores."""

    dataset = load_requirement_extraction_dataset(DATASET_PATH)

    result = run_requirement_extraction_benchmark(
        dataset=dataset,
        extractor=FakeRequirementExtractor(cases=dataset.cases),
        identity=_identity(),
        progress_path=(tmp_path / "progress.json"),
    )

    result_path = tmp_path / "result.json"

    write_requirement_extraction_benchmark_result(
        result=result,
        path=result_path,
    )

    payload = json.loads(
        result_path.read_text(
            encoding="utf-8",
        )
    )

    assert payload["identity"]["model_name"] == "test-model"

    assert payload["identity"]["prompt_version"] == "test-prompt-v1"

    assert payload["report"]["passed_cases"] == 6

    assert len(payload["outputs"]) == 6
