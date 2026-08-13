"""Tests for CareerOps LangSmith evaluation synchronization."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest
from langsmith import Client
from langsmith.schemas import Example, Run

from careerops_agent_engine.domain.enums import (
    RequirementCategory,
)
from careerops_agent_engine.domain.models.job import (
    JobRequirement,
    JobRequirementExtraction,
)
from careerops_agent_engine.evaluation.benchmark import (
    build_requirement_extraction_benchmark_identity,
)
from careerops_agent_engine.evaluation.datasets import (
    load_requirement_extraction_dataset,
)
from careerops_agent_engine.evaluation.langsmith import (
    LANGSMITH_MANAGED_BY,
    build_requirement_extraction_langsmith_experiment_contract,
    build_requirement_extraction_langsmith_target,
    evaluate_requirement_extraction_langsmith,
    sync_requirement_extraction_langsmith_dataset,
)

DATASET_PATH = (
    Path(__file__).resolve().parents[3] / "evals" / "requirement_extraction" / "v1.json"
)


@dataclass
class _FakeDataset:
    id: str
    name: str


@dataclass
class _FakeExample:
    id: str
    inputs: dict[
        str,
        Any,
    ]
    outputs: (
        dict[
            str,
            Any,
        ]
        | None
    )
    metadata: (
        dict[
            str,
            Any,
        ]
        | None
    )


@dataclass
class _FakeRun:
    """Minimal LangSmith run used by evaluator tests."""

    outputs: (
        dict[
            str,
            Any,
        ]
        | None
    )


class _FakeRequirementExtractor:
    """Deterministic target test double."""

    def __init__(
        self,
        extraction: JobRequirementExtraction,
    ) -> None:
        self.extraction = extraction
        self.calls: list[
            tuple[
                str,
                str,
            ]
        ] = []

    def extract(
        self,
        job_description: str,
        *,
        job_id: str,
    ) -> JobRequirementExtraction:
        self.calls.append(
            (
                job_description,
                job_id,
            )
        )

        return self.extraction


class _FakeLangSmithClient:
    """Small mutable LangSmith test double."""

    def __init__(self) -> None:
        self.datasets: list[_FakeDataset] = []

        self.examples: list[_FakeExample] = []

        self.create_calls = 0
        self.update_calls = 0
        self.delete_calls = 0

    def list_datasets(
        self,
        *,
        dataset_name: str,
    ) -> list[_FakeDataset]:
        return [dataset for dataset in self.datasets if dataset.name == dataset_name]

    def create_dataset(
        self,
        *,
        dataset_name: str,
        description: str,
    ) -> _FakeDataset:
        del description

        dataset = _FakeDataset(
            id="DS-001",
            name=dataset_name,
        )

        self.datasets.append(dataset)

        return dataset

    def list_examples(
        self,
        *,
        dataset_id: str,
    ) -> list[_FakeExample]:
        del dataset_id

        return list(self.examples)

    def create_examples(
        self,
        *,
        dataset_id: str,
        examples: list[
            dict[
                str,
                Any,
            ]
        ],
    ) -> None:
        del dataset_id

        for example in examples:
            self.create_calls += 1

            self.examples.append(
                _FakeExample(
                    id=(f"EX-{len(self.examples) + 1:03d}"),
                    inputs=dict(example["inputs"]),
                    outputs=dict(example["outputs"]),
                    metadata=dict(example["metadata"]),
                )
            )

    def update_example(
        self,
        *,
        example_id: str,
        inputs: dict[
            str,
            Any,
        ],
        outputs: dict[
            str,
            Any,
        ],
        metadata: dict[
            str,
            Any,
        ],
    ) -> None:
        self.update_calls += 1

        example = next(
            example for example in self.examples if example.id == str(example_id)
        )

        example.inputs = dict(inputs)
        example.outputs = dict(outputs)
        example.metadata = dict(metadata)

    def delete_example(
        self,
        *,
        example_id: str,
    ) -> None:
        self.delete_calls += 1

        self.examples = [
            example for example in self.examples if example.id != str(example_id)
        ]


def _dataset():
    return load_requirement_extraction_dataset(DATASET_PATH)


def _sync(
    fake: _FakeLangSmithClient,
):
    return sync_requirement_extraction_langsmith_dataset(
        client=cast(
            Client,
            fake,
        ),
        dataset=_dataset(),
        dataset_path=DATASET_PATH,
    )


def test_first_sync_creates_dataset_and_six_examples() -> None:
    """An absent remote benchmark should be created from Git."""

    fake = _FakeLangSmithClient()

    result = _sync(fake)

    assert result.dataset_created is True
    assert result.created_examples == 6
    assert result.updated_examples == 0
    assert result.deleted_examples == 0
    assert result.unchanged_examples == 0
    assert result.total_examples == 6

    assert len(fake.examples) == 6


def test_second_sync_is_idempotent() -> None:
    """Identical Git and LangSmith state should make zero writes."""

    fake = _FakeLangSmithClient()

    _sync(fake)

    first_create_calls = fake.create_calls

    result = _sync(fake)

    assert result.dataset_created is False
    assert result.created_examples == 0
    assert result.updated_examples == 0
    assert result.deleted_examples == 0
    assert result.unchanged_examples == 6

    assert fake.create_calls == first_create_calls
    assert fake.update_calls == 0
    assert fake.delete_calls == 0


def test_changed_managed_example_is_updated() -> None:
    """Remote drift should be corrected from the Git source."""

    fake = _FakeLangSmithClient()

    _sync(fake)

    fake.examples[0].inputs = {"job_description": ("remote drift")}

    result = _sync(fake)

    assert result.updated_examples == 1
    assert result.unchanged_examples == 5
    assert fake.update_calls == 1


def test_stale_managed_example_is_deleted() -> None:
    """Cases removed from Git should not remain in the managed dataset."""

    fake = _FakeLangSmithClient()

    _sync(fake)

    fake.examples.append(
        _FakeExample(
            id="EX-STALE",
            inputs={"job_description": ("stale")},
            outputs={"reference_case": {}},
            metadata={
                "managed_by": (LANGSMITH_MANAGED_BY),
                "case_id": ("REQ-EVAL-999"),
            },
        )
    )

    result = _sync(fake)

    assert result.deleted_examples == 1
    assert result.unchanged_examples == 6
    assert fake.delete_calls == 1


def test_unmanaged_remote_example_is_rejected() -> None:
    """The synchronizer must not delete or overwrite unknown data."""

    fake = _FakeLangSmithClient()

    _sync(fake)

    fake.examples.append(
        _FakeExample(
            id="EX-UNMANAGED",
            inputs={},
            outputs={},
            metadata={},
        )
    )

    with pytest.raises(
        ValueError,
        match="unmanaged example",
    ):
        _sync(fake)


def test_duplicate_remote_case_id_is_rejected() -> None:
    """Ambiguous remote case ownership must fail explicitly."""

    fake = _FakeLangSmithClient()

    _sync(fake)

    existing = fake.examples[0]

    fake.examples.append(
        _FakeExample(
            id="EX-DUPLICATE",
            inputs=dict(existing.inputs),
            outputs=dict(existing.outputs or {}),
            metadata=dict(existing.metadata or {}),
        )
    )

    with pytest.raises(
        ValueError,
        match="duplicate case_id",
    ):
        _sync(fake)


def test_experiment_contract_contains_provenance() -> None:
    """Future LangSmith experiments should remain comparable."""

    dataset = _dataset()

    identity = build_requirement_extraction_benchmark_identity(
        dataset=dataset,
        dataset_path=DATASET_PATH,
        provider="google",
        model_name="gemini-3.5-flash",
        prompt_version=("job-requirements-v1"),
        temperature=1.0,
    )

    contract = build_requirement_extraction_langsmith_experiment_contract(
        identity=identity,
    )

    assert contract.dataset_name == dataset.dataset_name

    assert "gemini-3.5-flash" in contract.experiment_prefix

    assert contract.metadata["prompt_version"] == "job-requirements-v1"

    assert contract.metadata["models"] == ["google:gemini-3.5-flash"]

    assert contract.metadata["dataset_sha256"] == identity.dataset_sha256


def test_langsmith_owned_metadata_does_not_trigger_update() -> None:
    """Remote system metadata should not create false Git drift."""

    fake = _FakeLangSmithClient()

    _sync(fake)

    metadata = fake.examples[0].metadata

    assert metadata is not None

    metadata["dataset_split"] = ["base"]

    result = _sync(fake)

    assert result.created_examples == 0
    assert result.updated_examples == 0
    assert result.deleted_examples == 0
    assert result.unchanged_examples == 6
    assert fake.update_calls == 0


def test_langsmith_target_uses_real_extractor_contract() -> None:
    """The LangSmith target should call the production extractor port."""

    extraction = JobRequirementExtraction(
        role_title="Junior AI Engineer",
        requirements=[
            JobRequirement(
                requirement_id="REQ-001",
                name="FastAPI",
                category=RequirementCategory.ESSENTIAL,
                evidence_expected=("Hands-on FastAPI experience."),
                importance_score=5,
                source_text=("Hands-on experience building APIs with FastAPI."),
            )
        ],
    )

    extractor = _FakeRequirementExtractor(extraction)

    target = build_requirement_extraction_langsmith_target(extractor=extractor)

    output = target(
        {
            "job_description": (
                "Junior AI Engineer\nHands-on experience building APIs with FastAPI."
            )
        }
    )

    assert output["extraction"]["role_title"] == "Junior AI Engineer"

    assert len(extractor.calls) == 1

    assert extractor.calls[0][1].startswith("JOB-LS-EVAL-")


def test_langsmith_target_rejects_missing_job_description() -> None:
    """Malformed remote dataset inputs should fail explicitly."""

    extraction = JobRequirementExtraction(
        role_title="Junior AI Engineer",
        requirements=[
            JobRequirement(
                requirement_id="REQ-001",
                name="FastAPI",
                category=RequirementCategory.ESSENTIAL,
                evidence_expected=("Hands-on FastAPI experience."),
                importance_score=5,
                source_text=("Hands-on experience building APIs with FastAPI."),
            )
        ],
    )

    target = build_requirement_extraction_langsmith_target(
        extractor=_FakeRequirementExtractor(extraction)
    )

    with pytest.raises(
        ValueError,
        match=("must contain a string job_description"),
    ):
        target({})


def test_langsmith_evaluator_returns_deterministic_metrics() -> None:
    """LangSmith feedback should reuse CareerOps benchmark rules."""

    dataset = _dataset()

    case = dataset.cases[0]

    extraction = JobRequirementExtraction(
        role_title="Junior AI Engineer",
        requirements=[
            JobRequirement(
                requirement_id="REQ-001",
                name="FastAPI",
                category=RequirementCategory.ESSENTIAL,
                evidence_expected=("Hands-on experience with FastAPI."),
                importance_score=5,
                source_text=("Hands-on experience building APIs with FastAPI."),
            )
        ],
    )

    run = cast(
        Run,
        _FakeRun(
            outputs={
                "extraction": extraction.model_dump(
                    mode="json",
                )
            }
        ),
    )

    example = cast(
        Example,
        _FakeExample(
            id="EX-EVAL-001",
            inputs={
                "job_description": case.job_description,
            },
            outputs={
                "reference_case": case.model_dump(
                    mode="json",
                )
            },
            metadata={},
        ),
    )

    results = evaluate_requirement_extraction_langsmith(
        run,
        example,
    )

    evaluation_results = results.get("results")

    assert evaluation_results is not None

    scores = {result.key: result.score for result in evaluation_results}

    assert scores["careerops_pass"] is True

    assert scores["careerops_score"] == 1.0

    assert scores["role_title_correct"] is True

    assert scores["expectation_accuracy"] == 1.0
