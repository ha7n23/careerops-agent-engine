"""LangSmith dataset synchronization for CareerOps evaluations."""

import re
from collections.abc import Callable
from hashlib import sha256
from pathlib import Path
from typing import Any

from langsmith import Client
from langsmith.evaluation import (
    EvaluationResult,
    EvaluationResults,
)
from langsmith.schemas import (
    Example,
    Run,
)

from careerops_agent_engine.application.ports.requirement_extractor import (
    RequirementExtractor,
)
from careerops_agent_engine.domain.models.job import (
    JobRequirementExtraction,
)
from careerops_agent_engine.evaluation.datasets import (
    compute_dataset_sha256,
)
from careerops_agent_engine.evaluation.models import (
    RequirementExtractionBenchmarkIdentity,
    RequirementExtractionEvalCase,
    RequirementExtractionEvalDataset,
    RequirementExtractionLangSmithDatasetSyncResult,
    RequirementExtractionLangSmithExperimentContract,
)
from careerops_agent_engine.evaluation.requirement_extraction import (
    evaluate_requirement_extraction,
    to_requirement_extraction_evaluation_output,
)

LANGSMITH_MANAGED_BY = "careerops-git"

LANGSMITH_DATASET_DESCRIPTION = (
    "Git-managed synthetic CareerOps requirement-extraction "
    "benchmark. The committed local evaluation dataset is "
    "the source of truth."
)

type LangSmithTarget = Callable[
    [
        dict[str, Any],
    ],
    dict[str, Any],
]


def sync_requirement_extraction_langsmith_dataset(
    *,
    client: Client,
    dataset: RequirementExtractionEvalDataset,
    dataset_path: Path,
) -> RequirementExtractionLangSmithDatasetSyncResult:
    """Mirror one committed requirement dataset into LangSmith."""

    dataset_sha256 = compute_dataset_sha256(dataset_path)

    matching_datasets = list(
        client.list_datasets(
            dataset_name=dataset.dataset_name,
        )
    )

    if len(matching_datasets) > 1:
        raise ValueError(
            "Multiple LangSmith datasets have the same exact CareerOps dataset name."
        )

    dataset_created = False

    if matching_datasets:
        remote_dataset = matching_datasets[0]
    else:
        remote_dataset = client.create_dataset(
            dataset_name=(dataset.dataset_name),
            description=(LANGSMITH_DATASET_DESCRIPTION),
        )
        dataset_created = True

    remote_examples = list(
        client.list_examples(
            dataset_id=remote_dataset.id,
        )
    )

    remote_by_case_id: dict[
        str,
        Any,
    ] = {}

    for example in remote_examples:
        metadata = dict(example.metadata or {})

        managed_by = metadata.get("managed_by")

        case_id = metadata.get("case_id")

        if (
            managed_by != LANGSMITH_MANAGED_BY
            or not isinstance(
                case_id,
                str,
            )
            or not case_id.strip()
        ):
            raise ValueError(
                "LangSmith dataset contains an "
                "unmanaged example. Refusing to "
                "modify ambiguous remote data."
            )

        if case_id in remote_by_case_id:
            raise ValueError(
                f"LangSmith dataset contains duplicate case_id {case_id!r}."
            )

        remote_by_case_id[case_id] = example

    desired_examples = {
        case.case_id: (
            _build_langsmith_example(
                case=case,
                dataset_version=(dataset.version),
                dataset_sha256=(dataset_sha256),
            )
        )
        for case in dataset.cases
    }

    created_examples = 0
    updated_examples = 0
    deleted_examples = 0
    unchanged_examples = 0

    examples_to_create: list[dict[str, Any]] = []

    for (
        case_id,
        desired,
    ) in desired_examples.items():
        remote = remote_by_case_id.get(case_id)

        if remote is None:
            examples_to_create.append(desired)
            created_examples += 1
            continue

        if _remote_example_matches(
            remote=remote,
            desired=desired,
        ):
            unchanged_examples += 1
            continue

        client.update_example(
            example_id=remote.id,
            inputs=desired["inputs"],
            outputs=desired["outputs"],
            metadata=desired["metadata"],
        )

        updated_examples += 1

    if examples_to_create:
        client.create_examples(
            dataset_id=remote_dataset.id,
            examples=examples_to_create,
        )

    stale_case_ids = set(remote_by_case_id) - set(desired_examples)

    for case_id in sorted(stale_case_ids):
        client.delete_example(example_id=(remote_by_case_id[case_id].id))

        deleted_examples += 1

    return RequirementExtractionLangSmithDatasetSyncResult(
        dataset_name=(dataset.dataset_name),
        dataset_id=str(remote_dataset.id),
        dataset_created=(dataset_created),
        created_examples=(created_examples),
        updated_examples=(updated_examples),
        deleted_examples=(deleted_examples),
        unchanged_examples=(unchanged_examples),
        total_examples=len(dataset.cases),
    )


def build_requirement_extraction_langsmith_experiment_contract(
    *,
    identity: RequirementExtractionBenchmarkIdentity,
) -> RequirementExtractionLangSmithExperimentContract:
    """Build stable LangSmith experiment naming and provenance."""

    model_slug = _slug(identity.model_name)

    prompt_slug = _slug(identity.prompt_version)

    return RequirementExtractionLangSmithExperimentContract(
        dataset_name=(identity.dataset_name),
        experiment_prefix=(
            f"careerops-requirement-extraction-{model_slug}-{prompt_slug}"
        ),
        description=(
            "CareerOps requirement-extraction "
            "benchmark evaluated against the "
            "Git-managed deterministic reference "
            "dataset."
        ),
        metadata={
            "benchmark": ("requirement-extraction"),
            "dataset_version": (identity.dataset_version),
            "dataset_sha256": (identity.dataset_sha256),
            "provider": (identity.provider),
            "model_name": (identity.model_name),
            "prompt_version": (identity.prompt_version),
            "temperature": (identity.temperature),
            "models": [(f"{identity.provider}:{identity.model_name}")],
        },
    )


def _build_langsmith_example(
    *,
    case: RequirementExtractionEvalCase,
    dataset_version: str,
    dataset_sha256: str,
) -> dict[str, Any]:
    """Convert one local gold case into one LangSmith example."""

    return {
        "inputs": {
            "job_description": (case.job_description),
        },
        "outputs": {
            "reference_case": (case.model_dump(mode="json")),
        },
        "metadata": {
            "managed_by": (LANGSMITH_MANAGED_BY),
            "case_id": case.case_id,
            "purpose": case.purpose,
            "tags": case.tags,
            "dataset_version": (dataset_version),
            "dataset_sha256": (dataset_sha256),
        },
    }


def _remote_example_matches(
    *,
    remote: Any,
    desired: dict[
        str,
        Any,
    ],
) -> bool:
    """Return whether one remote example matches the Git-managed state."""

    remote_inputs = dict(remote.inputs or {})

    remote_outputs = dict(remote.outputs or {})

    remote_metadata = dict(remote.metadata or {})

    desired_metadata = desired["metadata"]

    managed_metadata_matches = all(
        remote_metadata.get(key) == value for key, value in desired_metadata.items()
    )

    return bool(
        remote_inputs == desired["inputs"]
        and remote_outputs == desired["outputs"]
        and managed_metadata_matches
    )


def _slug(
    value: str,
) -> str:
    """Convert one provenance label into a readable experiment slug."""

    slug = re.sub(
        r"[^a-zA-Z0-9._-]+",
        "-",
        value.strip(),
    )

    return slug.strip("-").lower()


def build_requirement_extraction_langsmith_target(
    *,
    extractor: RequirementExtractor,
) -> LangSmithTarget:
    """Build the real CareerOps target used by LangSmith experiments."""

    def target(
        inputs: dict[
            str,
            Any,
        ],
    ) -> dict[
        str,
        Any,
    ]:
        job_description = inputs.get("job_description")

        if not isinstance(
            job_description,
            str,
        ):
            raise ValueError(
                "LangSmith evaluation input must contain a string job_description."
            )

        normalized_description = job_description.strip()

        if not normalized_description:
            raise ValueError("LangSmith evaluation job_description cannot be blank.")

        job_fingerprint = (
            sha256(normalized_description.encode("utf-8")).hexdigest()[:12].upper()
        )

        extraction = extractor.extract(
            normalized_description,
            job_id=(f"JOB-LS-EVAL-{job_fingerprint}"),
        )

        return {"extraction": (extraction.model_dump(mode="json"))}

    return target


def evaluate_requirement_extraction_langsmith(
    run: Run,
    example: Example | None,
) -> EvaluationResults:
    """Score one LangSmith example with CareerOps deterministic rules."""

    if example is None:
        raise ValueError("LangSmith evaluator requires a dataset example.")

    if run.outputs is None:
        raise ValueError("LangSmith run must contain outputs.")

    if example.outputs is None:
        raise ValueError("LangSmith example must contain reference outputs.")

    reference_case = example.outputs.get("reference_case")

    if not isinstance(
        reference_case,
        dict,
    ):
        raise ValueError("LangSmith reference output must contain reference_case.")

    extraction_payload = run.outputs.get("extraction")

    if not isinstance(
        extraction_payload,
        dict,
    ):
        raise ValueError("LangSmith target output must contain extraction.")

    case = RequirementExtractionEvalCase.model_validate(reference_case)

    extraction = JobRequirementExtraction.model_validate(extraction_payload)

    evaluation = evaluate_requirement_extraction(
        case=case,
        actual=(to_requirement_extraction_evaluation_output(extraction)),
    )

    expectation_accuracy = sum(
        1
        for result in evaluation.expectation_results
        if (
            result.matched
            and result.category_correct
            and result.match_terms_present
            and result.source_terms_present
        )
    ) / len(evaluation.expectation_results)

    return EvaluationResults(
        results=[
            EvaluationResult(
                key="careerops_pass",
                score=evaluation.passed,
            ),
            EvaluationResult(
                key="careerops_score",
                score=evaluation.score,
            ),
            EvaluationResult(
                key="role_title_correct",
                score=(evaluation.role_title_correct),
            ),
            EvaluationResult(
                key="requirement_count_correct",
                score=(evaluation.requirement_count_correct),
            ),
            EvaluationResult(
                key="forbidden_terms_absent",
                score=(evaluation.forbidden_terms_absent),
            ),
            EvaluationResult(
                key="source_grounding_correct",
                score=(evaluation.source_grounding_correct),
            ),
            EvaluationResult(
                key="expectation_accuracy",
                score=expectation_accuracy,
            ),
        ]
    )
