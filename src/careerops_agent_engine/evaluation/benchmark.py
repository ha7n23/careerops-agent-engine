"""Resumable real-model CareerOps evaluation benchmarks."""

from collections.abc import Callable
from hashlib import sha256
from pathlib import Path
from typing import Literal

from careerops_agent_engine.application.ports.requirement_extractor import (
    RequirementExtractor,
)
from careerops_agent_engine.domain.models.job import (
    JobRequirementExtraction,
)
from careerops_agent_engine.evaluation.models import (
    RequirementExtractionBenchmarkIdentity,
    RequirementExtractionBenchmarkProgress,
    RequirementExtractionBenchmarkResult,
    RequirementExtractionEvalDataset,
)
from careerops_agent_engine.evaluation.reporting import (
    write_evaluation_model_json_atomic,
)
from careerops_agent_engine.evaluation.runner import (
    evaluate_requirement_extraction_dataset,
)
from careerops_agent_engine.infrastructure.llm.schemas import (
    ExtractedRequirement,
    ExtractedRequirementSet,
)

type BenchmarkCaseStatus = Literal[
    "cached",
    "started",
    "completed",
]

type BenchmarkObserver = Callable[
    [
        str,
        BenchmarkCaseStatus,
    ],
    None,
]


def build_requirement_extraction_benchmark_identity(
    *,
    dataset: RequirementExtractionEvalDataset,
    dataset_path: Path,
    provider: str,
    model_name: str,
    prompt_version: str,
    temperature: float,
) -> RequirementExtractionBenchmarkIdentity:
    """Build provenance binding cached outputs to one exact setup."""

    dataset_sha256 = sha256(dataset_path.read_bytes()).hexdigest()

    return RequirementExtractionBenchmarkIdentity(
        dataset_name=dataset.dataset_name,
        dataset_version=dataset.version,
        dataset_sha256=dataset_sha256,
        provider=provider,
        model_name=model_name,
        prompt_version=prompt_version,
        temperature=temperature,
    )


def run_requirement_extraction_benchmark(
    *,
    dataset: RequirementExtractionEvalDataset,
    extractor: RequirementExtractor,
    identity: RequirementExtractionBenchmarkIdentity,
    progress_path: Path,
    observer: BenchmarkObserver | None = None,
) -> RequirementExtractionBenchmarkResult:
    """Run or resume one real-model requirement benchmark."""

    progress = _load_or_create_progress(
        identity=identity,
        path=progress_path,
    )

    expected_case_ids = {case.case_id for case in dataset.cases}

    unexpected_cached_case_ids = set(progress.outputs) - expected_case_ids

    if unexpected_cached_case_ids:
        raise ValueError(
            "Benchmark progress contains "
            "unexpected cases: " + ", ".join(sorted(unexpected_cached_case_ids))
        )

    for case in dataset.cases:
        if case.case_id in progress.outputs:
            _notify(
                observer=observer,
                case_id=case.case_id,
                status="cached",
            )
            continue

        _notify(
            observer=observer,
            case_id=case.case_id,
            status="started",
        )

        extraction = extractor.extract(
            case.job_description,
            job_id=(f"JOB-{case.case_id}"),
        )

        progress.outputs[case.case_id] = extraction

        write_evaluation_model_json_atomic(
            model=progress,
            path=progress_path,
        )

        _notify(
            observer=observer,
            case_id=case.case_id,
            status="completed",
        )

    evaluation_outputs = {
        case_id: (_to_evaluation_output(extraction))
        for case_id, extraction in progress.outputs.items()
    }

    report = evaluate_requirement_extraction_dataset(
        dataset=dataset,
        outputs=evaluation_outputs,
    )

    return RequirementExtractionBenchmarkResult(
        identity=identity,
        outputs=progress.outputs,
        report=report,
    )


def _load_or_create_progress(
    *,
    identity: RequirementExtractionBenchmarkIdentity,
    path: Path,
) -> RequirementExtractionBenchmarkProgress:
    """Load compatible progress or create a fresh benchmark state."""

    if not path.exists():
        return RequirementExtractionBenchmarkProgress(
            identity=identity,
        )

    progress = RequirementExtractionBenchmarkProgress.model_validate_json(
        path.read_text(
            encoding="utf-8",
        )
    )

    if progress.identity != identity:
        raise ValueError(
            "Existing benchmark progress belongs "
            "to a different dataset/model/prompt "
            "configuration. Reset it before rerunning."
        )

    return progress


def _to_evaluation_output(
    extraction: JobRequirementExtraction,
) -> ExtractedRequirementSet:
    """Convert production extraction output to evaluator input."""

    return ExtractedRequirementSet(
        role_title=extraction.role_title,
        requirements=[
            ExtractedRequirement(
                name=requirement.name,
                category=requirement.category,
                evidence_expected=(requirement.evidence_expected),
                importance_score=(requirement.importance_score),
                source_text=(requirement.source_text),
            )
            for requirement in extraction.requirements
        ],
    )


def _notify(
    *,
    observer: BenchmarkObserver | None,
    case_id: str,
    status: BenchmarkCaseStatus,
) -> None:
    """Emit an optional benchmark progress event."""

    if observer is not None:
        observer(
            case_id,
            status,
        )
