"""Serialize CareerOps evaluation artifacts safely."""

from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel

from careerops_agent_engine.evaluation.models import (
    RequirementExtractionBenchmarkResult,
    RequirementExtractionEvaluationReport,
)


def write_evaluation_model_json_atomic(
    *,
    model: BaseModel,
    path: Path,
) -> None:
    """Atomically replace one JSON evaluation artifact."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")

    try:
        temporary_path.write_text(
            model.model_dump_json(
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        temporary_path.replace(path)

    finally:
        temporary_path.unlink(
            missing_ok=True,
        )


def write_requirement_extraction_report(
    *,
    report: RequirementExtractionEvaluationReport,
    path: Path,
) -> None:
    """Write one deterministic evaluation report as formatted JSON."""

    write_evaluation_model_json_atomic(
        model=report,
        path=path,
    )


def write_requirement_extraction_benchmark_result(
    *,
    result: RequirementExtractionBenchmarkResult,
    path: Path,
) -> None:
    """Write one completed real-model benchmark result."""

    write_evaluation_model_json_atomic(
        model=result,
        path=path,
    )
