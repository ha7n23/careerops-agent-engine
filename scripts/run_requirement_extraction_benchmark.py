"""Run the real CareerOps requirement-extraction benchmark."""

from pathlib import Path

from careerops_agent_engine.agents.prompts.job_requirements import (
    PROMPT_VERSION,
)
from careerops_agent_engine.core.config import (
    get_settings,
)
from careerops_agent_engine.evaluation.benchmark import (
    BenchmarkCaseStatus,
    build_requirement_extraction_benchmark_identity,
    run_requirement_extraction_benchmark,
)
from careerops_agent_engine.evaluation.datasets import (
    load_requirement_extraction_dataset,
)
from careerops_agent_engine.evaluation.models import (
    RequirementExtractionBenchmarkResult,
)
from careerops_agent_engine.evaluation.reporting import (
    write_requirement_extraction_benchmark_result,
)
from careerops_agent_engine.infrastructure.llm.factory import (
    create_requirement_extractor,
)

DATASET_PATH = Path("evals/requirement_extraction/v1.json")

OUTPUT_ROOT = Path(".careerops_data/evaluations/requirement_extraction")

PROGRESS_PATH = OUTPUT_ROOT / "v1-progress.json"

RESULT_PATH = OUTPUT_ROOT / "v1-result.json"


def main() -> None:
    """Run or resume the real-model benchmark."""

    settings = get_settings()

    dataset = load_requirement_extraction_dataset(DATASET_PATH)

    identity = build_requirement_extraction_benchmark_identity(
        dataset=dataset,
        dataset_path=DATASET_PATH,
        provider="google",
        model_name=settings.llm_model,
        prompt_version=PROMPT_VERSION,
        temperature=(settings.llm_temperature),
    )

    extractor = create_requirement_extractor(settings)

    print("=== CareerOps requirement extraction benchmark ===")
    print(f"dataset: {identity.dataset_name}")
    print(f"version: {identity.dataset_version}")
    print(f"model: {identity.model_name}")
    print(f"prompt: {identity.prompt_version}")
    print(f"rate limit: {settings.llm_requests_per_minute:g} requests/minute")
    print(f"progress: {PROGRESS_PATH}")
    print()

    result = run_requirement_extraction_benchmark(
        dataset=dataset,
        extractor=extractor,
        identity=identity,
        progress_path=PROGRESS_PATH,
        observer=_print_case_status,
    )

    write_requirement_extraction_benchmark_result(
        result=result,
        path=RESULT_PATH,
    )

    _print_result(result)


def _print_case_status(
    case_id: str,
    status: BenchmarkCaseStatus,
) -> None:
    """Print one resumable benchmark event."""

    print(f"[{status.upper():9}] {case_id}")


def _print_result(
    result: RequirementExtractionBenchmarkResult,
) -> None:
    """Print benchmark-level and case-level scores."""

    report = result.report

    print()
    print("=== Deterministic evaluation ===")
    print(f"passed: {report.passed_cases}/{report.total_cases}")
    print(f"pass rate: {report.pass_rate:.1%}")
    print(f"average score: {report.average_score:.3f}")

    print()
    print("Case results:")

    for case_result in report.results:
        outcome = "PASS" if case_result.passed else "FAIL"

        print(f"  {case_result.case_id}: {outcome} score={case_result.score:.3f}")

        if case_result.passed:
            continue

        print(f"    role_title_correct={case_result.role_title_correct}")
        print(f"    requirement_count_correct={case_result.requirement_count_correct}")
        print(f"    forbidden_terms_absent={case_result.forbidden_terms_absent}")
        print(f"    source_grounding_correct={case_result.source_grounding_correct}")

        for expectation in case_result.expectation_results:
            if (
                expectation.matched
                and expectation.category_correct
                and expectation.match_terms_present
                and expectation.source_terms_present
            ):
                continue

            print(
                "    expectation "
                f"{expectation.expectation_id}: "
                f"matched={expectation.matched}, "
                "category_correct="
                f"{expectation.category_correct}, "
                "source_terms_present="
                f"{expectation.source_terms_present}"
            )

    print()
    print(f"result: {RESULT_PATH}")
    print("CAREEROPS REAL REQUIREMENT BENCHMARK COMPLETED")


if __name__ == "__main__":
    main()
