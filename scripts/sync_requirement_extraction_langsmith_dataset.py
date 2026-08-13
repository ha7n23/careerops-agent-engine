"""Synchronize the CareerOps requirement benchmark to LangSmith."""

from pathlib import Path

from langsmith import Client

from careerops_agent_engine.agents.prompts.job_requirements import (
    PROMPT_VERSION,
)
from careerops_agent_engine.core.config import (
    get_settings,
)
from careerops_agent_engine.evaluation.benchmark import (
    build_requirement_extraction_benchmark_identity,
)
from careerops_agent_engine.evaluation.datasets import (
    load_requirement_extraction_dataset,
)
from careerops_agent_engine.evaluation.langsmith import (
    build_requirement_extraction_langsmith_experiment_contract,
    sync_requirement_extraction_langsmith_dataset,
)

DATASET_PATH = Path("evals/requirement_extraction/v1.json")


def main() -> None:
    """Synchronize Git gold data and print future experiment provenance."""

    settings = get_settings()

    dataset = load_requirement_extraction_dataset(DATASET_PATH)

    client = Client()

    sync_result = sync_requirement_extraction_langsmith_dataset(
        client=client,
        dataset=dataset,
        dataset_path=DATASET_PATH,
    )

    identity = build_requirement_extraction_benchmark_identity(
        dataset=dataset,
        dataset_path=DATASET_PATH,
        provider="google",
        model_name=settings.llm_model,
        prompt_version=PROMPT_VERSION,
        temperature=(settings.llm_temperature),
    )

    contract = build_requirement_extraction_langsmith_experiment_contract(
        identity=identity,
    )

    print("=== CareerOps LangSmith dataset sync ===")
    print(f"dataset: {sync_result.dataset_name}")
    print(f"dataset_id: {sync_result.dataset_id}")
    print(f"dataset_created: {sync_result.dataset_created}")
    print(f"created: {sync_result.created_examples}")
    print(f"updated: {sync_result.updated_examples}")
    print(f"deleted: {sync_result.deleted_examples}")
    print(f"unchanged: {sync_result.unchanged_examples}")
    print(f"total: {sync_result.total_examples}")

    print()
    print("=== Future experiment contract ===")
    print(f"experiment_prefix: {contract.experiment_prefix}")
    print(f"model: {identity.model_name}")
    print(f"prompt_version: {identity.prompt_version}")
    print(f"dataset_version: {identity.dataset_version}")
    print(f"dataset_sha256: {identity.dataset_sha256}")

    print()
    print("LANGSMITH DATASET SYNC PASSED")


if __name__ == "__main__":
    main()
