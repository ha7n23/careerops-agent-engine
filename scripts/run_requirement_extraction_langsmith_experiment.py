"""Run the CareerOps requirement-extraction LangSmith experiment."""

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
    build_requirement_extraction_langsmith_target,
    evaluate_requirement_extraction_langsmith,
)
from careerops_agent_engine.infrastructure.llm.factory import (
    create_requirement_extractor,
)

DATASET_PATH = Path("evals/requirement_extraction/v1.json")


def main() -> None:
    """Run one sequential real-model LangSmith experiment."""

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

    contract = build_requirement_extraction_langsmith_experiment_contract(
        identity=identity
    )

    extractor = create_requirement_extractor(settings)

    target = build_requirement_extraction_langsmith_target(extractor=extractor)

    client = Client()

    print("=== CareerOps LangSmith experiment ===")
    print(f"dataset: {contract.dataset_name}")
    print(f"experiment_prefix: {contract.experiment_prefix}")
    print(f"model: {identity.model_name}")
    print(f"prompt: {identity.prompt_version}")
    print(f"rate limit: {settings.llm_requests_per_minute:g} requests/minute")
    print("max_concurrency: 1")
    print()

    client.evaluate(
        target,
        data=contract.dataset_name,
        evaluators=[evaluate_requirement_extraction_langsmith],
        experiment_prefix=(contract.experiment_prefix),
        description=(contract.description),
        metadata=(contract.metadata),
        max_concurrency=1,
        num_repetitions=1,
    )

    print()
    print("LANGSMITH REQUIREMENT EXPERIMENT COMPLETED")


if __name__ == "__main__":
    main()
