"""LangChain implementation of structured requirement extraction."""

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from careerops_agent_engine.agents.prompts.job_requirements import (
    JOB_REQUIREMENT_PROMPT,
    PROMPT_VERSION,
)
from careerops_agent_engine.domain.models.job import (
    JobRequirement,
    JobRequirementExtraction,
)
from careerops_agent_engine.infrastructure.llm.schemas import (
    ExtractedRequirementSet,
)
from careerops_agent_engine.infrastructure.observability.langsmith import (
    build_langsmith_run_config,
)


class LangChainRequirementExtractor:
    """Extract job requirements using provider-neutral structured output."""

    def __init__(
        self,
        *,
        model: BaseChatModel,
        model_name: str,
    ) -> None:
        """Initialise the structured-output adapter."""

        self._structured_model = model.with_structured_output(
            schema=ExtractedRequirementSet.model_json_schema(),
            method="json_schema",
        )
        self._model_name = model_name

    def extract(
        self,
        job_description: str,
        *,
        job_id: str,
    ) -> JobRequirementExtraction:
        """Extract and validate requirements from a job description."""

        messages = JOB_REQUIREMENT_PROMPT.format_messages(
            job_description=job_description
        )

        raw_result: Any = self._structured_model.invoke(
            messages,
            config=build_langsmith_run_config(
                run_name="extract_job_requirements",
                tags=[
                    "job-analysis",
                    "requirement-extraction",
                    "structured-output",
                    "llm",
                ],
                metadata={
                    "component": "requirement_extractor",
                    "prompt_version": PROMPT_VERSION,
                    "ls_model_name": self._model_name,
                },
            ),
        )

        extracted = ExtractedRequirementSet.model_validate(raw_result)

        requirements = [
            JobRequirement(
                requirement_id=f"REQ-{index:03d}",
                name=requirement.name,
                category=requirement.category,
                evidence_expected=requirement.evidence_expected,
                importance_score=requirement.importance_score,
                source_text=requirement.source_text,
            )
            for index, requirement in enumerate(
                extracted.requirements,
                start=1,
            )
        ]

        return JobRequirementExtraction(
            role_title=extracted.role_title,
            requirements=requirements,
        )
