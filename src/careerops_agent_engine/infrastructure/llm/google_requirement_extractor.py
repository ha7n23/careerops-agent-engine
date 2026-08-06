"""Gemini implementation of structured requirement extraction."""

from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI

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


class GoogleRequirementExtractor:
    """Extract job requirements using Gemini structured output."""

    def __init__(
        self,
        *,
        model_name: str,
        temperature: float,
        timeout_seconds: float,
        max_retries: int,
    ) -> None:
        """Initialise the provider adapter."""

        model = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            timeout=timeout_seconds,
            max_retries=max_retries,
            thinking_level="minimal",
        )

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
            config={
                "run_name": "extract_job_requirements",
                "tags": [
                    "careerops",
                    "job-analysis",
                    "structured-output",
                ],
                "metadata": {
                    "job_id": job_id,
                    "prompt_version": PROMPT_VERSION,
                    "model_name": self._model_name,
                },
            },
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
