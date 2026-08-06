"""Factories for model-backed application adapters."""

from careerops_agent_engine.application.ports.requirement_extractor import (
    RequirementExtractor,
)
from careerops_agent_engine.core.config import Settings, get_settings
from careerops_agent_engine.infrastructure.llm.google_requirement_extractor import (
    GoogleRequirementExtractor,
)


def create_requirement_extractor(
    settings: Settings | None = None,
) -> RequirementExtractor:
    """Create the configured requirement-extraction adapter."""

    resolved_settings = settings or get_settings()

    return GoogleRequirementExtractor(
        model_name=resolved_settings.llm_model,
        temperature=resolved_settings.llm_temperature,
        timeout_seconds=resolved_settings.llm_timeout_seconds,
        max_retries=resolved_settings.llm_max_retries,
    )
