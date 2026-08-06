"""Factories for model-backed application adapters."""

from careerops_agent_engine.application.ports.evidence_discovery import (
    EvidenceDiscoveryRunner,
)
from careerops_agent_engine.application.ports.evidence_repository import (
    EvidenceRepository,
)
from careerops_agent_engine.application.ports.requirement_extractor import (
    RequirementExtractor,
)
from careerops_agent_engine.core.config import Settings, get_settings
from careerops_agent_engine.infrastructure.llm.evidence_discovery_agent import (
    LangChainEvidenceDiscoveryAgent,
)
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


def create_evidence_discovery_runner(
    repository: EvidenceRepository,
    settings: Settings | None = None,
) -> EvidenceDiscoveryRunner:
    """Create the configured approved-evidence discovery agent."""

    resolved_settings = settings or get_settings()

    return LangChainEvidenceDiscoveryAgent(
        repository=repository,
        settings=resolved_settings,
    )
