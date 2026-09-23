"""Factories for model-backed application adapters."""

from langchain_core.rate_limiters import (
    BaseRateLimiter,
)

from careerops_agent_engine.application.ports.cv_claim_verifier import (
    CVClaimVerifier,
)
from careerops_agent_engine.application.ports.cv_evidence_extractor import (
    CVEvidenceExtractor,
)
from careerops_agent_engine.application.ports.cv_proposal_generator import (
    CVProposalGenerator,
)
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
from careerops_agent_engine.infrastructure.llm.cv_claim_verifier import (
    LangChainCVClaimVerifier,
)
from careerops_agent_engine.infrastructure.llm.cv_evidence_extractor import (
    LangChainCVEvidenceExtractor,
)
from careerops_agent_engine.infrastructure.llm.cv_proposal_generator import (
    LangChainCVProposalGenerator,
)
from careerops_agent_engine.infrastructure.llm.evidence_discovery_agent import (
    LangChainEvidenceDiscoveryAgent,
)
from careerops_agent_engine.infrastructure.llm.model_factory import (
    ChatModelProfile,
    create_chat_model,
    resolve_model_name,
)
from careerops_agent_engine.infrastructure.llm.rate_limiting import (
    get_shared_llm_rate_limiter,
)
from careerops_agent_engine.infrastructure.llm.requirement_extractor import (
    LangChainRequirementExtractor,
)


def _get_rate_limiter(
    settings: Settings,
    profile: ChatModelProfile,
) -> BaseRateLimiter:
    """Return the shared limiter for one provider and model."""

    return get_shared_llm_rate_limiter(
        provider=settings.llm_provider,
        model_name=resolve_model_name(
            settings,
            profile,
        ),
        requests_per_minute=settings.llm_requests_per_minute,
    )


def create_requirement_extractor(
    settings: Settings | None = None,
) -> RequirementExtractor:
    """Create the configured requirement-extraction adapter."""

    resolved_settings = settings or get_settings()
    profile = ChatModelProfile.FAST_STRUCTURED_OUTPUT
    rate_limiter = _get_rate_limiter(
        resolved_settings,
        profile,
    )
    model = create_chat_model(
        settings=resolved_settings,
        profile=profile,
        rate_limiter=rate_limiter,
    )

    return LangChainRequirementExtractor(
        model=model,
        model_name=resolve_model_name(
            resolved_settings,
            profile,
        ),
    )


def create_evidence_discovery_runner(
    repository: EvidenceRepository,
    settings: Settings | None = None,
) -> EvidenceDiscoveryRunner:
    """Create the configured approved-evidence discovery agent."""

    resolved_settings = settings or get_settings()
    profile = ChatModelProfile.TOOL_CALLING
    rate_limiter = _get_rate_limiter(
        resolved_settings,
        profile,
    )
    model = create_chat_model(
        settings=resolved_settings,
        profile=profile,
        rate_limiter=rate_limiter,
    )

    return LangChainEvidenceDiscoveryAgent(
        repository=repository,
        settings=resolved_settings,
        model=model,
        model_name=resolve_model_name(
            resolved_settings,
            profile,
        ),
    )


def create_cv_proposal_generator(
    settings: Settings | None = None,
) -> CVProposalGenerator:
    """Create the configured CV proposal generator."""

    resolved_settings = settings or get_settings()
    profile = ChatModelProfile.QUALITY_STRUCTURED_OUTPUT
    rate_limiter = _get_rate_limiter(
        resolved_settings,
        profile,
    )
    model = create_chat_model(
        settings=resolved_settings,
        profile=profile,
        rate_limiter=rate_limiter,
    )

    return LangChainCVProposalGenerator(
        model=model,
        model_name=resolve_model_name(
            resolved_settings,
            profile,
        ),
    )


def create_cv_claim_verifier(
    settings: Settings | None = None,
) -> CVClaimVerifier:
    """Create the configured CV claim-verification adapter."""

    resolved_settings = settings or get_settings()
    profile = ChatModelProfile.QUALITY_STRUCTURED_OUTPUT
    rate_limiter = _get_rate_limiter(
        resolved_settings,
        profile,
    )
    model = create_chat_model(
        settings=resolved_settings,
        profile=profile,
        rate_limiter=rate_limiter,
    )

    return LangChainCVClaimVerifier(
        model=model,
        model_name=resolve_model_name(
            resolved_settings,
            profile,
        ),
    )


def create_cv_evidence_extractor(
    settings: Settings | None = None,
) -> CVEvidenceExtractor:
    """Create the configured CV evidence extractor."""

    resolved_settings = settings or get_settings()
    profile = ChatModelProfile.FAST_STRUCTURED_OUTPUT
    rate_limiter = _get_rate_limiter(
        resolved_settings,
        profile,
    )
    model = create_chat_model(
        settings=resolved_settings,
        profile=profile,
        rate_limiter=rate_limiter,
    )

    return LangChainCVEvidenceExtractor(
        model=model,
        model_name=resolve_model_name(
            resolved_settings,
            profile,
        ),
    )
