"""Tests for shared CareerOps model request limiting."""

from langchain_core.rate_limiters import (
    InMemoryRateLimiter,
)

from careerops_agent_engine.infrastructure.llm.rate_limiting import (
    get_shared_llm_rate_limiter,
)


def setup_function() -> None:
    """Isolate the process-local limiter cache between tests."""

    get_shared_llm_rate_limiter.cache_clear()


def teardown_function() -> None:
    """Leave no cached limiter behind after a test."""

    get_shared_llm_rate_limiter.cache_clear()


def build_limiter(
    *,
    provider: str = "groq",
    model_name: str = "openai/gpt-oss-20b",
    requests_per_minute: float = 4.0,
) -> InMemoryRateLimiter:
    """Build one typed limiter for a test configuration."""

    limiter = get_shared_llm_rate_limiter(
        provider=provider,
        model_name=model_name,
        requests_per_minute=requests_per_minute,
    )

    assert isinstance(limiter, InMemoryRateLimiter)

    return limiter


def test_same_provider_model_and_rate_reuse_limiter() -> None:
    """Adapters using one model must coordinate through one bucket."""

    first = build_limiter()
    second = build_limiter()

    assert first is second


def test_different_models_use_different_limiters() -> None:
    """Independent model quotas must use independent local buckets."""

    fast = build_limiter(
        model_name="openai/gpt-oss-20b",
    )
    quality = build_limiter(
        model_name="openai/gpt-oss-120b",
    )

    assert fast is not quality


def test_different_providers_use_different_limiters() -> None:
    """Provider quotas must not share one local bucket."""

    google = build_limiter(
        provider="google",
        model_name="shared-name",
    )
    groq = build_limiter(
        provider="groq",
        model_name="shared-name",
    )

    assert google is not groq


def test_different_rates_use_different_limiters() -> None:
    """Distinct rate configurations must not share a bucket."""

    first = build_limiter(
        requests_per_minute=4.0,
    )
    second = build_limiter(
        requests_per_minute=10.0,
    )

    assert first is not second


def test_non_positive_rate_is_rejected() -> None:
    """Invalid limiter configuration must fail immediately."""

    try:
        get_shared_llm_rate_limiter(
            provider="groq",
            model_name="openai/gpt-oss-20b",
            requests_per_minute=0.0,
        )
    except ValueError as exc:
        assert str(exc) == ("LLM requests per minute must be positive.")
    else:
        raise AssertionError("Expected non-positive rate to be rejected.")
