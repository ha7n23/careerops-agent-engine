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


def test_same_rate_reuses_one_shared_limiter() -> None:
    """Adapters using the same configuration must coordinate."""

    first = get_shared_llm_rate_limiter(requests_per_minute=4.0)

    second = get_shared_llm_rate_limiter(requests_per_minute=4.0)

    assert first is second
    assert isinstance(
        first,
        InMemoryRateLimiter,
    )


def test_different_rates_use_different_limiters() -> None:
    """Distinct runtime configurations must not share a bucket."""

    first = get_shared_llm_rate_limiter(requests_per_minute=4.0)

    second = get_shared_llm_rate_limiter(requests_per_minute=10.0)

    assert first is not second


def test_non_positive_rate_is_rejected() -> None:
    """Invalid limiter configuration must fail immediately."""

    try:
        get_shared_llm_rate_limiter(requests_per_minute=0.0)
    except ValueError as exc:
        assert str(exc) == ("LLM requests per minute must be positive.")
    else:
        raise AssertionError("Expected non-positive rate to be rejected.")
