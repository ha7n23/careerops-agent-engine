"""Shared request-rate limiting for CareerOps model providers."""

from functools import lru_cache

from langchain_core.rate_limiters import (
    BaseRateLimiter,
    InMemoryRateLimiter,
)


@lru_cache(maxsize=32)
def get_shared_llm_rate_limiter(
    *,
    provider: str,
    model_name: str,
    requests_per_minute: float,
) -> BaseRateLimiter:
    """Return one process-local limiter per provider and model."""

    if not provider.strip():
        raise ValueError("LLM provider must not be empty.")

    if not model_name.strip():
        raise ValueError("LLM model name must not be empty.")

    if requests_per_minute <= 0:
        raise ValueError("LLM requests per minute must be positive.")

    return InMemoryRateLimiter(
        requests_per_second=(requests_per_minute / 60.0),
        check_every_n_seconds=0.1,
        max_bucket_size=1,
    )
