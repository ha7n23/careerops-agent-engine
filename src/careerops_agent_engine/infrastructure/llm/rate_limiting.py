"""Shared request-rate limiting for CareerOps model providers."""

from functools import lru_cache

from langchain_core.rate_limiters import (
    BaseRateLimiter,
    InMemoryRateLimiter,
)


@lru_cache(maxsize=16)
def get_shared_llm_rate_limiter(
    *,
    requests_per_minute: float,
) -> BaseRateLimiter:
    """Return one shared process-local model request limiter."""

    if requests_per_minute <= 0:
        raise ValueError("LLM requests per minute must be positive.")

    return InMemoryRateLimiter(
        requests_per_second=(requests_per_minute / 60.0),
        check_every_n_seconds=0.1,
        max_bucket_size=1,
    )
