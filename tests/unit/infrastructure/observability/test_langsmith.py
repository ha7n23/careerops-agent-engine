"""Tests for privacy-conscious LangSmith configuration."""

from collections.abc import Mapping
from typing import cast

import pytest

from careerops_agent_engine.infrastructure.observability.langsmith import (
    TraceMetadataValue,
    build_langsmith_run_config,
)


def test_safe_trace_configuration_is_built() -> None:
    """Approved technical metadata should remain observable."""

    config = build_langsmith_run_config(
        run_name="extract_job_requirements",
        tags=[
            "job-analysis",
            "llm",
            "job-analysis",
        ],
        metadata={
            "component": ("requirement_extractor"),
            "prompt_version": ("job-requirements-v1"),
            "ls_model_name": ("gemini-3.5-flash"),
        },
    )

    assert config.get("run_name") == "extract_job_requirements"

    assert config.get("tags") == [
        "careerops",
        "job-analysis",
        "llm",
    ]

    assert config.get("metadata") == {
        "component": ("requirement_extractor"),
        "prompt_version": ("job-requirements-v1"),
        "ls_model_name": ("gemini-3.5-flash"),
    }


@pytest.mark.parametrize(
    "unsafe_key",
    [
        "user_id",
        "job_description",
        "reviewer_feedback",
        "cv_text",
    ],
)
def test_sensitive_metadata_keys_are_rejected(
    unsafe_key: str,
) -> None:
    """Potentially identifying/content-bearing metadata must be blocked."""

    with pytest.raises(
        ValueError,
        match="Unsafe LangSmith metadata keys",
    ):
        build_langsmith_run_config(
            run_name="unsafe-run",
            tags=["test"],
            metadata={unsafe_key: "sensitive-value"},
        )


def test_complex_metadata_values_are_rejected() -> None:
    """Trace metadata must not become a hidden payload channel."""

    unsafe_metadata = cast(
        Mapping[
            str,
            TraceMetadataValue,
        ],
        {"component": {"raw": "payload"}},
    )

    with pytest.raises(
        ValueError,
        match="must be scalar",
    ):
        build_langsmith_run_config(
            run_name="unsafe-run",
            tags=["test"],
            metadata=unsafe_metadata,
        )
