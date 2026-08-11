"""Tests for CareerOps job-analysis trace configuration."""

from careerops_agent_engine.application.services.job_analysis import (
    JOB_ANALYSIS_WORKFLOW_VERSION,
    build_job_analysis_config,
)


def test_new_job_analysis_has_threaded_root_trace() -> None:
    """A new graph turn should expose only safe workflow metadata."""

    config = build_job_analysis_config(
        thread_id="THR-001",
        resume=False,
    )

    assert config.get("run_name") == "careerops_job_analysis_start"

    assert config.get("configurable") == {"thread_id": "THR-001"}

    assert config.get("metadata") == {
        "component": ("job_analysis_graph"),
        "workflow": ("job-analysis"),
        "workflow_version": (JOB_ANALYSIS_WORKFLOW_VERSION),
        "thread_id": "THR-001",
    }

    assert config.get("tags") == [
        "careerops",
        "job-analysis",
        "langgraph",
        "start",
    ]


def test_review_resume_uses_same_langsmith_thread() -> None:
    """Human review should become another trace in the same thread."""

    config = build_job_analysis_config(
        thread_id="THR-001",
        resume=True,
    )

    assert config.get("run_name") == ("careerops_job_analysis_review_resume")

    metadata = config.get("metadata")

    assert metadata is not None

    assert metadata.get("thread_id") == "THR-001"

    tags = config.get("tags")

    assert tags is not None

    assert "review-resume" in tags
