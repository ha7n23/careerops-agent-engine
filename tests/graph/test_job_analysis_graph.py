"""Tests for the initial deterministic LangGraph workflow."""

from careerops_agent_engine.agents.graphs.job_analysis import (
    job_analysis_graph,
)


def test_valid_job_runs_complete_analysis_path() -> None:
    """Valid input should create requirements and calculate fit."""

    result = job_analysis_graph.invoke(
        {
            "job_id": "JOB-001",
            "job_description": (
                "We require strong Python development and LangGraph "
                "workflow experience."
            ),
            "matched_requirement_ids": ["REQ-PYTHON"],
            "audit_events": [],
        }
    )

    assert result["status"] == "completed"
    assert result["validation_error"] is None
    assert len(result["requirements"]) == 2
    assert result["fit_score"] == 55.56

    assert [event["event"] for event in result["audit_events"]] == [
        "job_input_validated",
        "mock_requirements_created",
        "fit_score_calculated",
        "job_analysis_completed",
    ]


def test_invalid_job_skips_analysis_nodes() -> None:
    """Invalid input should route directly to the rejection node."""

    result = job_analysis_graph.invoke(
        {
            "job_id": "JOB-002",
            "job_description": "Too short",
            "matched_requirement_ids": [],
            "audit_events": [],
        }
    )

    assert result["status"] == "invalid"
    assert result["validation_error"] is not None
    assert "requirements" not in result
    assert "fit_score" not in result

    assert [event["event"] for event in result["audit_events"]] == [
        "job_input_invalid",
        "job_analysis_rejected",
    ]
