"""Nodes for the initial deterministic job-analysis graph."""

from careerops_agent_engine.agents.states.job_analysis import (
    JobAnalysisState,
    JobAnalysisUpdate,
)
from careerops_agent_engine.application.services.fit_scoring import (
    calculate_weighted_fit,
)
from careerops_agent_engine.domain.enums import RequirementCategory
from careerops_agent_engine.domain.models.job import JobRequirement

MINIMUM_JOB_DESCRIPTION_LENGTH = 20


def validate_job_input(
    state: JobAnalysisState,
) -> JobAnalysisUpdate:
    """Validate and normalise the supplied job description."""

    normalised_description = state["job_description"].strip()

    if len(normalised_description) < MINIMUM_JOB_DESCRIPTION_LENGTH:
        return {
            "job_description": normalised_description,
            "status": "invalid",
            "validation_error": (
                "Job description must contain at least "
                f"{MINIMUM_JOB_DESCRIPTION_LENGTH} characters."
            ),
            "audit_events": [
                {
                    "node": "validate_job_input",
                    "event": "job_input_invalid",
                }
            ],
        }

    return {
        "job_description": normalised_description,
        "status": "validated",
        "validation_error": None,
        "audit_events": [
            {
                "node": "validate_job_input",
                "event": "job_input_validated",
            }
        ],
    }


def create_mock_requirements(
    state: JobAnalysisState,
) -> JobAnalysisUpdate:
    """Create temporary requirements before LLM extraction is introduced."""

    del state

    requirements = [
        JobRequirement(
            requirement_id="REQ-PYTHON",
            name="Python",
            category=RequirementCategory.ESSENTIAL,
            evidence_expected=("Practical Python software-engineering experience."),
            importance_score=5,
            source_text="Strong Python development experience is required.",
        ),
        JobRequirement(
            requirement_id="REQ-LANGGRAPH",
            name="LangGraph",
            category=RequirementCategory.ESSENTIAL,
            evidence_expected=("Implementation of stateful agent workflows."),
            importance_score=4,
            source_text="Experience with LangGraph is required.",
        ),
    ]

    return {
        "requirements": [
            requirement.model_dump(mode="json") for requirement in requirements
        ],
        "audit_events": [
            {
                "node": "create_mock_requirements",
                "event": "mock_requirements_created",
            }
        ],
    }


def calculate_mock_fit(
    state: JobAnalysisState,
) -> JobAnalysisUpdate:
    """Calculate a deterministic score from the mock requirements."""

    requirement_payloads = state.get("requirements", [])
    requirements = [
        JobRequirement.model_validate(payload) for payload in requirement_payloads
    ]

    fit_score = calculate_weighted_fit(
        requirements=requirements,
        matched_requirement_ids=state["matched_requirement_ids"],
    )

    return {
        "fit_score": fit_score,
        "audit_events": [
            {
                "node": "calculate_mock_fit",
                "event": "fit_score_calculated",
            }
        ],
    }


def complete_analysis(
    state: JobAnalysisState,
) -> JobAnalysisUpdate:
    """Mark a valid deterministic analysis as complete."""

    del state

    return {
        "status": "completed",
        "audit_events": [
            {
                "node": "complete_analysis",
                "event": "job_analysis_completed",
            }
        ],
    }


def mark_invalid(
    state: JobAnalysisState,
) -> JobAnalysisUpdate:
    """Finish an invalid workflow without running later analysis nodes."""

    del state

    return {
        "status": "invalid",
        "audit_events": [
            {
                "node": "mark_invalid",
                "event": "job_analysis_rejected",
            }
        ],
    }
