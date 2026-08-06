"""Nodes for the initial job-analysis graph."""

from collections.abc import Callable

from careerops_agent_engine.agents.states.job_analysis import (
    JobAnalysisState,
    JobAnalysisUpdate,
)
from careerops_agent_engine.application.ports.requirement_extractor import (
    RequirementExtractor,
)
from careerops_agent_engine.application.services.fit_scoring import (
    calculate_weighted_fit,
)
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


def create_extract_requirements_node(
    extractor: RequirementExtractor,
) -> Callable[[JobAnalysisState], JobAnalysisUpdate]:
    """Create a graph node using the configured extraction adapter."""

    def extract_requirements(
        state: JobAnalysisState,
    ) -> JobAnalysisUpdate:
        extraction = extractor.extract(
            state["job_description"],
            job_id=state["job_id"],
        )

        return {
            "role_title": extraction.role_title,
            "requirements": [
                requirement.model_dump(mode="json")
                for requirement in extraction.requirements
            ],
            "audit_events": [
                {
                    "node": "extract_requirements",
                    "event": "requirements_extracted",
                }
            ],
        }

    return extract_requirements


def calculate_fit(
    state: JobAnalysisState,
) -> JobAnalysisUpdate:
    """Calculate a deterministic score from extracted requirements."""

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
                "node": "calculate_fit",
                "event": "fit_score_calculated",
            }
        ],
    }


def complete_analysis(
    state: JobAnalysisState,
) -> JobAnalysisUpdate:
    """Mark a valid analysis as complete."""

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
    """Finish an invalid workflow without running analysis nodes."""

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
