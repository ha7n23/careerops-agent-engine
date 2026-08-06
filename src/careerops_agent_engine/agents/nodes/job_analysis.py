"""Nodes for the CareerOps job-analysis graph."""

from collections.abc import Callable

from careerops_agent_engine.agents.states.job_analysis import (
    JobAnalysisState,
    JobAnalysisUpdate,
)
from careerops_agent_engine.application.ports.evidence_discovery import (
    EvidenceDiscoveryRunner,
)
from careerops_agent_engine.application.ports.requirement_extractor import (
    RequirementExtractor,
)
from careerops_agent_engine.application.services.fit_scoring import (
    calculate_evidence_weighted_fit,
)
from careerops_agent_engine.domain.models.evidence import EvidenceMatch
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


def create_discover_evidence_node(
    evidence_discovery_runner: EvidenceDiscoveryRunner,
) -> Callable[[JobAnalysisState], JobAnalysisUpdate]:
    """Create a node that discovers evidence for every requirement."""

    def discover_evidence(
        state: JobAnalysisState,
    ) -> JobAnalysisUpdate:
        requirements = [
            JobRequirement.model_validate(payload)
            for payload in state.get("requirements", [])
        ]

        matches = [
            evidence_discovery_runner.discover(
                requirement,
                user_id=state["user_id"],
            )
            for requirement in requirements
        ]

        return {
            "evidence_matches": [match.model_dump(mode="json") for match in matches],
            "audit_events": [
                {
                    "node": "discover_evidence",
                    "event": "evidence_discovery_completed",
                }
            ],
        }

    return discover_evidence


def calculate_fit(
    state: JobAnalysisState,
) -> JobAnalysisUpdate:
    """Calculate a deterministic score from evidence matches."""

    requirements = [
        JobRequirement.model_validate(payload)
        for payload in state.get("requirements", [])
    ]
    evidence_matches = [
        EvidenceMatch.model_validate(payload)
        for payload in state.get("evidence_matches", [])
    ]

    fit_score = calculate_evidence_weighted_fit(
        requirements=requirements,
        evidence_matches=evidence_matches,
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
    """Finish invalid input without running later nodes."""

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
