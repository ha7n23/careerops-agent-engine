"""Application service for running job analysis."""

from typing import cast

from careerops_agent_engine.agents.graphs.job_analysis import (
    build_job_analysis_graph,
)
from careerops_agent_engine.agents.states.job_analysis import (
    JobAnalysisState,
)
from careerops_agent_engine.application.ports.evidence_discovery import (
    EvidenceDiscoveryRunner,
)
from careerops_agent_engine.application.ports.requirement_extractor import (
    RequirementExtractor,
)


class JobAnalysisService:
    """Run the CareerOps job-analysis workflow."""

    def __init__(
        self,
        *,
        requirement_extractor: RequirementExtractor,
        evidence_discovery_runner: EvidenceDiscoveryRunner,
    ) -> None:
        """Build the graph with its required external adapters."""

        self._graph = build_job_analysis_graph(
            requirement_extractor=requirement_extractor,
            evidence_discovery_runner=evidence_discovery_runner,
        )

    def analyse(
        self,
        *,
        job_id: str,
        user_id: str,
        job_description: str,
    ) -> JobAnalysisState:
        """Run one complete job-analysis workflow."""

        initial_state: JobAnalysisState = {
            "job_id": job_id,
            "user_id": user_id,
            "job_description": job_description,
            "audit_events": [],
        }

        result = self._graph.invoke(initial_state)

        return cast(JobAnalysisState, result)
