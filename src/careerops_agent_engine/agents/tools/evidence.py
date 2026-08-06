"""Read-only tools available to the evidence-discovery agent."""

from typing import Annotated

from langchain.tools import ToolRuntime, tool
from langchain_core.tools import BaseTool
from pydantic import Field

from careerops_agent_engine.agents.runtime_context import (
    EvidenceAgentContext,
)
from careerops_agent_engine.agents.tools.schemas import (
    EvidenceSearchToolResult,
    EvidenceToolRecord,
    ProjectDetailsToolResult,
    VerifiedSkill,
    VerifiedSkillsToolResult,
)
from careerops_agent_engine.application.ports.evidence_repository import (
    EvidenceRepository,
)
from careerops_agent_engine.domain.enums import EvidenceCategory


def create_evidence_tools(
    repository: EvidenceRepository,
) -> list[BaseTool]:
    """Create narrowly scoped read-only evidence tools."""

    @tool
    def search_approved_evidence(
        query: Annotated[
            str,
            Field(
                min_length=1,
                max_length=200,
                description=(
                    "Specific skill, technology, capability, or "
                    "experience to search for."
                ),
            ),
        ],
        runtime: ToolRuntime[EvidenceAgentContext],
        limit: Annotated[
            int,
            Field(
                ge=1,
                le=10,
                description=("Maximum number of evidence records to return."),
            ),
        ] = 5,
    ) -> str:
        """Search only human-approved evidence for the current user."""

        records = repository.search_approved(
            user_id=runtime.context.user_id,
            query=query,
            limit=limit,
        )

        result = EvidenceSearchToolResult(
            query=query,
            records=[EvidenceToolRecord.from_evidence(record) for record in records],
        )

        return result.model_dump_json()

    @tool
    def get_project_details(
        evidence_id: Annotated[
            str,
            Field(
                min_length=1,
                max_length=64,
                description=("Exact approved project evidence identifier."),
            ),
        ],
        runtime: ToolRuntime[EvidenceAgentContext],
    ) -> str:
        """Retrieve one approved project belonging to the current user."""

        evidence = repository.get_approved(
            user_id=runtime.context.user_id,
            evidence_id=evidence_id,
        )

        if evidence is None or evidence.category is not EvidenceCategory.PROJECT:
            result = ProjectDetailsToolResult(
                found=False,
                record=None,
                message=("No approved project evidence was found for that identifier."),
            )
            return result.model_dump_json()

        result = ProjectDetailsToolResult(
            found=True,
            record=EvidenceToolRecord.from_evidence(evidence),
            message="Approved project evidence found.",
        )

        return result.model_dump_json()

    @tool
    def list_verified_skills(
        runtime: ToolRuntime[EvidenceAgentContext],
    ) -> str:
        """List technologies supported by approved user evidence."""

        evidence_records = repository.list_approved(
            user_id=runtime.context.user_id,
            limit=100,
        )

        display_names: dict[str, str] = {}
        evidence_ids_by_skill: dict[str, set[str]] = {}

        for evidence in evidence_records:
            for technology in evidence.technologies:
                normalised_name = technology.casefold()

                display_names.setdefault(
                    normalised_name,
                    technology,
                )
                evidence_ids_by_skill.setdefault(
                    normalised_name,
                    set(),
                ).add(evidence.evidence_id)

        skills = [
            VerifiedSkill(
                name=display_names[normalised_name],
                evidence_ids=sorted(evidence_ids_by_skill[normalised_name]),
            )
            for normalised_name in sorted(display_names)
        ]

        return VerifiedSkillsToolResult(skills=skills).model_dump_json()

    return [
        search_approved_evidence,
        get_project_details,
        list_verified_skills,
    ]
