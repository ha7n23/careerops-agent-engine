"""Tests for the evidence-discovery tool definitions."""

from langchain_core.tools import BaseTool

from careerops_agent_engine.agents.tools.evidence import (
    create_evidence_tools,
)
from careerops_agent_engine.infrastructure.repositories.in_memory_evidence import (
    InMemoryEvidenceRepository,
)


def build_tools() -> dict[str, BaseTool]:
    """Create evidence tools backed by an empty repository."""

    tools = create_evidence_tools(InMemoryEvidenceRepository())

    return {tool.name: tool for tool in tools}


def test_agent_receives_only_narrow_read_tools() -> None:
    """The evidence agent must not receive write or delete tools."""

    tools = build_tools()

    assert set(tools) == {
        "search_approved_evidence",
        "get_project_details",
        "list_verified_skills",
    }


def test_trusted_identity_is_not_model_tool_input() -> None:
    """User identity must be injected through runtime context."""

    tools = build_tools()

    for evidence_tool in tools.values():
        model_visible_arguments = evidence_tool.args

        assert "user_id" not in model_visible_arguments
        assert "runtime" not in model_visible_arguments
