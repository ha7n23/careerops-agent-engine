"""Tests for evidence-agent trajectory inspection."""

from langchain_core.messages import AIMessage, ToolMessage

from careerops_agent_engine.infrastructure.llm.agent_trajectory import (
    inspect_evidence_agent_trajectory,
)


def test_trajectory_collects_tool_calls_and_evidence_ids() -> None:
    """Tool names and observed evidence IDs should be recorded."""

    messages = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_approved_evidence",
                    "args": {"query": "LangGraph"},
                    "id": "CALL-001",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            name="search_approved_evidence",
            tool_call_id="CALL-001",
            content=(
                '{"query":"LangGraph","records":['
                '{"evidence_id":"EVD-001","category":"project",'
                '"title":"CareerOps","technologies":["LangGraph"],'
                '"capabilities":[],"approved_claims":[]}]}'
            ),
        ),
    ]

    trajectory = inspect_evidence_agent_trajectory(messages)

    assert trajectory.called_tools == ("search_approved_evidence",)
    assert trajectory.observed_evidence_ids == frozenset({"EVD-001"})


def test_invalid_tool_json_is_ignored_safely() -> None:
    """Malformed tool output should not create false provenance."""

    messages = [
        ToolMessage(
            name="search_approved_evidence",
            tool_call_id="CALL-002",
            content="not-json",
        )
    ]

    trajectory = inspect_evidence_agent_trajectory(messages)

    assert trajectory.called_tools == ()
    assert trajectory.observed_evidence_ids == frozenset()
