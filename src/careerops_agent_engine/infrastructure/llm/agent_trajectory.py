"""Utilities for inspecting evidence-agent tool trajectories."""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage


@dataclass(frozen=True, slots=True)
class EvidenceAgentTrajectory:
    """Tools called and evidence identifiers observed during one run."""

    called_tools: tuple[str, ...]
    observed_evidence_ids: frozenset[str]


def inspect_evidence_agent_trajectory(
    messages: Sequence[BaseMessage],
) -> EvidenceAgentTrajectory:
    """Inspect tool calls and collect evidence IDs from tool outputs."""

    called_tools: list[str] = []
    observed_evidence_ids: set[str] = set()

    for message in messages:
        if isinstance(message, AIMessage):
            called_tools.extend(
                str(tool_call["name"]) for tool_call in message.tool_calls
            )

        if isinstance(message, ToolMessage):
            payload = _parse_tool_payload(message.content)

            if payload is not None:
                observed_evidence_ids.update(_extract_evidence_ids(payload))

    return EvidenceAgentTrajectory(
        called_tools=tuple(called_tools),
        observed_evidence_ids=frozenset(observed_evidence_ids),
    )


def _parse_tool_payload(content: object) -> object | None:
    """Parse the JSON string produced by a CareerOps evidence tool."""

    if not isinstance(content, str):
        return None

    try:
        return cast(object, json.loads(content))
    except json.JSONDecodeError:
        return None


def _extract_evidence_ids(value: object) -> set[str]:
    """Recursively find evidence identifiers in a tool payload."""

    evidence_ids: set[str] = set()

    if isinstance(value, dict):
        for key, item in value.items():
            if key == "evidence_id" and isinstance(item, str):
                evidence_ids.add(item)

            elif key == "evidence_ids" and isinstance(item, list):
                evidence_ids.update(
                    evidence_id for evidence_id in item if isinstance(evidence_id, str)
                )

            evidence_ids.update(_extract_evidence_ids(item))

    elif isinstance(value, list):
        for item in value:
            evidence_ids.update(_extract_evidence_ids(item))

    return evidence_ids
