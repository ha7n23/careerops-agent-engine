"""Prove durable LangGraph human review using PostgreSQL."""

from typing import Literal, TypedDict, cast

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from careerops_agent_engine.infrastructure.database.checkpoint import (
    open_postgres_checkpointer,
)

THREAD_ID = "HITL-SMOKE-001"


class HumanReviewState(TypedDict):
    """Minimal state for demonstrating durable human review."""

    proposal_id: str
    original_text: str
    proposed_text: str
    review_status: Literal[
        "pending",
        "approved",
        "rejected",
    ]
    final_text: str | None


class HumanReviewResponse(TypedDict):
    """JSON-serializable response supplied by the reviewer."""

    action: Literal["approve", "reject"]
    edited_text: str | None


def request_human_review(
    state: HumanReviewState,
) -> dict[str, object]:
    """Pause execution and wait for a human decision."""

    response = interrupt(
        {
            "type": "cv_change_review",
            "proposal_id": state["proposal_id"],
            "original_text": state["original_text"],
            "proposed_text": state["proposed_text"],
            "allowed_actions": [
                "approve",
                "reject",
            ],
        }
    )

    review = cast(HumanReviewResponse, response)

    if review["action"] == "approve":
        final_text = (
            review["edited_text"]
            if review["edited_text"] is not None
            else state["proposed_text"]
        )

        return {
            "review_status": "approved",
            "final_text": final_text,
        }

    return {
        "review_status": "rejected",
        "final_text": state["original_text"],
    }


def build_graph(checkpointer):
    """Compile the demonstration graph."""

    builder = StateGraph(HumanReviewState)

    builder.add_node(
        "request_human_review",
        request_human_review,
    )

    builder.add_edge(START, "request_human_review")
    builder.add_edge("request_human_review", END)

    return builder.compile(
        checkpointer=checkpointer,
    )


def main() -> None:
    """Pause, reconnect and resume one human-review thread."""

    config: RunnableConfig = {
        "configurable": {
            "thread_id": THREAD_ID,
        }
    }

    # ---------------------------------------------------------
    # Process A: run until human review interrupts the graph.
    # ---------------------------------------------------------
    with open_postgres_checkpointer() as checkpointer:
        checkpointer.delete_thread(THREAD_ID)

        graph = build_graph(checkpointer)

        paused_result = graph.invoke(
            {
                "proposal_id": "CVP-SMOKE-001",
                "original_text": ("Built Python applications."),
                "proposed_text": (
                    "Built production-style Python AI "
                    "applications using FastAPI and LangGraph."
                ),
                "review_status": "pending",
                "final_text": None,
            },
            config=config,
        )

        print("Paused result:")
        print(paused_result)

        interrupts = paused_result.get("__interrupt__", ())

        assert len(interrupts) == 1

        interrupt_payload = interrupts[0].value

        print()
        print("Human review payload:")
        print(interrupt_payload)

        assert interrupt_payload["type"] == "cv_change_review"
        assert interrupt_payload["proposal_id"] == "CVP-SMOKE-001"

    # The first PostgreSQL connection is now closed.
    print()
    print("First database connection closed.")

    # ---------------------------------------------------------
    # Imagine a user reviews the proposal minutes/hours later.
    # ---------------------------------------------------------
    human_response: HumanReviewResponse = {
        "action": "approve",
        "edited_text": ("Built Python AI applications using FastAPI and LangGraph."),
    }

    # ---------------------------------------------------------
    # Process B: reconnect and resume the SAME graph thread.
    # ---------------------------------------------------------
    with open_postgres_checkpointer() as checkpointer:
        graph = build_graph(checkpointer)

        resumed_result = graph.invoke(
            Command(resume=human_response),
            config=config,
        )

        print()
        print("Resumed result:")
        print(resumed_result)

        assert resumed_result["review_status"] == "approved"
        assert resumed_result["final_text"] == (
            "Built Python AI applications using FastAPI and LangGraph."
        )

        snapshot = graph.get_state(config)

        assert snapshot.values["review_status"] == "approved"

        checkpointer.delete_thread(THREAD_ID)

    print()
    print("Durable LangGraph human-review smoke test passed.")


if __name__ == "__main__":
    main()
