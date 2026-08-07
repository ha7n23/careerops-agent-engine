"""Prove that LangGraph state survives checkpointer reconnection."""

from typing import TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from careerops_agent_engine.infrastructure.database.checkpoint import (
    open_postgres_checkpointer,
)

THREAD_ID = "CHECKPOINT-SMOKE-001"


class PersistenceSmokeState(TypedDict):
    """Minimal state used to demonstrate durable checkpointing."""

    counter: int
    message: str


def increment_counter(
    state: PersistenceSmokeState,
) -> PersistenceSmokeState:
    """Return a predictable state update."""

    return {
        "counter": state["counter"] + 1,
        "message": "Checkpoint persisted successfully.",
    }


def build_graph(checkpointer):
    """Compile the smoke graph with a supplied checkpointer."""

    builder = StateGraph(PersistenceSmokeState)

    builder.add_node("increment_counter", increment_counter)
    builder.add_edge(START, "increment_counter")
    builder.add_edge("increment_counter", END)

    return builder.compile(checkpointer=checkpointer)


def main() -> None:
    """Persist state, reconnect, and prove it can be recovered."""

    config: RunnableConfig = {
        "configurable": {
            "thread_id": THREAD_ID,
        }
    }

    # First connection: execute and persist the graph.
    with open_postgres_checkpointer() as checkpointer:
        checkpointer.setup()

        # Keep repeated smoke-test runs deterministic.
        checkpointer.delete_thread(THREAD_ID)

        graph = build_graph(checkpointer)

        result = graph.invoke(
            {
                "counter": 41,
                "message": "Before checkpoint.",
            },
            config=config,
        )

        print(
            "Initial graph result:",
            result,
        )

    # The first database connection is now closed.

    # Second connection: reconstruct the graph and recover the same thread.
    with open_postgres_checkpointer() as checkpointer:
        graph = build_graph(checkpointer)

        snapshot = graph.get_state(config)

        print(
            "Recovered state:",
            snapshot.values,
        )

        assert snapshot.values["counter"] == 42
        assert snapshot.values["message"] == ("Checkpoint persisted successfully.")

        # Remove only this synthetic smoke-test thread.
        checkpointer.delete_thread(THREAD_ID)

    print("LangGraph PostgreSQL persistence smoke test passed.")


if __name__ == "__main__":
    main()
