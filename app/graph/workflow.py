"""
LangGraph state machine construction for the Deep Research Agent.

Graph shape:

    planner -> search -> critic --(insufficient & budget left)--> search
                            |
                            +--(sufficient, or budget exhausted)--> human_review*
                                                                          |
                                                            +--(approved)--> writer -> END
                                                            |
                                                            +--(rejected)--> planner

    * human_review is skipped straight to `writer` when `enable_human_review`
      is False in the run configuration (see route_after_critic).

A checkpointer is required for the `human_review` interrupt/resume pattern to
work: `MemorySaver` is used here for simplicity. For a multi-worker production
deployment, swap it for a persistent checkpointer (e.g. Postgres or Redis) so
that a paused run can be resumed by a different process than the one that
started it.
"""
from __future__ import annotations

from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from app.graph.nodes import (
    critic_node,
    human_review_node,
    planner_node,
    route_after_critic,
    route_after_human,
    search_node,
    writer_node,
)
from app.graph.state import ResearchState


def build_graph() -> Any:
    """Construct and compile the research StateGraph with an in-memory checkpointer."""
    graph = StateGraph(ResearchState)

    graph.add_node("planner", planner_node)
    graph.add_node("search", search_node)
    graph.add_node("critic", critic_node)
    graph.add_node("human_review", human_review_node)
    graph.add_node("writer", writer_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "search")
    graph.add_edge("search", "critic")
    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {"search": "search", "human_review": "human_review", "writer": "writer"},
    )
    graph.add_conditional_edges(
        "human_review",
        route_after_human,
        {"writer": "writer", "planner": "planner"},
    )
    graph.add_edge("writer", END)

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


# Module-level singleton: the graph structure is static, so it's built once
# at import time and reused (thread-safely) across concurrent API requests.
research_graph = build_graph()
