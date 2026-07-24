"""
Unit tests for graph state reducers, node behavior, and routing logic.
External LLM (litellm) and search (Tavily) calls are mocked so these tests
run offline and without API keys.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.graph.nodes import (
    critic_node,
    planner_node,
    route_after_critic,
    route_after_human,
    search_node,
    writer_node,
)
from app.graph.state import CritiqueResult, SearchResult, new_initial_state


def _fake_llm_response(payload: dict) -> SimpleNamespace:
    message = SimpleNamespace(content=json.dumps(payload))
    choice = SimpleNamespace(message=message)
    usage = SimpleNamespace(prompt_tokens=100, completion_tokens=50)
    return SimpleNamespace(choices=[choice], usage=usage)


# --------------------------------------------------------------------------------------
# Planner
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_planner_node_generates_sub_queries():
    state = new_initial_state(topic="Quantum computing in drug discovery")
    fake_response = _fake_llm_response({"sub_queries": ["query one", "query two", "query three"]})

    with patch("app.graph.nodes.litellm.acompletion", new=AsyncMock(return_value=fake_response)):
        update = await planner_node(state)

    assert update["sub_queries"] == ["query one", "query two", "query three"]
    assert len(update["agent_log"]) == 1
    assert update["agent_log"][0].node == "planner"
    assert update["token_usage"].total_tokens == 150


@pytest.mark.asyncio
async def test_planner_node_falls_back_to_topic_when_llm_returns_no_queries():
    state = new_initial_state(topic="Fusion energy")
    fake_response = _fake_llm_response({"sub_queries": []})

    with patch("app.graph.nodes.litellm.acompletion", new=AsyncMock(return_value=fake_response)):
        update = await planner_node(state)

    assert update["sub_queries"] == ["Fusion energy"]


@pytest.mark.asyncio
async def test_planner_node_caps_at_five_sub_queries():
    state = new_initial_state(topic="Broad topic")
    fake_response = _fake_llm_response({"sub_queries": [f"q{i}" for i in range(8)]})

    with patch("app.graph.nodes.litellm.acompletion", new=AsyncMock(return_value=fake_response)):
        update = await planner_node(state)

    assert len(update["sub_queries"]) == 5


# --------------------------------------------------------------------------------------
# Search
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_node_deduplicates_against_seen_urls():
    state = new_initial_state(topic="Topic")
    state["sub_queries"] = ["q1"]
    state["seen_urls"] = ["https://already-seen.example.com"]

    fake_results = [
        SearchResult(query="q1", url="https://already-seen.example.com", title="Old", content="x"),
        SearchResult(query="q1", url="https://new.example.com", title="New", content="y"),
    ]
    with patch("app.graph.nodes.run_searches_concurrently", new=AsyncMock(return_value=fake_results)):
        update = await search_node(state)

    urls = [r.url for r in update["search_results"]]
    assert urls == ["https://new.example.com"]
    assert update["seen_urls"] == ["https://new.example.com"]


# --------------------------------------------------------------------------------------
# Critic
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_critic_node_parses_sufficiency_and_increments_iteration():
    state = new_initial_state(topic="Topic")
    state["iteration"] = 1
    state["search_results"] = [
        SearchResult(query="q1", url="https://a.example.com", title="A", content="evidence")
    ]
    fake_response = _fake_llm_response(
        {
            "sufficient": True,
            "reasoning": "Coverage is adequate.",
            "contradictions": [],
            "gaps": [],
            "follow_up_queries": [],
        }
    )
    with patch("app.graph.nodes.litellm.acompletion", new=AsyncMock(return_value=fake_response)):
        update = await critic_node(state)

    assert update["iteration"] == 2
    assert isinstance(update["critique"], CritiqueResult)
    assert update["critique"].sufficient is True


@pytest.mark.asyncio
async def test_critic_node_surfaces_follow_up_queries_when_insufficient():
    state = new_initial_state(topic="Topic")
    fake_response = _fake_llm_response(
        {
            "sufficient": False,
            "reasoning": "Missing recent data.",
            "contradictions": [],
            "gaps": ["no 2026 figures"],
            "follow_up_queries": ["latest 2026 figures for topic"],
        }
    )
    with patch("app.graph.nodes.litellm.acompletion", new=AsyncMock(return_value=fake_response)):
        update = await critic_node(state)

    assert update["critique"].sufficient is False
    assert update["sub_queries"] == ["latest 2026 figures for topic"]


# --------------------------------------------------------------------------------------
# Routing
# --------------------------------------------------------------------------------------


def test_route_after_critic_loops_when_insufficient_and_under_max():
    state = new_initial_state(topic="Topic", max_iterations=3)
    state["iteration"] = 1
    state["sub_queries"] = ["follow up query"]
    state["critique"] = CritiqueResult(sufficient=False, reasoning="gaps", follow_up_queries=["follow up query"])
    assert route_after_critic(state) == "search"


def test_route_after_critic_stops_at_max_iterations():
    state = new_initial_state(topic="Topic", max_iterations=2)
    state["iteration"] = 2
    state["sub_queries"] = ["follow up query"]
    state["critique"] = CritiqueResult(sufficient=False, reasoning="gaps", follow_up_queries=["follow up query"])
    assert route_after_critic(state) == "writer"


def test_route_after_critic_goes_to_human_review_when_enabled():
    state = new_initial_state(topic="Topic", max_iterations=2, enable_human_review=True)
    state["iteration"] = 2
    state["critique"] = CritiqueResult(sufficient=True, reasoning="ok")
    assert route_after_critic(state) == "human_review"


def test_route_after_critic_defaults_to_writer_without_human_review():
    state = new_initial_state(topic="Topic", max_iterations=2, enable_human_review=False)
    state["iteration"] = 2
    state["critique"] = CritiqueResult(sufficient=True, reasoning="ok")
    assert route_after_critic(state) == "writer"


def test_route_after_human_approved_goes_to_writer():
    state = new_initial_state(topic="Topic")
    state["approved"] = True
    assert route_after_human(state) == "writer"


def test_route_after_human_rejected_goes_to_planner():
    state = new_initial_state(topic="Topic")
    state["approved"] = False
    assert route_after_human(state) == "planner"


# --------------------------------------------------------------------------------------
# Writer
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_writer_node_builds_report_and_deduplicated_citations():
    state = new_initial_state(topic="Topic")
    state["search_results"] = [
        SearchResult(query="q1", url="https://a.example.com", title="A", content="first"),
        SearchResult(query="q1", url="https://a.example.com", title="A dup", content="dup"),
        SearchResult(query="q1", url="https://b.example.com", title="B", content="second"),
    ]
    fake_response = _fake_llm_response({"unused": True})
    fake_response.choices[0].message.content = "# Report\n\nSome findings [1][2].\n\n## Sources\n"

    with patch("app.graph.nodes.litellm.acompletion", new=AsyncMock(return_value=fake_response)):
        update = await writer_node(state)

    assert "Report" in update["report"]
    assert len(update["citations"]) == 2
    assert update["citations"][0].id == 1
    assert update["citations"][0].url == "https://a.example.com"
