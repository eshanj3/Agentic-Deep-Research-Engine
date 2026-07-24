"""
Integration tests for the FastAPI application surface (health check, request
validation, and SSE streaming contract). The LangGraph workflow's internal
logic is exercised in test_graph.py; here the compiled graph's `astream` /
`aget_state` methods are stubbed so these tests run offline.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_start_research_rejects_short_topic():
    response = client.post("/api/research/stream", json={"topic": "ab"})
    assert response.status_code == 422


def test_start_research_rejects_invalid_iterations():
    response = client.post(
        "/api/research/stream", json={"topic": "A valid topic", "max_iterations": 99}
    )
    assert response.status_code == 422


def test_resume_unknown_thread_returns_404():
    fake_snapshot = SimpleNamespace(values={}, next=())

    with patch("app.main.research_graph.aget_state", new=AsyncMock(return_value=fake_snapshot)):
        response = client.post("/api/research/unknown-thread/resume", json={"approved": True})

    assert response.status_code == 404


def test_get_state_returns_404_for_unknown_thread():
    fake_snapshot = SimpleNamespace(values={}, next=())

    with patch("app.main.research_graph.aget_state", new=AsyncMock(return_value=fake_snapshot)):
        response = client.get("/api/research/unknown-thread/state")

    assert response.status_code == 404


def test_start_research_streams_node_updates():
    async def fake_astream(input_data, config=None, stream_mode=None):
        yield {
            "planner": {
                "sub_queries": ["q1", "q2"],
                "agent_log": [
                    {"node": "planner", "message": "Generated 2 search sub-queries.", "data": {}}
                ],
            }
        }

    fake_snapshot = SimpleNamespace(values={"token_usage": None}, next=())

    with patch("app.main.research_graph.astream", new=fake_astream), patch(
        "app.main.research_graph.aget_state", new=AsyncMock(return_value=fake_snapshot)
    ):
        with client.stream(
            "POST", "/api/research/stream", json={"topic": "A valid research topic"}
        ) as response:
            assert response.status_code == 200
            body = "".join(response.iter_text())

    assert "event: node_update" in body
    assert "planner" in body
    assert "event: done" in body


def test_start_research_emits_interrupt_event_and_stops():
    async def fake_astream(input_data, config=None, stream_mode=None):
        yield {"__interrupt__": (SimpleNamespace(value={"message": "please review"}),)}

    with patch("app.main.research_graph.astream", new=fake_astream):
        with client.stream(
            "POST",
            "/api/research/stream",
            json={"topic": "A valid research topic", "enable_human_review": True},
        ) as response:
            assert response.status_code == 200
            body = "".join(response.iter_text())

    assert "event: interrupt" in body
    assert "please review" in body
    assert "event: done" not in body
