"""
FastAPI application exposing the Deep Research Agent over a streaming SSE API.

Endpoints
---------
GET  /health                              liveness check
POST /api/research/stream                 start a new research run (SSE)
POST /api/research/{thread_id}/resume     resume a run paused at human_review (SSE)
GET  /api/research/{thread_id}/state      fetch the current state snapshot for a run
"""
from __future__ import annotations

import json
import logging
import uuid
from typing import Any, AsyncIterator, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langgraph.types import Command
from pydantic import BaseModel, Field

from app.graph.state import new_initial_state
from app.graph.workflow import research_graph
from app.utils.telemetry import init_telemetry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("deep_research.api")

init_telemetry()

app = FastAPI(
    title="Deep Research & Report Generation Engine",
    description="Autonomous multi-agent research API built on LangGraph.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class StartResearchRequest(BaseModel):
    topic: str = Field(..., min_length=3, max_length=500)
    max_iterations: int = Field(default=3, ge=1, le=5)
    results_per_query: int = Field(default=5, ge=1, le=10)
    enable_human_review: bool = Field(default=False)


class ResumeResearchRequest(BaseModel):
    approved: bool
    feedback: Optional[str] = Field(default=None, max_length=1000)


def _sse_event(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


def _serialize_log_entries(agent_log: list) -> list[dict]:
    return [entry.model_dump() if hasattr(entry, "model_dump") else entry for entry in agent_log]


def _serialize(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, list):
        return [_serialize(v) for v in value]
    return value


async def _stream_graph(input_data: Any, config: dict) -> AsyncIterator[str]:
    """Drive the graph with `input_data`, translating each update into an SSE frame.

    Yields `node_update` events as each node finishes, an `interrupt` event
    (and stops) if the graph pauses for human review, a `done` event with
    final token/cost totals on successful completion, or an `error` event if
    the run raises.
    """
    thread_id = config["configurable"]["thread_id"]
    try:
        async for chunk in research_graph.astream(input_data, config=config, stream_mode="updates"):
            if "__interrupt__" in chunk:
                interrupt_obj = chunk["__interrupt__"][0]
                payload = getattr(interrupt_obj, "value", interrupt_obj)
                yield _sse_event("interrupt", {"thread_id": thread_id, "payload": payload})
                return

            for node_name, node_output in chunk.items():
                if not isinstance(node_output, dict):
                    continue
                event_data: dict[str, Any] = {"thread_id": thread_id, "node": node_name}
                if "agent_log" in node_output:
                    event_data["log"] = _serialize_log_entries(node_output["agent_log"])
                if node_name == "writer" and node_output.get("report"):
                    event_data["report"] = node_output["report"]
                    event_data["citations"] = _serialize(node_output.get("citations", []))
                yield _sse_event("node_update", event_data)

        snapshot = await research_graph.aget_state(config)
        if not snapshot.next:
            final_usage = snapshot.values.get("token_usage")
            yield _sse_event(
                "done",
                {"thread_id": thread_id, "token_usage": _serialize(final_usage)},
            )
    except Exception as exc:  # noqa: BLE001 - surfaced to the client as an SSE error event
        logger.exception("Research stream failed for thread %s", thread_id)
        yield _sse_event("error", {"thread_id": thread_id, "message": str(exc)})


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/research/stream")
async def start_research(request: StartResearchRequest) -> StreamingResponse:
    """Kick off a new research run and stream its progress as Server-Sent Events."""
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = new_initial_state(
        topic=request.topic,
        max_iterations=request.max_iterations,
        results_per_query=request.results_per_query,
        enable_human_review=request.enable_human_review,
        thread_id=thread_id,
    )
    return StreamingResponse(
        _stream_graph(initial_state, config),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/research/{thread_id}/resume")
async def resume_research(thread_id: str, request: ResumeResearchRequest) -> StreamingResponse:
    """Resume a run that is paused at the human_review interrupt."""
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = await research_graph.aget_state(config)
    if not snapshot.next:
        raise HTTPException(status_code=404, detail="No paused research run found for this thread_id.")

    resume_payload = Command(resume={"approved": request.approved, "feedback": request.feedback})
    return StreamingResponse(
        _stream_graph(resume_payload, config),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/research/{thread_id}/state")
async def get_research_state(thread_id: str) -> dict[str, Any]:
    """Fetch a point-in-time snapshot of a run's state, e.g. for debugging or polling."""
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = await research_graph.aget_state(config)
    if not snapshot.values:
        raise HTTPException(status_code=404, detail="Unknown thread_id.")

    values = {key: _serialize(value) for key, value in snapshot.values.items()}
    return {"thread_id": thread_id, "state": values, "awaiting_input": bool(snapshot.next)}
