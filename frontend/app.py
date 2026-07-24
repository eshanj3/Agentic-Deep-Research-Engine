"""
Streamlit UI for the Deep Research & Report Generation Engine.

Streams the live agent "thought process" (planning, searching, critiquing)
from the FastAPI backend over SSE, supports the optional human-in-the-loop
approval step, and renders the final cited Markdown report.

Run with:  streamlit run frontend/app.py
"""
from __future__ import annotations

import json
import os
from typing import Any, Iterator

import httpx
import streamlit as st

BACKEND_URL_DEFAULT = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Deep Research Agent", page_icon="🔎", layout="wide")

NODE_ICONS = {
    "planner": "🧭",
    "search": "🔍",
    "critic": "🧪",
    "human_review": "🙋",
    "writer": "📝",
}


def _init_session_state() -> None:
    defaults: dict[str, Any] = {
        "backend_url": BACKEND_URL_DEFAULT,
        "thread_id": None,
        "log_entries": [],
        "report": None,
        "citations": [],
        "token_usage": None,
        "awaiting_approval": False,
        "interrupt_payload": None,
        "running": False,
        "error": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _reset_run() -> None:
    st.session_state.thread_id = None
    st.session_state.log_entries = []
    st.session_state.report = None
    st.session_state.citations = []
    st.session_state.token_usage = None
    st.session_state.awaiting_approval = False
    st.session_state.interrupt_payload = None
    st.session_state.error = None


def _parse_sse_stream(response: httpx.Response) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield (event, data) pairs parsed from a text/event-stream response."""
    event_name = "message"
    data_lines: list[str] = []
    for line in response.iter_lines():
        if line == "":
            if data_lines:
                raw = "\n".join(data_lines)
                try:
                    yield event_name, json.loads(raw)
                except json.JSONDecodeError:
                    pass
            event_name, data_lines = "message", []
            continue
        if line.startswith("event:"):
            event_name = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:"):].strip())


def _render_log(log_container) -> None:
    with log_container.container():
        if not st.session_state.log_entries:
            st.caption("No activity yet — start a research run to see agents work in real time.")
        for entry in st.session_state.log_entries:
            node = entry.get("node", "agent")
            message = entry.get("message", "")
            icon = NODE_ICONS.get(node, "🤖")
            st.markdown(f"{icon} **{node}** — {message}")
            data = entry.get("data")
            if data:
                with st.expander(f"details ({node})", expanded=False):
                    st.json(data)


def _consume_stream(url: str, json_body: dict, log_container) -> None:
    st.session_state.running = True
    try:
        with httpx.stream("POST", url, json=json_body, timeout=None) as response:
            if response.status_code >= 400:
                response.read()
                st.session_state.error = f"Backend error {response.status_code}: {response.text}"
                return
            for event, data in _parse_sse_stream(response):
                if event == "node_update":
                    st.session_state.thread_id = data.get("thread_id", st.session_state.thread_id)
                    for entry in data.get("log", []):
                        st.session_state.log_entries.append(entry)
                    _render_log(log_container)
                    if data.get("report"):
                        st.session_state.report = data["report"]
                        st.session_state.citations = data.get("citations", [])
                elif event == "interrupt":
                    st.session_state.thread_id = data.get("thread_id", st.session_state.thread_id)
                    st.session_state.interrupt_payload = data.get("payload")
                    st.session_state.awaiting_approval = True
                elif event == "done":
                    st.session_state.token_usage = data.get("token_usage")
                elif event == "error":
                    st.session_state.error = data.get("message", "Unknown error")
    except httpx.HTTPError as exc:
        st.session_state.error = f"Connection error: {exc}"
    finally:
        st.session_state.running = False


def main() -> None:
    _init_session_state()

    st.title("🔎 Deep Research & Report Generation Engine")
    st.caption("Autonomous multi-agent research: plan → search → critique → (review) → write.")

    with st.sidebar:
        st.header("Configuration")
        st.session_state.backend_url = st.text_input("Backend URL", value=st.session_state.backend_url)
        max_iterations = st.slider("Max research iterations", min_value=1, max_value=5, value=3)
        results_per_query = st.slider("Results per query", min_value=1, max_value=10, value=5)
        enable_human_review = st.checkbox("Require human approval before writing", value=False)

        if st.session_state.token_usage:
            st.divider()
            st.subheader("Token usage & cost")
            usage = st.session_state.token_usage
            st.metric("Total tokens", usage.get("total_tokens", 0))
            st.metric("Estimated cost", f"${usage.get('estimated_cost_usd', 0):.4f}")
            st.caption(f"{usage.get('calls', 0)} LLM call(s)")

    topic = st.text_area(
        "Research topic",
        placeholder="e.g. The impact of small modular reactors on grid decarbonization",
    )

    col_start, col_reset = st.columns([1, 1])
    start_clicked = col_start.button(
        "Start Research",
        type="primary",
        disabled=st.session_state.running or not topic.strip(),
    )
    if col_reset.button("Reset"):
        _reset_run()
        st.rerun()

    st.subheader("Agent activity")
    log_container = st.empty()
    _render_log(log_container)

    if start_clicked:
        _reset_run()
        _render_log(log_container)
        url = f"{st.session_state.backend_url.rstrip('/')}/api/research/stream"
        body = {
            "topic": topic.strip(),
            "max_iterations": max_iterations,
            "results_per_query": results_per_query,
            "enable_human_review": enable_human_review,
        }
        _consume_stream(url, body, log_container)

    if st.session_state.error:
        st.error(st.session_state.error)

    if st.session_state.awaiting_approval:
        st.subheader("🙋 Human review requested")
        payload = st.session_state.interrupt_payload or {}
        st.write(payload.get("message", "Please review the research direction."))
        critique = payload.get("critique") or {}
        if critique:
            st.markdown(f"**Critic reasoning:** {critique.get('reasoning', '')}")
            if critique.get("gaps"):
                st.markdown("**Gaps identified:** " + ", ".join(critique["gaps"]))

        feedback = st.text_area("Feedback (used if you reject and request replanning)")
        col_approve, col_reject = st.columns([1, 1])

        if col_approve.button("✅ Approve, write the report"):
            resume_url = f"{st.session_state.backend_url.rstrip('/')}/api/research/{st.session_state.thread_id}/resume"
            st.session_state.awaiting_approval = False
            _consume_stream(resume_url, {"approved": True, "feedback": None}, log_container)
            st.rerun()

        if col_reject.button("🔁 Reject, replan with feedback"):
            resume_url = f"{st.session_state.backend_url.rstrip('/')}/api/research/{st.session_state.thread_id}/resume"
            st.session_state.awaiting_approval = False
            _consume_stream(resume_url, {"approved": False, "feedback": feedback or None}, log_container)
            st.rerun()

    if st.session_state.report:
        st.divider()
        st.subheader("📄 Research Report")
        st.markdown(st.session_state.report)
        if st.session_state.citations:
            with st.expander("Sources", expanded=False):
                for citation in st.session_state.citations:
                    st.markdown(f"**[{citation['id']}] {citation['title']}**  \n{citation['url']}")


if __name__ == "__main__":
    main()
