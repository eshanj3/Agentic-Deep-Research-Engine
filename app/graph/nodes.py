"""
Node implementations for the Deep Research Agent graph: Planner, Search,
Critic (Synthesizer), Human Review, and Report Writer — plus the two
conditional-edge routing functions that connect them.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, List

import litellm
from langgraph.types import interrupt

from app.graph.state import (
    AgentLogEntry,
    Citation,
    CritiqueResult,
    ResearchState,
    SearchResult,
    TokenUsage,
)
from app.tools.search import run_searches_concurrently
from app.utils.telemetry import usage_from_litellm_response

logger = logging.getLogger("deep_research.nodes")

RESEARCH_MODEL = os.getenv("RESEARCH_MODEL", "gpt-4o-mini")
WRITER_MODEL = os.getenv("WRITER_MODEL", os.getenv("RESEARCH_MODEL", "gpt-4o-mini"))
DEFAULT_RESULTS_PER_QUERY = int(os.getenv("MAX_SEARCH_RESULTS_PER_QUERY", "5"))
LLM_JSON_RETRY_ATTEMPTS = 2  # additional attempts after the first, on parse failure


class LLMOutputParseError(RuntimeError):
    """Raised when the model's response cannot be parsed as the expected JSON shape."""


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    fence_match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()
    return text


async def _call_llm_json(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.2,
) -> tuple[dict, TokenUsage]:
    """Call the LLM and parse a strict JSON object from its response.

    Retries with an increasingly explicit instruction if the model's output
    fails to parse as JSON, up to `LLM_JSON_RETRY_ATTEMPTS` extra attempts.
    """
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    last_error: Exception | None = None
    usage_total = TokenUsage()

    for attempt in range(1, LLM_JSON_RETRY_ATTEMPTS + 2):
        response = await litellm.acompletion(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        usage_total = usage_total.add(usage_from_litellm_response(model, response))
        raw_text = response.choices[0].message.content or ""
        cleaned = _strip_code_fences(raw_text)
        try:
            parsed = json.loads(cleaned)
            return parsed, usage_total
        except json.JSONDecodeError as exc:
            last_error = exc
            logger.warning(
                "LLM JSON parse failed (attempt %s/%s): %s",
                attempt,
                LLM_JSON_RETRY_ATTEMPTS + 1,
                exc,
            )
            messages.append({"role": "assistant", "content": raw_text})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "That was not valid JSON. Respond again with ONLY a single valid "
                        "JSON object, no markdown fences, no commentary."
                    ),
                }
            )

    raise LLMOutputParseError(f"Failed to parse LLM JSON output after retries: {last_error}")


# --------------------------------------------------------------------------------------
# Planner
# --------------------------------------------------------------------------------------


async def planner_node(state: ResearchState) -> dict[str, Any]:
    """Generate 3-5 distinct, non-overlapping search sub-queries for the topic."""
    topic = state["topic"]
    feedback = state.get("human_feedback")

    feedback_block = (
        f"\n\nA previous plan was reviewed by the user, who gave this feedback — "
        f"incorporate it into the new plan: {feedback}"
        if feedback
        else ""
    )

    system_prompt = (
        "You are the Planning agent of a deep research system. Break a research "
        "topic into 3 to 5 distinct search sub-queries that together give broad, "
        "non-redundant coverage of the topic (e.g. background/definitions, current "
        "state or recent developments, key evidence or players, controversies or "
        "open questions, and outlook, as applicable to this topic). Respond with "
        'ONLY a JSON object of the shape {"sub_queries": ["...", "..."]}.'
    )
    user_prompt = f"Research topic: {topic}{feedback_block}"

    parsed, usage = await _call_llm_json(
        model=RESEARCH_MODEL, system_prompt=system_prompt, user_prompt=user_prompt
    )
    sub_queries = [str(q).strip() for q in parsed.get("sub_queries", []) if str(q).strip()][:5]
    if not sub_queries:
        # Deterministic fallback so the graph never stalls on a bad LLM response.
        sub_queries = [topic]

    log = AgentLogEntry(
        node="planner",
        message=f"Generated {len(sub_queries)} search sub-quer{'y' if len(sub_queries) == 1 else 'ies'}.",
        data={"sub_queries": sub_queries},
    )
    return {
        "sub_queries": sub_queries,
        "agent_log": [log],
        "token_usage": state.get("token_usage", TokenUsage()).add(usage),
        "human_feedback": None,
    }


# --------------------------------------------------------------------------------------
# Search
# --------------------------------------------------------------------------------------


async def search_node(state: ResearchState) -> dict[str, Any]:
    """Execute all pending sub-queries concurrently and append normalized results."""
    queries = state.get("sub_queries", [])
    results_per_query = state.get("results_per_query", DEFAULT_RESULTS_PER_QUERY)

    raw_results: List[SearchResult] = await run_searches_concurrently(queries, results_per_query)

    seen = set(state.get("seen_urls", []))
    new_results = [r for r in raw_results if r.url and r.url not in seen]
    new_urls = [r.url for r in new_results]

    log = AgentLogEntry(
        node="search",
        message=f"Ran {len(queries)} quer{'y' if len(queries) == 1 else 'ies'}, "
        f"retrieved {len(new_results)} new source(s).",
        data={"queries": queries, "result_count": len(new_results)},
    )
    return {
        "search_results": new_results,
        "seen_urls": new_urls,
        "agent_log": [log],
    }


# --------------------------------------------------------------------------------------
# Critic / Synthesizer
# --------------------------------------------------------------------------------------


async def critic_node(state: ResearchState) -> dict[str, Any]:
    """Evaluate gathered evidence for sufficiency, contradictions, and coverage gaps."""
    topic = state["topic"]
    results = state.get("search_results", [])
    iteration = state.get("iteration", 0) + 1

    context = "\n\n".join(r.as_context_block(i + 1) for i, r in enumerate(results))
    system_prompt = (
        "You are the Synthesizer/Critic agent of a deep research system. Evaluate "
        "whether the evidence below is sufficient to write a comprehensive, accurate "
        "report on the topic. Look specifically for contradictions between sources "
        "and remaining coverage gaps. Respond with ONLY a JSON object of the shape "
        '{"sufficient": bool, "reasoning": "...", "contradictions": ["..."], '
        '"gaps": ["..."], "follow_up_queries": ["..."]}. '
        "follow_up_queries must be empty if sufficient is true; otherwise give 1-3 "
        "targeted queries that would close the identified gaps."
    )
    user_prompt = f"Topic: {topic}\n\nGathered evidence:\n\n{context or '(no evidence gathered yet)'}"

    parsed, usage = await _call_llm_json(
        model=RESEARCH_MODEL, system_prompt=system_prompt, user_prompt=user_prompt
    )
    critique = CritiqueResult(
        sufficient=bool(parsed.get("sufficient", False)),
        reasoning=str(parsed.get("reasoning", "")),
        contradictions=[str(c) for c in (parsed.get("contradictions") or [])],
        gaps=[str(g) for g in (parsed.get("gaps") or [])],
        follow_up_queries=[str(q) for q in (parsed.get("follow_up_queries") or [])][:3],
    )

    summary = (
        "evidence sufficient."
        if critique.sufficient
        else f"{len(critique.gaps)} gap(s) identified, {len(critique.follow_up_queries)} follow-up quer(y/ies) queued."
    )
    log = AgentLogEntry(
        node="critic",
        message=f"Iteration {iteration}: {summary}",
        data=critique.model_dump(),
    )
    return {
        "critique": critique,
        "iteration": iteration,
        "sub_queries": critique.follow_up_queries,
        "agent_log": [log],
        "token_usage": state.get("token_usage", TokenUsage()).add(usage),
    }


def route_after_critic(state: ResearchState) -> str:
    """Decide whether to keep searching, move to human review, or write the report."""
    critique = state.get("critique")
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 3)

    needs_more = bool(critique and not critique.sufficient)
    has_follow_ups = bool(state.get("sub_queries"))

    if needs_more and has_follow_ups and iteration < max_iterations:
        return "search"
    return "human_review" if state.get("enable_human_review") else "writer"


# --------------------------------------------------------------------------------------
# Human-in-the-loop review
# --------------------------------------------------------------------------------------


def human_review_node(state: ResearchState) -> dict[str, Any]:
    """Pause the graph for user approval of the research direction.

    Resumed by the caller via `Command(resume={"approved": bool, "feedback":
    Optional[str]})`. Requires the graph to be compiled with a checkpointer
    (see `app/graph/workflow.py`).
    """
    critique = state.get("critique")
    payload = {
        "type": "approval_request",
        "topic": state["topic"],
        "iteration": state.get("iteration", 0),
        "sources_found": len(state.get("search_results", [])),
        "critique": critique.model_dump() if critique else None,
        "message": (
            "Review the research gathered so far. Approve to proceed to report "
            "writing, or reject with feedback to trigger another planning round."
        ),
    }
    decision = interrupt(payload)

    approved = bool(decision.get("approved", True)) if isinstance(decision, dict) else True
    feedback = decision.get("feedback") if isinstance(decision, dict) else None

    log = AgentLogEntry(
        node="human_review",
        message="Approved by user." if approved else "Rejected by user — replanning.",
        data={"approved": approved, "feedback": feedback},
    )
    return {"approved": approved, "human_feedback": feedback, "agent_log": [log]}


def route_after_human(state: ResearchState) -> str:
    """Route to the writer if approved, otherwise loop back to planning."""
    return "writer" if state.get("approved", True) else "planner"


# --------------------------------------------------------------------------------------
# Report writer
# --------------------------------------------------------------------------------------


def _build_citations(results: List[SearchResult]) -> List[Citation]:
    citations: List[Citation] = []
    seen_urls: set[str] = set()
    for result in results:
        if result.url in seen_urls:
            continue
        seen_urls.add(result.url)
        citations.append(
            Citation(
                id=len(citations) + 1,
                url=result.url,
                title=result.title,
                snippet=result.content.strip()[:280],
            )
        )
    return citations


async def writer_node(state: ResearchState) -> dict[str, Any]:
    """Synthesize all gathered evidence into a cited Markdown report."""
    topic = state["topic"]
    results = state.get("search_results", [])
    citations = _build_citations(results)

    context = "\n\n".join(
        f"[{c.id}] {c.title}\nURL: {c.url}\nExcerpt: {c.snippet}" for c in citations
    )

    system_prompt = (
        "You are the Report Writer agent of a deep research system. Write a "
        "comprehensive, well-structured Markdown report on the given topic using "
        "ONLY the numbered sources provided below. Use numbered inline citations "
        "like [1], [2] immediately after claims drawn from that source, and include "
        "a '## Sources' section at the end listing every source by number, title, "
        "and URL. Use clear headings, avoid repetition, and explicitly note any "
        "unresolved contradictions or open questions between sources. Do not "
        "fabricate sources or facts that are not supported by the provided material."
    )
    user_prompt = f"Topic: {topic}\n\nAvailable sources:\n\n{context or '(no sources available)'}"

    response = await litellm.acompletion(
        model=WRITER_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
    )
    usage = usage_from_litellm_response(WRITER_MODEL, response)
    report = response.choices[0].message.content or ""

    log = AgentLogEntry(
        node="writer",
        message=f"Report generated ({len(report.split())} words, {len(citations)} source(s) cited).",
    )
    return {
        "report": report,
        "citations": citations,
        "agent_log": [log],
        "token_usage": state.get("token_usage", TokenUsage()).add(usage),
    }
