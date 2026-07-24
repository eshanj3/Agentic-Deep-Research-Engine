"""
Pydantic & TypedDict schemas for the Deep Research Agent's LangGraph state.

`ResearchState` is the single object threaded through every node. List fields
that multiple nodes append to across loop iterations (search results, seen
URLs, the agent activity log) use `operator.add` as their LangGraph reducer so
that partial node outputs are concatenated rather than overwritten. Scalar /
object fields (critique, iteration, token_usage, report, ...) use the default
"last write wins" merge, so nodes that update them must read-modify-write the
previous value themselves.
"""
from __future__ import annotations

import operator
from datetime import datetime, timezone
from typing import Annotated, Any, List, Optional, TypedDict

from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    """A single normalized result returned by a web search provider."""

    query: str
    url: str
    title: str
    content: str
    score: float = 0.0
    published_date: Optional[str] = None
    source: str = "tavily"

    def as_context_block(self, index: int) -> str:
        """Render this result as a numbered context block for LLM prompts."""
        date_part = f" ({self.published_date})" if self.published_date else ""
        return (
            f"[{index}] {self.title}{date_part}\n"
            f"URL: {self.url}\n"
            f"Content: {self.content.strip()[:2500]}\n"
        )


class Citation(BaseModel):
    """A deduplicated citation entry used in the final report's reference list."""

    id: int
    url: str
    title: str
    snippet: str


class CritiqueResult(BaseModel):
    """Output of the Synthesizer/Critic node's evaluation of gathered evidence."""

    sufficient: bool = Field(
        description="True if gathered evidence is sufficient to write a comprehensive report."
    )
    reasoning: str = Field(default="", description="Explanation of the sufficiency decision.")
    contradictions: List[str] = Field(default_factory=list)
    gaps: List[str] = Field(default_factory=list)
    follow_up_queries: List[str] = Field(default_factory=list)


class TokenUsage(BaseModel):
    """Cumulative token / cost accounting for a single research run."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    calls: int = 0

    def add(self, other: "TokenUsage") -> "TokenUsage":
        """Return a new TokenUsage that is the sum of this one and `other`."""
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            estimated_cost_usd=round(self.estimated_cost_usd + other.estimated_cost_usd, 6),
            calls=self.calls + other.calls,
        )


class AgentLogEntry(BaseModel):
    """A single human-readable trace event, streamed to the client in real time."""

    node: str
    message: str
    data: Optional[dict[str, Any]] = None
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ResearchState(TypedDict, total=False):
    """Full graph state threaded through every node of the research workflow."""

    # --- run configuration ---------------------------------------------------
    topic: str
    max_iterations: int
    results_per_query: int
    enable_human_review: bool
    thread_id: str

    # --- planning --------------------------------------------------------------
    sub_queries: List[str]

    # --- evidence gathering (accumulates across loop iterations) ---------------
    search_results: Annotated[List[SearchResult], operator.add]
    seen_urls: Annotated[List[str], operator.add]

    # --- critique / control flow -------------------------------------------------
    critique: Optional[CritiqueResult]
    iteration: int

    # --- human-in-the-loop ---------------------------------------------------------
    approved: Optional[bool]
    human_feedback: Optional[str]

    # --- output --------------------------------------------------------------------
    report: Optional[str]
    citations: List[Citation]

    # --- observability ---------------------------------------------------------------
    agent_log: Annotated[List[AgentLogEntry], operator.add]
    token_usage: TokenUsage
    error: Optional[str]


def new_initial_state(
    topic: str,
    max_iterations: int = 3,
    results_per_query: int = 5,
    enable_human_review: bool = False,
    thread_id: str = "",
) -> ResearchState:
    """Factory for a fresh, empty research run."""
    return ResearchState(
        topic=topic,
        max_iterations=max_iterations,
        results_per_query=results_per_query,
        enable_human_review=enable_human_review,
        thread_id=thread_id,
        sub_queries=[],
        search_results=[],
        seen_urls=[],
        critique=None,
        iteration=0,
        approved=None,
        human_feedback=None,
        report=None,
        citations=[],
        agent_log=[],
        token_usage=TokenUsage(),
        error=None,
    )
