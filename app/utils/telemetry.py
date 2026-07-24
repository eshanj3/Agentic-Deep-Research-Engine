"""
LangSmith tracing setup and lightweight LLM token/cost accounting.

LangSmith itself is configured purely through environment variables that
LangChain/LangGraph read automatically (LANGCHAIN_TRACING_V2, LANGCHAIN_API_KEY,
LANGCHAIN_PROJECT) — `init_telemetry()` just makes sure those are set
consistently and safely defaults to "off" when no API key is present, so the
app runs fine without a LangSmith account.
"""
from __future__ import annotations

import os
from typing import Any, Optional

from app.graph.state import TokenUsage

# Approximate USD price per 1M tokens. These are illustrative defaults for
# cost *estimation* only — update them to match your provider's current
# pricing page before relying on these numbers for actual billing/budgeting.
MODEL_PRICING_PER_MILLION_TOKENS: dict[str, dict[str, float]] = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.00},
    "default": {"input": 1.00, "output": 3.00},
}


def init_telemetry() -> None:
    """Configure LangSmith tracing from environment variables.

    Safe to call even if LangSmith credentials are absent — tracing is simply
    disabled in that case rather than raising.
    """
    if os.getenv("LANGCHAIN_API_KEY"):
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        os.environ.setdefault(
            "LANGCHAIN_PROJECT", os.getenv("LANGCHAIN_PROJECT", "deep-research-agent")
        )
    else:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate USD cost for a single LLM call given token counts."""
    pricing = MODEL_PRICING_PER_MILLION_TOKENS.get(model, MODEL_PRICING_PER_MILLION_TOKENS["default"])
    input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
    output_cost = (completion_tokens / 1_000_000) * pricing["output"]
    return round(input_cost + output_cost, 6)


def _get_int_field(source: Any, field: str) -> int:
    if source is None:
        return 0
    if isinstance(source, dict):
        value = source.get(field, 0)
    else:
        value = getattr(source, field, 0)
    return int(value or 0)


def usage_from_litellm_response(model: str, response: Any) -> TokenUsage:
    """Extract a TokenUsage record from a LiteLLM/OpenAI-style completion response."""
    usage: Optional[Any] = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")

    prompt_tokens = _get_int_field(usage, "prompt_tokens")
    completion_tokens = _get_int_field(usage, "completion_tokens")
    total_tokens = prompt_tokens + completion_tokens

    return TokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        estimated_cost_usd=estimate_cost(model, prompt_tokens, completion_tokens),
        calls=1,
    )
