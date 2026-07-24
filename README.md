# Deep Research & Report Generation Engine

An autonomous, multi-agent research system built on **LangGraph**. Give it a
topic; it plans search queries, gathers evidence concurrently, critiques its
own findings for gaps and contradictions, optionally pauses for your
approval, and writes a cited Markdown report — all streamed to the client in
real time over SSE.

```
Planner ──► Search ──► Critic ─┬─(insufficient, budget left)──► Search
                                │
                                └─(sufficient, or budget exhausted)──► [Human Review]* ──► Writer
                                                                             │
                                                                    (rejected) └──► Planner
```
\* skipped automatically when `enable_human_review` is `false`.

## Architecture

| Node | Responsibility |
|---|---|
| **Planner** | Breaks the topic into 3–5 distinct, non-overlapping search sub-queries. |
| **Search** | Runs all pending sub-queries concurrently against Tavily, dedupes against previously-seen URLs. |
| **Critic / Synthesizer** | Evaluates gathered evidence for sufficiency, contradictions, and gaps; emits targeted follow-up queries when more research is needed (max 3 loops by default). |
| **Human Review** *(optional)* | Interrupts the graph and waits for the user to approve the research direction or reject it with feedback, which routes back to the Planner. |
| **Report Writer** | Synthesizes all evidence into a Markdown report with numbered inline citations `[1]`, `[2]`, ... and a `## Sources` section. |

State flows through the graph as a single `ResearchState` TypedDict
(`app/graph/state.py`), with Pydantic models for every structured sub-object
(`SearchResult`, `CritiqueResult`, `Citation`, `TokenUsage`, `AgentLogEntry`).

## Repository layout

```
deep_research_agent/
├── .env.example
├── Dockerfile                  # backend image
├── docker-compose.yml
├── requirements.txt
├── pyproject.toml
├── README.md
├── app/
│   ├── main.py                 # FastAPI app, SSE endpoints
│   ├── graph/
│   │   ├── state.py             # TypedDict + Pydantic state schemas
│   │   ├── nodes.py             # Planner / Search / Critic / Human Review / Writer
│   │   └── workflow.py          # StateGraph construction + compilation
│   ├── tools/
│   │   └── search.py            # Tavily async search integration
│   └── utils/
│       └── telemetry.py         # LangSmith setup + token/cost accounting
├── frontend/
│   ├── app.py                   # Streamlit UI
│   ├── requirements.txt
│   └── Dockerfile
└── tests/
    ├── test_graph.py            # node + routing unit tests (mocked LLM/search)
    └── test_api.py              # FastAPI contract tests (mocked graph)
```

## Prerequisites

- Python 3.11+
- API keys: an LLM provider (OpenAI and/or Anthropic) + [Tavily](https://tavily.com) for web search
- Optionally, a [LangSmith](https://smith.langchain.com) API key for tracing
- Docker + Docker Compose (only if you want the containerized run)

## 1. Local setup (no Docker)

```bash
# Clone and enter the repo
git clone <your-fork-url> deep_research_agent
cd deep_research_agent

# Create and activate a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Install backend dependencies
pip install -r requirements.txt

# Install frontend dependencies (separate, lighter env is also fine)
pip install -r frontend/requirements.txt

# Configure environment variables
cp .env.example .env
# then edit .env and fill in OPENAI_API_KEY / ANTHROPIC_API_KEY / TAVILY_API_KEY
```

### Run the backend

```bash
uvicorn app.main:app --reload --port 8000
```

The API is now live at `http://localhost:8000` (interactive docs at `/docs`).

### Run the frontend

In a second terminal (same venv, or its own):

```bash
export BACKEND_URL=http://localhost:8000   # Windows: set BACKEND_URL=http://localhost:8000
streamlit run frontend/app.py
```

Open the URL Streamlit prints (defaults to `http://localhost:8501`).

### Run the tests

```bash
pip install pytest pytest-asyncio   # or: pip install ".[dev]"
pytest -v
```

All tests mock the LLM and search calls, so they run without API keys or
network access.

## 2. Run with Docker Compose

```bash
cp .env.example .env
# edit .env with your API keys

docker compose up --build
```

- Backend: `http://localhost:8000`
- Frontend: `http://localhost:8501`

## API reference

### `POST /api/research/stream`

Starts a new research run and streams progress as Server-Sent Events.

Request body:

```json
{
  "topic": "The impact of small modular reactors on grid decarbonization",
  "max_iterations": 3,
  "results_per_query": 5,
  "enable_human_review": false
}
```

SSE events emitted:

| Event | Payload | Meaning |
|---|---|---|
| `node_update` | `{thread_id, node, log[], report?, citations?}` | A node finished; `report`/`citations` are present only on the `writer` node's update. |
| `interrupt` | `{thread_id, payload}` | The graph paused at `human_review`; `payload.critique` summarizes findings so far. Call the resume endpoint to continue. |
| `done` | `{thread_id, token_usage}` | The run completed successfully. |
| `error` | `{thread_id, message}` | The run failed. |

### `POST /api/research/{thread_id}/resume`

Resumes a run paused at `human_review`.

```json
{ "approved": true, "feedback": null }
```

Rejecting (`"approved": false`) with `feedback` routes back to the Planner,
which incorporates the feedback into a new set of sub-queries.

### `GET /api/research/{thread_id}/state`

Returns a point-in-time snapshot of a run's full state — useful for
debugging, polling as an SSE alternative, or building a different frontend.

### `GET /health`

Liveness check, used by the Docker healthcheck.

## Configuration reference (`.env`)

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | — | Provider key(s) for whichever `RESEARCH_MODEL`/`WRITER_MODEL` you select (routed through LiteLLM). |
| `TAVILY_API_KEY` | — | Required for the Search node. |
| `RESEARCH_MODEL` | `gpt-4o-mini` | Model used by the Planner and Critic. |
| `WRITER_MODEL` | `gpt-4o-mini` | Model used by the Report Writer. |
| `MAX_SEARCH_RESULTS_PER_QUERY` | `5` | Results fetched per sub-query. |
| `SEARCH_TIMEOUT_SECONDS` | `30` | Per-request Tavily timeout. |
| `LANGCHAIN_API_KEY` | — | If set, enables LangSmith tracing automatically. |
| `LANGCHAIN_PROJECT` | `deep-research-agent` | LangSmith project name. |
| `BACKEND_URL` | `http://localhost:8000` | Used by the Streamlit frontend to reach the API. |

## Production considerations

This repository is a complete, runnable reference implementation. Before
putting it in front of real traffic, consider:

- **Checkpointer**: `app/graph/workflow.py` uses LangGraph's in-memory
  `MemorySaver`, which loses paused runs on restart and doesn't work across
  multiple backend replicas. Swap it for a persistent checkpointer (e.g. a
  Postgres- or Redis-backed one) so a `human_review` interrupt can be
  resumed by any process.
- **Auth & rate limiting**: the API has no authentication or per-user rate
  limiting out of the box — add a reverse proxy or FastAPI middleware
  appropriate to your deployment.
- **Cost estimation**: `app/utils/telemetry.py` ships an illustrative
  per-model pricing table for estimating spend. Update it to match your
  provider's current pricing, or rely on LangSmith's own usage dashboards
  for authoritative numbers.
- **Search provider quotas**: Tavily calls are retried with exponential
  backoff (`tenacity`) and degrade to an empty result set on repeated
  failure rather than crashing a run — tune `SEARCH_TIMEOUT_SECONDS` and the
  retry policy in `app/tools/search.py` to your plan's rate limits.
