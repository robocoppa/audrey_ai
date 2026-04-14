# Audrey — LangGraph Auto-Router

A self-hosted AI router that sends requests to the best available model using a **fast-path / deep-panel** architecture with agentic capabilities. Runs entirely on your own hardware with [Ollama](https://ollama.com), supports cloud-bridged models, automatic tool calling via OpenAPI, ReAct agent loops, planning, reflection, and web search — all behind an OpenAI-compatible API.

## Architecture

```
Client (OpenAI-compatible) ──► /v1/chat/completions
                                      │
                              ┌───────▼────────┐
                              │  Router (4B)   │  ← classifies: code / reasoning / vl / general
                              └───────┬────────┘
                                      │
                         ┌────────────┼────────────┐
                         ▼                         ▼
                   ⚡ Fast Path               🧠 Deep Panel
                (ReAct agent loop          (2-3 workers in parallel
                 + tool calling             → synthesizer merges)
                 + reflection)                     │
                         │                    📋 Planning
                         │                 (optional sub-task
                         │                  decomposition)
                         │                         │
                    Adaptive Escalation       Reflection Gate
                   (quality check →           (quality check →
                    escalate to deep           re-synthesize
                    if needed)                 if needed)
                         │                         │
                         └────────────┬────────────┘
                                      │
                              ┌───────▼────────┐
                              │  Tool Registry  │  ← auto-discovers OpenAPI servers
                              │  + Web Search   │
                              └────────────────┘
```

**Three virtual models** are exposed:

| Model            | Behavior |
|------------------|----------|
| `audrey_deep`    | Auto-routes: fast path (ReAct) when confident + input is simple; deep panel otherwise. Mixed cloud + local workers. |
| `audrey_cloud`   | Always deep panel with cloud-only workers + synthesizer. |
| `audrey_local`   | Always deep panel with local-only workers + synthesizer. |

## Features

- **Smart routing** — A small router model (`qwen3:4b`) classifies requests by type and confidence, with keyword pre-filters for common patterns.
- **Complexity gate** — Large inputs (code reviews, document analysis) are automatically forced to the deep panel regardless of confidence, preventing single-model fast-path from handling complex multi-faceted requests.
- **Fast path with ReAct** — High-confidence simple requests go to a single top model running a ReAct agent loop (think → act → observe) with tool support.
- **Adaptive escalation** — Fast-path responses are quality-checked; short or poor answers automatically escalate to the deep panel.
- **Deep panel with planning** — Complex requests are optionally decomposed into sub-tasks, then fanned out to 2-3 specialist workers in parallel. A synthesizer merges their outputs.
- **Reflection gates** — Both fast path and deep panel outputs pass through a reflection check. Poor quality triggers retry or re-synthesis.
- **Automatic tool calling** — Discovers tools from any OpenAPI-compatible server at startup. Zero hardcoding. Context compression prevents window exhaustion during multi-round tool use.
- **Web search** — SearXNG (local, no API key) or Brave Search. Auto-triggered by temporal/factual queries.
- **Model health tracking** — Exponential backoff on failures, automatic fallback to healthy models.
- **Separate cloud worker cap** — Cloud workers run truly in parallel (no GPU semaphore), so `MAX_DEEP_WORKERS_CLOUD` can be higher than the local cap.
- **Caching** — LRU cache with TTL for repeated queries.
- **OpenAI-compatible API** — Drop-in replacement for any client that speaks `/v1/chat/completions`.
- **Streaming** — Full SSE streaming support with status updates.

## Quick Start

### Prerequisites

- **Docker** and **Docker Compose**
- **Ollama** running with your desired models pulled (see [Model Setup](#model-setup))
- **GPU** recommended for local models (Audrey respects a GPU concurrency semaphore)

### 1. Clone and configure

```bash
git clone https://github.com/robocoppa/audrey_ai.git
cd audrey_ai
cp .env.example .env
# Edit .env with your settings (Ollama URL, API key, etc.)
```

### 2. Pull your Ollama models

```bash
# Required: the router model
ollama pull qwen3:4b

# Recommended local models (adjust to your VRAM)
ollama pull qwen3.5:35b-a3b
ollama pull deepseek-r1:32b
ollama pull qwen3-coder-next:latest
ollama pull gemma4:31b

# Cloud models are accessed via Ollama's cloud bridge — no pull needed
```

### 3. Start everything

```bash
# Full stack: Audrey + tools + SearXNG
docker compose up -d

# Or just the tool services if running Audrey outside Docker
docker compose -f docker-compose_tools.yaml up -d
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

### 4. Test it

```bash
# Health check
curl http://localhost:8000/health

# Chat completion
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "audrey_deep",
    "messages": [{"role": "user", "content": "Explain quicksort with a Python implementation"}],
    "stream": false
  }'

# Streaming
curl -N http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "audrey_deep",
    "messages": [{"role": "user", "content": "What happened in the news today?"}],
    "stream": true
  }'
```

## Model Setup

Edit `config.yaml` to match the models you have available. The config has four sections:

### `model_registry`
Priority-ordered lists per task type (`code`, `reasoning`, `vl`, `general`). Used by the fast path to pick the best single model.

### `deep_panel` / `deep_panel_cloud` / `deep_panel_local`
Worker and synthesizer assignments per task type for the three virtual models. Workers run in parallel; the synthesizer merges their outputs.

### Cloud models
Models with `:cloud` in their name (e.g., `qwen3.5:397b-cloud`) are treated as cloud-bridged — they bypass the GPU semaphore and aren't checked in `ollama tags`. Configure these through Ollama's cloud model support.

### Adapting to your hardware

If you have **limited VRAM** (e.g., 24 GB), trim down to smaller models:

```yaml
deep_panel_local:
  code:
    workers:
      - "qwen2.5-coder:32b"
      - "deepseek-coder-v2:16b"
    synthesizer: "qwen3.5:35b-a3b"
```

If you're **cloud-only**, use `audrey_cloud` and configure `deep_panel_cloud` with your preferred cloud models.

## Configuration Reference

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama API endpoint |
| `ROUTER_MODEL` | `qwen3:4b` | Small model for request classification |
| `DEFAULT_TEMPERATURE` | `0.2` | Default temperature for generation |
| `GPU_CONCURRENCY` | `1` | Max concurrent local model requests |
| `MAX_DEEP_WORKERS` | `2` | Max parallel local workers in deep panel |
| `MAX_DEEP_WORKERS_CLOUD` | `3` | Max parallel cloud workers in deep panel |
| `API_KEY` | _(empty)_ | Bearer token for API auth (disabled if empty) |
| `TOOLS_ENABLED` | `true` | Enable/disable tool calling |
| `SEARCH_BACKEND` | `searxng` | `searxng` or `brave` |
| `SEARXNG_URL` | `http://searxng:8080` | SearXNG instance URL |
| `BRAVE_API_KEY` | _(empty)_ | Brave Search API key (if using Brave) |
| `EMIT_ROUTING_BANNER` | `true` | Show routing metadata in responses |
| `EMIT_STATUS_UPDATES` | `true` | Show status messages during streaming |
| `COMPLEXITY_FORCE_DEEP` | `true` | Force deep panel for large inputs |
| `COMPLEXITY_TOKEN_THRESHOLD` | `500` | Token estimate above which deep panel is forced |
| `LOG_LEVEL` | `INFO` | Logging level |
| `REACT_MAX_ROUNDS` | `3` | Max ReAct agent tool-calling rounds |
| `REFLECTION_ENABLED` | `true` | Enable response quality reflection |
| `PLANNING_ENABLED` | `true` | Enable sub-task decomposition for deep panel |
| `ESCALATION_ENABLED` | `true` | Enable fast-path → deep-panel escalation |

### config.yaml

See the inline comments in [`config.yaml`](config.yaml) for full documentation of model registries, panel configs, timeouts, and cache settings.

## Agentic Features

### Complexity Gate
When a user pastes a large code file or document and asks for analysis/review, the complexity gate detects that the input exceeds the token threshold and forces the request to the deep panel. This ensures multi-model consensus for complex tasks instead of relying on a single fast-path model. Controlled by `COMPLEXITY_FORCE_DEEP` and `COMPLEXITY_TOKEN_THRESHOLD`.

### ReAct Agent (Fast Path)
The fast path runs a ReAct (Reason + Act) loop: the model thinks about the request, optionally calls tools, observes the results, and repeats until it has enough information to respond.

### Planning (Deep Panel)
For complex queries, the router model can decompose the request into 2-3 focused sub-tasks before dispatching to workers. Each worker gets a specific sub-task assignment, and the synthesizer combines the focused answers.

### Reflection Gates
Both paths include quality checks. A reflection model evaluates whether the response is complete and of good quality. Poor responses trigger either a retry (fast path) or re-synthesis (deep panel).

### Adaptive Escalation
If the fast-path response is too short relative to the question complexity, or if reflection rates it as poor, the request automatically escalates to the full deep panel.

### Context Compression
During multi-round tool use, older tool call/result pairs are compressed into summaries to prevent context window exhaustion. Recent exchanges are preserved in full.

## Adding Custom Tools

Audrey auto-discovers tools from any OpenAPI-compatible HTTP server. To add new tools:

1. **Create a FastAPI/Flask/Express server** with an OpenAPI spec (FastAPI generates this automatically).
2. **Add its URL** to `tool_servers` in `config.yaml`:
   ```yaml
   tool_servers:
     - "http://custom-tools:8001"
     - "http://my-new-tool:8002"
   ```
3. **Restart** Audrey (or call `POST /v1/tools/rediscover`).

The built-in tool server (`custom_tools.py`) provides: web search, key-value memory, Python sandbox, system monitor, filesystem ops, SQL queries, and document reading.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat completion |
| `GET`  | `/v1/models` | List available virtual models |
| `GET`  | `/health` | Health check with Ollama status, cache stats, tool info, agentic config |
| `POST` | `/v1/tools/rediscover` | Re-scan tool servers without restart |

## Using with Clients

Any OpenAI-compatible client works. Set the base URL to your Audrey instance and use `audrey_deep`, `audrey_cloud`, or `audrey_local` as the model name.

**Open WebUI:**
```
Settings → Connections → OpenAI API → http://localhost:8000/v1
```

**Python (openai SDK):**
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="your-key")
response = client.chat.completions.create(
    model="audrey_deep",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

**curl:**
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer your-key" \
  -H "Content-Type: application/json" \
  -d '{"model":"audrey_deep","messages":[{"role":"user","content":"Hi"}]}'
```

## Project Structure

```
audrey_ai/
├── main.py                  # FastAPI app, endpoints, lifespan, request dispatch
├── config.py                # Config loading, env vars, constants
├── state.py                 # Shared mutable state (sessions, semaphore, registry)
├── models.py                # Pydantic request models, AudreyState TypedDict
├── helpers.py               # Message manipulation, token estimation, datetime, role prompts
├── health.py                # Model health tracking with exponential backoff
├── cache.py                 # LRU cache with TTL
├── ollama.py                # Ollama payload building, chat once/stream, model runners
├── search.py                # Web search detection and SearXNG/Brave backends
├── classifier.py            # Keyword prefilter, router classification, complexity gate
├── agents.py                # Planning, reflection, ReAct agent, adaptive escalation
├── pipeline.py              # LangGraph nodes, graph builders, compiled graphs
├── streaming.py             # SSE formatting, banner builder, fast/deep streamers
├── tool_registry.py         # OpenAPI tool discovery, dispatch, context compression
├── custom_tools.py          # Built-in tool server (search, memory, python, etc.)
├── config.yaml              # Model registry, panels, agentic config, timeouts
├── requirements.txt         # Python dependencies (Audrey)
├── requirements_tools.txt   # Python dependencies (tool server)
├── Dockerfile               # Audrey container
├── Dockerfile.tools         # Tool server container
├── docker-compose.yaml      # Full stack (Audrey + tools + SearXNG)
├── docker-compose_tools.yaml # Tools only (SearXNG + custom-tools)
├── searxng/
│   └── settings.yml         # SearXNG config (enables JSON API)
├── .env.example             # Environment variable template
├── .gitignore
└── LICENSE
```

## License

MIT — see [LICENSE](LICENSE).
