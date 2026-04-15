# Audrey — LangGraph Auto-Router

Audrey is a self-hosted AI router that exposes an OpenAI-compatible API and routes requests across local and cloud-backed Ollama models using a fast-path / deep-panel architecture.

It supports:
- request classification by task type
- single-model fast-path execution for simple requests
- multi-worker deep-panel synthesis for complex requests
- automatic web search
- OpenAPI-based tool discovery
- optional ReAct-style tool loops
- planning, reflection, and adaptive escalation
- OpenAI-style SSE responses

## Architecture

```text
Client (OpenAI-compatible) ──► /v1/chat/completions
                                      │
                              ┌───────▼────────┐
                              │  Router (4B)   │
                              └───────┬────────┘
                                      │
                         ┌────────────┼────────────┐
                         ▼                         ▼
                   Fast Path                 Deep Panel
             (single model, optional      (2–3 workers in parallel
              tools, reflection)           → synthesizer merges)
                         │                         │
                         │                    Optional planning
                         │                         │
                    Adaptive escalation      Reflection gate
                         │                         │
                         └────────────┬────────────┘
                                      │
                              ┌───────▼────────┐
                              │ Tool Registry   │
                              │ + Web Search    │
                              └────────────────┘
```

## Virtual Models

| Model | Behavior |
|---|---|
| `audrey_deep` | Auto-routes: tries fast path for high-confidence, simple requests; otherwise uses the mixed deep panel. |
| `audrey_cloud` | Always uses the cloud-only deep panel. |
| `audrey_local` | Always uses the local-only deep panel. |

## Core Features

### Smart routing
A small router model classifies requests into `code`, `reasoning`, `vl`, or `general`, with keyword pre-filters handling many common cases before the router model is called.

### Complexity gate
Large inputs can be forced to the deep panel even when classification confidence is high. This helps prevent large code reviews or document analysis from being handled by a single-model fast path.

### Fast path
For eligible `audrey_deep` requests, Audrey selects a single best-fit model from the configured registry. If tools are enabled, a tool registry is available, and the selected model supports tools, the fast path can run a ReAct-style tool loop. Otherwise it performs a direct single-model completion.

### Deep panel
Complex requests are sent to multiple specialist workers in parallel. Their drafts are then merged by a synthesizer model into one final answer.

### Planning, reflection, and escalation
Complex requests can be decomposed into sub-tasks before worker dispatch. Reflection checks can evaluate response quality, and weak fast-path results can escalate automatically to the deep panel.

### Tool discovery
Audrey discovers tools dynamically from one or more OpenAPI-compatible HTTP servers at startup. No hardcoded wrappers are required.

### Web search
Audrey can use either SearXNG or Brave for current-information queries, based on simple search-trigger heuristics.

### Model health tracking
Failed models are temporarily cooled down using exponential backoff so the router can avoid repeatedly selecting unhealthy models.

### SSE responses
Audrey returns OpenAI-style SSE chunks for streaming clients.

## Behavior Notes

### Fast path is not always a tool loop
The fast path only becomes a ReAct/tool run when all of the following are true:
- fast path is selected
- tools are enabled
- a tool registry exists and has discovered tools
- the chosen model is listed as tool-capable

Otherwise the fast path is a normal single-model completion.

### Streaming behavior differs by path
- **Deep panel:** worker drafts are generated first, then the synthesizer streams output.
- **Fast path:** the current streaming endpoint runs the full fast-path flow first, then emits the completed response in chunks. It is not yet true live token passthrough for the ReAct path.

### Cache behavior
Caching currently applies only to non-streaming requests.

## Quick Start

### Prerequisites

- Docker
- Docker Compose
- Ollama running with the models you plan to use
- An external Docker network named `ollama-net`
- GPU recommended for local models

### 1. Clone the repository

```bash
git clone git@github.com:robocoppa/audrey_ai.git
cd audrey_ai
```

### 2. Configure the project

Edit `config.yaml` for your model registry, deep-panel assignments, tool servers, timeouts, and agentic settings.

Set environment variables as needed for your deployment, including `OLLAMA_BASE_URL`, search backend settings, and any API key you want Audrey to require.

### 3. Pull required models

At minimum, pull the router model:

```bash
ollama pull qwen3:4b
```

Then pull the local worker and synthesizer models referenced in your `config.yaml`.

### 4. Start the stack

Full stack:

```bash
docker compose up -d
```

Tools only:

```bash
docker compose -f docker-compose_tools.yaml up -d
```

Run Audrey outside Docker:

```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

### 5. Test it

Health check:

```bash
curl http://localhost:8000/health
```

Non-streaming request:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "audrey_deep",
    "messages": [{"role": "user", "content": "Explain quicksort with a Python implementation"}],
    "stream": false
  }'
```

Streaming request:

```bash
curl -N http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "audrey_deep",
    "messages": [{"role": "user", "content": "What happened in the news today?"}],
    "stream": true
  }'
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|---|---:|---|
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama API base URL |
| `ROUTER_MODEL` | `qwen3:4b` | Router model used for classification and helper tasks |
| `DEFAULT_TEMPERATURE` | `0.2` | Default generation temperature |
| `GPU_CONCURRENCY` | `1` | Maximum concurrent local model calls |
| `MAX_DEEP_WORKERS` | `2` | Max deep-panel workers for local or mixed mode |
| `MAX_DEEP_WORKERS_CLOUD` | `3` | Max deep-panel workers for cloud mode |
| `API_KEY` | empty | Optional bearer token required by Audrey |
| `TOOLS_ENABLED` | `true` | Enable or disable tool calling |
| `SEARCH_BACKEND` | `searxng` | Search backend: `searxng` or `brave` |
| `SEARXNG_URL` | `http://searxng:8080` | SearXNG endpoint |
| `BRAVE_API_KEY` | empty | Brave Search API key |
| `EMIT_ROUTING_BANNER` | `true` | Show routing banner in responses |
| `EMIT_STATUS_UPDATES` | `true` | Show status updates during streaming |
| `COMPLEXITY_FORCE_DEEP` | `true` | Force deep panel for large inputs |
| `COMPLEXITY_TOKEN_THRESHOLD` | `500` | Estimated token threshold for deep-panel forcing |
| `REACT_MAX_ROUNDS` | `3` | Maximum ReAct/tool rounds |
| `REFLECTION_ENABLED` | `true` | Enable reflection checks |
| `PLANNING_ENABLED` | `true` | Enable sub-task decomposition |
| `ESCALATION_ENABLED` | `true` | Enable fast-path escalation |

### `config.yaml`

The main config file defines:
- `model_registry` for fast-path candidate ranking
- `deep_panel`, `deep_panel_cloud`, and `deep_panel_local` for worker/synthesizer selection
- `agentic` settings for react, planning, reflection, and escalation
- `timeouts`, `cache`, `search`, `tool_servers`, and `tools`

## Tool Server Architecture

The included `custom_tools.py` service exposes:
- web search
- memory store / recall / search / list
- Python sandbox execution
- system stats
- document reading
- sandboxed file read / write / list
- SQLite query and schema inspection

These endpoints are discovered from the server’s OpenAPI spec and converted into Ollama-compatible tool definitions.

## Docker Compose Notes

Two compose files are included:
- `docker-compose.yaml` for Audrey + SearXNG + custom tools
- `docker-compose_tools.yaml` for SearXNG + custom tools only

Both expect an external Docker network named `ollama-net`.

The current compose files use named Docker volumes for tool data and sandbox storage. That works, but on Unraid you may prefer bind mounts into `/mnt/user/appdata/...` so the data is easier to inspect, back up, and manage.

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat completions |
| `GET` | `/v1/models` | List virtual models |
| `GET` | `/health` | Health check with Ollama, tool, cache, and agentic status |
| `POST` | `/v1/tools/rediscover` | Re-scan tool servers without restart |

## Client Usage

Use Audrey with any OpenAI-compatible client by setting the base URL to your Audrey instance and choosing one of:
- `audrey_deep`
- `audrey_cloud`
- `audrey_local`

### Python example

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="your-key")
response = client.chat.completions.create(
    model="audrey_deep",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

## Project Structure

```text
audrey_ai/
├── main.py
├── config.py
├── state.py
├── models.py
├── helpers.py
├── health.py
├── cache.py
├── ollama.py
├── search.py
├── classifier.py
├── agents.py
├── pipeline.py
├── streaming.py
├── tool_registry.py
├── custom_tools.py
├── config.yaml
├── requirements.txt
├── requirements_tools.txt
├── Dockerfile
├── Dockerfile.tools
├── docker-compose.yaml
├── docker-compose_tools.yaml
└── README.md
```

## License

MIT
