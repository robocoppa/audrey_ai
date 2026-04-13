# Audrey — LangGraph Auto-Orchestrator

A self-hosted AI orchestrator that routes requests to the best available model using a **fast-path / deep-panel** architecture. Runs entirely on your own hardware with [Ollama](https://ollama.com), supports cloud-bridged models, automatic tool calling via OpenAPI, and web search — all behind an OpenAI-compatible API.

## Architecture

```
Client (OpenAI-compatible) ──► /v1/chat/completions
                                      │
                              ┌───────▼────────┐
                              │  Router (1.5B)  │  ← classifies: code / reasoning / vl / general
                              └───────┬────────┘
                                      │
                         ┌────────────┼────────────┐
                         ▼                         ▼
                   ⚡ Fast Path               🧠 Deep Panel
                 (single best model         (2-3 workers in parallel
                  + tool calling)            → synthesizer merges)
                         │                         │
                         └────────────┬────────────┘
                                      │
                              ┌───────▼────────┐
                              │  Tool Registry  │  ← auto-discovers OpenAPI servers
                              │  + Web Search   │
                              └────────────────┘
```

**Three virtual models** are exposed:

| Model           | Behavior |
|-----------------|----------|
| `audrey_deep`   | Auto-routes: fast path when confident, deep panel otherwise. Mixed cloud + local workers. |
| `audrey_cloud`  | Always deep panel with cloud-only workers + synthesizer. |
| `audrey_local`  | Always deep panel with local-only workers + synthesizer. |

## Features

- **Smart routing** — A small router model classifies requests by type and confidence, then picks the optimal path.
- **Fast path** — High-confidence requests go to a single top model with tool support. Low latency.
- **Deep panel** — Complex requests fan out to 2-3 specialist workers in parallel, then a synthesizer merges their outputs.
- **Automatic tool calling** — Discovers tools from any OpenAPI-compatible server at startup. Zero hardcoding.
- **Web search** — SearXNG (local, no API key) or Brave Search. Auto-triggered by temporal/factual queries.
- **Model health tracking** — Exponential backoff on failures, automatic fallback to healthy models.
- **Caching** — LRU cache with TTL for repeated queries.
- **OpenAI-compatible API** — Drop-in replacement for any client that speaks `/v1/chat/completions`.
- **Streaming** — Full SSE streaming support.

## Quick Start

### Prerequisites

- **Docker** and **Docker Compose**
- **Ollama** running with your desired models pulled (see [Model Setup](#model-setup))
- **GPU** recommended for local models (the orchestrator respects a GPU concurrency semaphore)

### 1. Clone and configure

```bash
git clone https://github.com/YOUR_USERNAME/audrey-orchestrator.git
cd audrey-orchestrator
cp .env.example .env
# Edit .env with your settings (Ollama URL, API key, etc.)
```

### 2. Pull your Ollama models

```bash
# Required: the router model
ollama pull qwen2.5:1.5b

# Recommended local models (adjust to your VRAM)
ollama pull qwen3.5:35b-a3b
ollama pull deepseek-r1:32b
ollama pull qwen3-coder-next:latest
ollama pull gemma4:31b

# Cloud models are accessed via Ollama's cloud bridge — no pull needed
```

### 3. Start everything

```bash
# Full stack: orchestrator + tools + SearXNG
docker compose up -d

# Or just the tool services if running the orchestrator outside Docker
docker compose -f docker-compose.tools.yaml up -d
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
| `ROUTER_MODEL` | `qwen2.5:1.5b` | Small model for request classification |
| `DEFAULT_TEMPERATURE` | `0.2` | Default temperature for generation |
| `GPU_CONCURRENCY` | `1` | Max concurrent local model requests |
| `MAX_DEEP_WORKERS` | `2` | Max parallel workers in deep panel |
| `API_KEY` | _(empty)_ | Bearer token for API auth (disabled if empty) |
| `TOOLS_ENABLED` | `true` | Enable/disable tool calling |
| `SEARCH_BACKEND` | `searxng` | `searxng` or `brave` |
| `SEARXNG_URL` | `http://searxng:8080` | SearXNG instance URL |
| `BRAVE_API_KEY` | _(empty)_ | Brave Search API key (if using Brave) |
| `EMIT_ROUTING_BANNER` | `true` | Show routing metadata in responses |
| `EMIT_STATUS_UPDATES` | `true` | Show status messages during streaming |
| `LOG_LEVEL` | `INFO` | Logging level |

### config.yaml

See the inline comments in [`config.yaml`](config.yaml) for full documentation of model registries, panel configs, timeouts, and cache settings.

## Adding Custom Tools

The orchestrator auto-discovers tools from any OpenAPI-compatible HTTP server. To add new tools:

1. **Create a FastAPI/Flask/Express server** with an OpenAPI spec (FastAPI generates this automatically).
2. **Add its URL** to `tool_servers` in `config.yaml`:
   ```yaml
   tool_servers:
     - "http://custom-tools:8001"
     - "http://my-new-tool:8002"
   ```
3. **Restart** the orchestrator (or call `POST /v1/tools/rediscover`).

The built-in tool server (`custom_tools.py`) provides: web search, key-value memory, Python sandbox, system monitor, filesystem ops, SQL queries, and document reading.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat completion |
| `GET`  | `/v1/models` | List available virtual models |
| `GET`  | `/health` | Health check with Ollama status, cache stats, tool info |
| `POST` | `/v1/tools/rediscover` | Re-scan tool servers without restart |

## Using with Clients

Any OpenAI-compatible client works. Set the base URL to your orchestrator and use `audrey_deep`, `audrey_cloud`, or `audrey_local` as the model name.

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
audrey-orchestrator/
├── main.py                  # FastAPI orchestrator (routing, panels, streaming)
├── tool_registry.py         # OpenAPI tool discovery and dispatch
├── custom_tools.py          # Built-in tool server (search, memory, python, etc.)
├── config.yaml              # Model registry, panels, timeouts, tool servers
├── requirements.txt         # Python dependencies (orchestrator)
├── requirements.tools.txt   # Python dependencies (tool server)
├── Dockerfile               # Orchestrator container
├── Dockerfile.tools         # Tool server container
├── docker-compose.yaml      # Full stack (orchestrator + tools + SearXNG)
├── docker-compose.tools.yaml # Tools only (SearXNG + custom-tools)
├── searxng/
│   └── settings.yml         # SearXNG config (enables JSON API)
├── .env.example             # Environment variable template
├── .gitignore
└── LICENSE
```

## License

MIT — see [LICENSE](LICENSE).
