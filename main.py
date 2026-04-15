"""
Audrey — LangGraph Auto-Router

FastAPI application, lifespan, endpoints, and request dispatch.
All logic lives in dedicated modules; this file wires them together.
"""

import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List

import aiohttp
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

import state
from cache import cache
from config import (
    ALL_VIRTUAL_MODELS,
    API_KEY,
    CACHE_ENABLED,
    DEFAULT_TEMPERATURE,
    EMIT_ROUTING_BANNER,
    EMIT_STATUS_UPDATES,
    ESCALATION_ENABLED,
    FAST_PATH_ENABLED,
    MAX_DEEP_WORKERS,
    MAX_DEEP_WORKERS_CLOUD,
    OLLAMA_BASE_URL,
    PLANNING_ENABLED,
    REACT_MAX_ROUNDS,
    REFLECTION_ENABLED,
    ROUTER_MODEL,
    SEARCH_BACKEND,
    TOOL_SERVER_URLS,
    TOOLS_ENABLED,
    is_cloud_model,
)
from helpers import estimate_tokens, flatten_messages
from models import ChatCompletionRequest
from pipeline import (
    FAST_GRAPH,
    PREP_GRAPH,
    SYNTH_GRAPH,
    node_classify,
    node_parallel_generate,
    node_plan_panel,
    node_prepare_synthesis,
)
from streaming import _sc, banner, stream_fast_path, stream_synthesis
from tool_registry import ToolRegistry

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("audrey")


# ══════════════════════════════════════════════════════════════════════════════
#  Startup — model validation
# ══════════════════════════════════════════════════════════════════════════════

async def validate_models():
    """Check which configured models are actually available in Ollama."""
    from config import MODEL_REGISTRY, DEEP_PANEL_MIXED, DEEP_PANEL_CLOUD, DEEP_PANEL_LOCAL

    try:
        async with state.ollama_session.get(
            f"{OLLAMA_BASE_URL}/api/tags",
            timeout=aiohttp.ClientTimeout(total=10),
        ) as r:
            if r.status != 200:
                logger.warning("Cannot reach Ollama for model validation")
                return
            data = await r.json()
        local_models = {m["name"] for m in data.get("models", [])}
        state.available_models = local_models
        logger.info(
            "Ollama models available: %d — %s",
            len(local_models),
            ", ".join(sorted(local_models)),
        )

        # Warn about configured models that are missing
        configured = set()
        for category in MODEL_REGISTRY.values():
            for entry in category:
                configured.add(entry["name"])
        for panel_set in [DEEP_PANEL_MIXED, DEEP_PANEL_CLOUD, DEEP_PANEL_LOCAL]:
            for cat in panel_set.values():
                for w in cat.get("workers", []):
                    configured.add(w)
                configured.add(cat.get("synthesizer", ""))
                configured.add(cat.get("fallback_synthesizer", ""))
        configured.discard("")

        for name in sorted(configured):
            if is_cloud_model(name):
                continue
            if name not in local_models:
                logger.warning(
                    "⚠ Configured model NOT in Ollama: %s — requests using it will fail",
                    name,
                )
    except Exception as e:
        logger.warning("Model validation failed: %s", e)


# ══════════════════════════════════════════════════════════════════════════════
#  FastAPI app
# ══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app):
    state.ollama_session = aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=360),
        connector=aiohttp.TCPConnector(limit=20),
    )
    state.ext_session = aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=15)
    )

    state.tool_registry = ToolRegistry(session=state.ext_session)

    if TOOL_SERVER_URLS and TOOLS_ENABLED:
        await state.tool_registry.discover(TOOL_SERVER_URLS)
        logger.info(
            "Tools: %d discovered from %d servers",
            state.tool_registry.tool_count,
            len(TOOL_SERVER_URLS),
        )

    await validate_models()

    logger.info(
        "Audrey ready  router=%s  tools=%s(%d)  search=%s  fast_path=%s  "
        "react=%s  reflect=%s  plan=%s  escalate=%s  "
        "max_workers_local=%d  max_workers_cloud=%d",
        ROUTER_MODEL,
        TOOLS_ENABLED,
        state.tool_registry.tool_count,
        SEARCH_BACKEND,
        FAST_PATH_ENABLED,
        REACT_MAX_ROUNDS,
        REFLECTION_ENABLED,
        PLANNING_ENABLED,
        ESCALATION_ENABLED,
        MAX_DEEP_WORKERS,
        MAX_DEEP_WORKERS_CLOUD,
    )

    yield

    await state.ollama_session.close()
    await state.ext_session.close()


app = FastAPI(
    title="Audrey",
    version="7.0.0",
    lifespan=lifespan,
)


# ── Auth ─────────────────────────────────────────────────────────────────────

async def verify_api_key(req: Request):
    if not API_KEY:
        return
    auth = req.headers.get("Authorization", "")
    if not auth.startswith("Bearer ") or auth[7:] != API_KEY:
        raise HTTPException(401, "Invalid or missing API key")


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
async def healthcheck():
    try:
        async with state.ollama_session.get(
            f"{OLLAMA_BASE_URL}/api/tags",
            timeout=aiohttp.ClientTimeout(total=5),
        ) as r:
            if r.status == 200:
                return {
                    "ok": True,
                    "ollama": "reachable",
                    "router_model": ROUTER_MODEL,
                    "cache": cache.stats,
                    "tools": {
                        "enabled": TOOLS_ENABLED,
                        "count": state.tool_registry.tool_count
                        if state.tool_registry
                        else 0,
                        "servers": state.tool_registry.server_info
                        if state.tool_registry
                        else {},
                    },
                    "fast_path": FAST_PATH_ENABLED,
                    "agentic": {
                        "react_max_rounds": REACT_MAX_ROUNDS,
                        "reflection": REFLECTION_ENABLED,
                        "planning": PLANNING_ENABLED,
                        "escalation": ESCALATION_ENABLED,
                    },
                    "max_workers": {
                        "local": MAX_DEEP_WORKERS,
                        "cloud": MAX_DEEP_WORKERS_CLOUD,
                    },
                    "available_models": len(state.available_models),
                }
            return JSONResponse({"ok": False, "ollama": f"status {r.status}"}, status_code=503)
    except Exception as e:
        return JSONResponse({"ok": False, "ollama": f"unreachable: {e}"}, status_code=503)


@app.get("/v1/models", dependencies=[Depends(verify_api_key)])
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": m, "object": "model", "owned_by": "audrey"}
            for m in sorted(ALL_VIRTUAL_MODELS)
        ],
    }


@app.post("/v1/tools/rediscover", dependencies=[Depends(verify_api_key)])
async def rediscover_tools():
    if state.tool_registry and TOOL_SERVER_URLS:
        await state.tool_registry.rediscover(TOOL_SERVER_URLS)
        return {
            "tools": state.tool_registry.tool_count,
            "names": state.tool_registry.tool_names,
            "servers": state.tool_registry.server_info,
        }
    return {"tools": 0}


# ══════════════════════════════════════════════════════════════════════════════
#  Request dispatch
# ══════════════════════════════════════════════════════════════════════════════

async def run_graph_dispatch(req, *, stream_prepare_only=False):
    msgs_raw = [m.model_dump() for m in req.messages]
    temp = req.temperature if req.temperature is not None else DEFAULT_TEMPERATURE

    # Cache check
    if CACHE_ENABLED and not req.stream:
        cached = cache.get(msgs_raw, req.model, temp)
        if cached is not None:
            logger.info("Cache hit for model=%s", req.model)
            return {
                "request_id": str(uuid.uuid4()),
                "requested_model": req.model,
                "result_text": cached,
                "selected_model": "cache",
                "task_type": "cached",
                "confidence": 1.0,
                "route_reason": "Cache hit",
                "latency_ms": 0,
                "prompt_tokens": estimate_tokens(flatten_messages(msgs_raw)),
                "completion_tokens": estimate_tokens(cached),
                "search_performed": False,
                "use_fast_path": False,
                "escalated": False,
            }

    s = {
        "request_id": str(uuid.uuid4()),
        "requested_model": req.model,
        "messages": msgs_raw,
        "stream": bool(req.stream),
        "temperature": temp,
        "max_tokens": req.max_tokens,
        "top_p": req.top_p,
        "stop": req.stop,
        "frequency_penalty": req.frequency_penalty,
        "presence_penalty": req.presence_penalty,
        "errors": [],
        "started_at": time.time(),
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "search_performed": False,
        "search_query": "",
        "search_results": [],
        "use_fast_path": False,
        "fast_model": "",
        "sub_tasks": None,
        "react_rounds": 0,
        "reflection_result": {},
        "reflection_retries": 0,
        "escalated": False,
        "tools_used": [],
    }

    # Try fast path first for audrey_deep
    if (
        FAST_PATH_ENABLED
        and req.model == "audrey_deep"
        and not stream_prepare_only
    ):
        r = await FAST_GRAPH.ainvoke(s)

        if r.get("use_fast_path") and r.get("result_text") and not r.get("escalated"):
            r["latency_ms"] = int((time.time() - s["started_at"]) * 1000)
            if CACHE_ENABLED:
                cache.put(msgs_raw, req.model, temp, r["result_text"])
            logger.info(
                json.dumps(
                    {
                        "rid": r["request_id"],
                        "model": r["requested_model"],
                        "type": r.get("task_type"),
                        "conf": r.get("confidence"),
                        "selected": r.get("selected_model"),
                        "path": "fast+react",
                        "react_rounds": r.get("react_rounds", 0),
                        "reflection": r.get("reflection_result", {}).get(
                            "quality", "n/a"
                        ),
                        "search": r.get("search_performed", False),
                        "search_query": r.get("search_query", ""),
                        "tools": [
                            t.get("tool", "") for t in r.get("tools_used", [])
                        ],
                        "ms": r["latency_ms"],
                    }
                )
            )
            return r

        # Fast path was skipped, failed, or escalated — fall through to deep panel
        s = r
        s["started_at"] = time.time()
        if r.get("escalated"):
            logger.info(
                "Escalated from fast→deep: %s", r.get("route_reason", "")
            )

    # Deep panel path
    g = PREP_GRAPH if stream_prepare_only else SYNTH_GRAPH
    r = await g.ainvoke(s)
    r["latency_ms"] = int((time.time() - s["started_at"]) * 1000)

    if CACHE_ENABLED and not stream_prepare_only and r.get("result_text"):
        cache.put(msgs_raw, req.model, temp, r["result_text"])

    logger.info(
        json.dumps(
            {
                "rid": r["request_id"],
                "model": r["requested_model"],
                "type": r.get("task_type"),
                "conf": r.get("confidence"),
                "selected": r.get("selected_model", r.get("synthesizer")),
                "path": "deep",
                "planned": bool(r.get("sub_tasks")),
                "reflection": r.get("reflection_result", {}).get("quality", "n/a"),
                "search": r.get("search_performed", False),
                "search_query": r.get("search_query", ""),
                "tools": [t.get("tool", "") for t in r.get("tools_used", [])],
                "escalated": r.get("escalated", False),
                "ms": r["latency_ms"],
            }
        )
    )
    return r


# ── Main endpoint ────────────────────────────────────────────────────────────

@app.post("/v1/chat/completions", dependencies=[Depends(verify_api_key)])
async def chat_completions(req: ChatCompletionRequest):
    if req.model not in ALL_VIRTUAL_MODELS:
        raise HTTPException(400, f"Unknown model: {req.model}")
    if not req.messages:
        raise HTTPException(400, "messages empty")
    if not any(m.role == "user" for m in req.messages):
        raise HTTPException(400, "No user message")
    if req.max_tokens and req.max_tokens > 128000:
        raise HTTPException(400, "max_tokens too large")

    # ── Streaming ────────────────────────────────────────────────────────
    if req.stream:

        async def _stream():
            ct = int(time.time())
            rid = f"chatcmpl-{uuid.uuid4().hex[:24]}"

            if EMIT_STATUS_UPDATES:
                yield _sc(rid, ct, req.model, "🔍 Analyzing...\n")

            init_state = {
                "request_id": str(uuid.uuid4()),
                "requested_model": req.model,
                "messages": [m.model_dump() for m in req.messages],
                "stream": True,
                "temperature": (
                    req.temperature
                    if req.temperature is not None
                    else DEFAULT_TEMPERATURE
                ),
                "max_tokens": req.max_tokens,
                "top_p": req.top_p,
                "stop": req.stop,
                "frequency_penalty": req.frequency_penalty,
                "presence_penalty": req.presence_penalty,
                "errors": [],
                "started_at": time.time(),
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "search_performed": False,
                "search_query": "",
                "search_results": [],
                "use_fast_path": False,
                "fast_model": "",
                "sub_tasks": None,
                "react_rounds": 0,
                "reflection_result": {},
                "reflection_retries": 0,
                "escalated": False,
                "tools_used": [],
            }

            # Classify (also initializes search fields and original_messages)
            classified = await node_classify(init_state)

            # Decide path — search is handled by tool-calling during generation
            if classified.get("use_fast_path") and classified.get("fast_model"):
                async for chunk in stream_fast_path(classified):
                    yield chunk
            else:
                # Deep panel
                planned = await node_plan_panel(classified)
                if planned.get("sub_tasks"):
                    yield _sc(
                        rid,
                        ct,
                        req.model,
                        f"📋 Planning: {len(planned['sub_tasks'])} sub-tasks\n",
                    )
                generated = await node_parallel_generate(planned)
                prepared = await node_prepare_synthesis(generated)
                async for chunk in stream_synthesis(prepared):
                    yield chunk

        return StreamingResponse(_stream(), media_type="text/event-stream")

    # ── Non-streaming ────────────────────────────────────────────────────
    final = await run_graph_dispatch(req)
    content = final["result_text"]
    if EMIT_ROUTING_BANNER:
        content = banner(final) + content
    return JSONResponse(
        {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": final.get("prompt_tokens", 0),
                "completion_tokens": final.get("completion_tokens", 0),
                "total_tokens": (
                    final.get("prompt_tokens", 0)
                    + final.get("completion_tokens", 0)
                ),
            },
        }
    )
