"""
Audrey — LangGraph Auto-Router

FastAPI application, lifespan, endpoints, and request dispatch.
All logic lives in dedicated modules; this file wires them together.
"""

import asyncio
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager, suppress
from typing import Any, AsyncGenerator, Awaitable

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
    EMIT_TIMELINE_EVENTS,
    EMIT_TRUST_SIGNALS,
    ESCALATION_ENABLED,
    FAST_PATH_ENABLED,
    FAST_PATH_CONFIDENCE,
    MAX_DEEP_WORKERS,
    MAX_DEEP_WORKERS_CLOUD,
    OLLAMA_BASE_URL,
    PLANNING_ENABLED,
    REACT_MAX_ROUNDS,
    REFLECTION_ENABLED,
    REFLECTION_MAX_RETRIES,
    ROUTER_MODEL,
    SEARCH_BACKEND,
    STREAM_HEARTBEAT_SECONDS,
    TOOL_SERVER_URLS,
    TOOLS_ENABLED,
    is_cloud_model,
)
from helpers import estimate_tokens, flatten_messages
from helpers import (
    append_timeline_event,
    build_initial_state,
    build_trust_signals,
    get_last_user_text,
    is_time_sensitive_query,
    normalize_audrey_mode,
)
from models import ChatCompletionRequest
from pipeline import (
    FAST_GRAPH,
    node_classify,
    node_parallel_generate,
    node_plan_panel,
    node_prepare_synthesis,
    node_reflect_deep,
    node_synthesize,
)
from streaming import _SSE_DONE, _sc, _sc_event, _sc_stop, banner, stream_synthesis
from tool_registry import ToolRegistry

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("audrey")
_STREAM_STAGE_DONE = object()
_FAST_ONLY_FALLBACK_TEXT = (
    "audrey_fast could not complete this request in fast mode. "
    "Try again, or use audrey_deep for full deep-panel fallback."
)
_STREAM_RESULT_CHUNK_CHARS = 1200


def _log_safe_query(query: str | None) -> str:
    """Return a log-safe version of a search query (masks content, keeps length)."""
    q = str(query or "").strip()
    return f"[redacted:{len(q)} chars]" if q else ""


def _apply_mode_overrides(state_obj: dict[str, Any]) -> dict[str, Any]:
    mode = normalize_audrey_mode(state_obj.get("audrey_mode"))
    state_obj["audrey_mode"] = mode

    # Balanced defaults.
    state_obj["fast_path_confidence"] = FAST_PATH_CONFIDENCE
    state_obj["force_deep_profile"] = False
    state_obj["planning_enabled_override"] = PLANNING_ENABLED
    state_obj["planning_min_tokens_override"] = None
    state_obj["reflection_enabled_override"] = REFLECTION_ENABLED
    state_obj["reflection_max_retries_override"] = REFLECTION_MAX_RETRIES
    state_obj["react_max_rounds_override"] = REACT_MAX_ROUNDS

    if mode == "quick":
        state_obj["fast_path_confidence"] = min(FAST_PATH_CONFIDENCE, 0.62)
        state_obj["planning_enabled_override"] = False
        state_obj["reflection_enabled_override"] = False
        state_obj["reflection_max_retries_override"] = 0
        state_obj["react_max_rounds_override"] = 1
    elif mode == "research":
        state_obj["fast_path_confidence"] = max(FAST_PATH_CONFIDENCE, 0.90)
        state_obj["force_deep_profile"] = state_obj.get("requested_model") == "audrey_deep"
        state_obj["planning_enabled_override"] = True
        state_obj["planning_min_tokens_override"] = 40
        state_obj["reflection_enabled_override"] = True
        state_obj["reflection_max_retries_override"] = max(2, REFLECTION_MAX_RETRIES)
        state_obj["react_max_rounds_override"] = max(4, REACT_MAX_ROUNDS)
    return state_obj


def _timeline(
    state_obj: dict[str, Any],
    *,
    stage: str,
    message: str,
    status: str = "info",
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return append_timeline_event(
        state_obj,
        stage=stage,
        message=message,
        status=status,
        details=details,
    )


async def _run_deep_sequence(
    state_obj: dict[str, Any],
    *,
    include_synthesis: bool,
) -> dict[str, Any]:
    s = state_obj
    if not s.get("task_type"):
        s = await node_classify(s)
        _timeline(
            s,
            stage="classify",
            message="Request classified for deep-panel route",
            details={"task_type": s.get("task_type"), "confidence": s.get("confidence")},
        )

    s = await node_plan_panel(s)
    _timeline(
        s,
        stage="plan",
        message="Deep-panel workers and sub-tasks selected",
        details={
            "workers": len(s.get("deep_workers", [])),
            "planned_sub_tasks": len(s.get("sub_tasks") or []),
        },
    )

    s = await node_parallel_generate(s)
    _timeline(
        s,
        stage="generate",
        message="Worker drafts generated",
        details={"outputs": len(s.get("worker_outputs", []))},
    )

    s = await node_prepare_synthesis(s)
    _timeline(
        s,
        stage="prepare_synthesis",
        message="Synthesis context prepared",
    )

    if include_synthesis:
        s = await node_synthesize(s)
        _timeline(
            s,
            stage="synthesize",
            message="Drafts synthesized into final answer",
            details={"selected_model": s.get("selected_model", s.get("synthesizer", ""))},
        )
        s = await node_reflect_deep(s)
        rr = s.get("reflection_result", {})
        if rr:
            _timeline(
                s,
                stage="reflection",
                message="Final answer evaluated by reflection gate",
                details={"quality": rr.get("quality", "n/a")},
            )
    return s


async def _await_stream_stage(
    awaitable: Awaitable[Any],
    *,
    rid: str,
    created: int,
    model_name: str,
    request_id: str,
    stage: str,
    heartbeat_text: str,
) -> AsyncGenerator[tuple[str | None, Any], None]:
    started = time.monotonic()
    emitted_heartbeat_text = False
    logger.info(
        "Streaming stage started rid=%s stage=%s model=%s",
        request_id,
        stage,
        model_name,
    )

    if STREAM_HEARTBEAT_SECONDS <= 0:
        result = await awaitable
        elapsed_ms = int((time.monotonic() - started) * 1000)
        logger.info(
            "Streaming stage finished rid=%s stage=%s ms=%d",
            request_id,
            stage,
            elapsed_ms,
        )
        yield None, result
        return

    task = asyncio.create_task(awaitable)
    try:
        while True:
            try:
                result = await asyncio.wait_for(
                    asyncio.shield(task),
                    timeout=STREAM_HEARTBEAT_SECONDS,
                )
                elapsed_ms = int((time.monotonic() - started) * 1000)
                logger.info(
                    "Streaming stage finished rid=%s stage=%s ms=%d",
                    request_id,
                    stage,
                    elapsed_ms,
                )
                if emitted_heartbeat_text and EMIT_STATUS_UPDATES:
                    # Finalize the in-place heartbeat row before next stage output.
                    yield (_sc(rid, created, model_name, "\n"), _STREAM_STAGE_DONE)
                yield None, result
                return
            except asyncio.TimeoutError:
                elapsed_s = max(1, int(time.monotonic() - started))
                logger.info(
                    "Streaming stage pending rid=%s stage=%s elapsed_s=%d",
                    request_id,
                    stage,
                    elapsed_s,
                )
                if EMIT_TIMELINE_EVENTS:
                    heartbeat_event = {
                        "stage": stage,
                        "status": "pending",
                        "message": heartbeat_text,
                        "details": {
                            "elapsed_s": elapsed_s,
                            # Clients can use this key to replace one banner/timer row
                            # instead of appending a new line per heartbeat tick.
                            "replace": True,
                            "replace_key": f"heartbeat:{stage}",
                        },
                    }
                    yield (
                        _sc_event(rid, created, model_name, heartbeat_event),
                        _STREAM_STAGE_DONE,
                    )
                if EMIT_STATUS_UPDATES:
                    emitted_heartbeat_text = True
                    yield (
                        _sc(
                            rid,
                            created,
                            model_name,
                            f"\r⏳ {heartbeat_text} ({elapsed_s}s)",
                        ),
                        _STREAM_STAGE_DONE,
                    )
    except Exception:
        elapsed_ms = int((time.monotonic() - started) * 1000)
        logger.exception(
            "Streaming stage failed rid=%s stage=%s ms=%d",
            request_id,
            stage,
            elapsed_ms,
        )
        if not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        raise


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
                    "ux": {
                        "timeline_events": EMIT_TIMELINE_EVENTS,
                        "trust_signals": EMIT_TRUST_SIGNALS,
                    },
                    "max_workers": {
                        "local": MAX_DEEP_WORKERS,
                        "cloud": MAX_DEEP_WORKERS_CLOUD,
                    },
                    "available_models": len(state.available_models),
                    "audrey_fast": dict(state.audrey_fast_health),
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
    mode = normalize_audrey_mode(getattr(req, "audrey_mode", "balanced"))
    cache_kwargs = dict(
        max_tokens=req.max_tokens,
        top_p=req.top_p,
        stop=req.stop,
        frequency_penalty=req.frequency_penalty,
        presence_penalty=req.presence_penalty,
    )

    s = build_initial_state(
        request_id=str(uuid.uuid4()),
        requested_model=req.model,
        messages=msgs_raw,
        audrey_mode=mode,
        stream=bool(req.stream),
        temperature=temp,
        max_tokens=req.max_tokens,
        top_p=req.top_p,
        stop=req.stop,
        frequency_penalty=req.frequency_penalty,
        presence_penalty=req.presence_penalty,
    )
    _apply_mode_overrides(s)
    s["needs_fresh_data"] = is_time_sensitive_query(get_last_user_text(msgs_raw))
    _timeline(
        s,
        stage="request_received",
        message="Request accepted",
        details={"mode": s["audrey_mode"], "stream": bool(req.stream)},
    )

    # Cache check (skipped for time-sensitive requests to avoid stale responses).
    cache_allowed = (
        CACHE_ENABLED
        and not req.stream
        and not stream_prepare_only
        and not s["needs_fresh_data"]
    )
    if cache_allowed:
        cached = cache.get(msgs_raw, req.model, temp, **cache_kwargs)
        if cached is not None:
            logger.info("Cache hit for model=%s", req.model)
            s.update(
                {
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
                    "cache_hit": True,
                }
            )
            _timeline(
                s,
                stage="cache",
                message="Response served from cache",
                status="success",
            )
            return s
    elif CACHE_ENABLED and not req.stream and s["needs_fresh_data"]:
        _timeline(
            s,
            stage="cache",
            message="Cache bypassed for freshness-sensitive query",
            details={"reason": "time_sensitive_query"},
        )

    # Try fast path for audrey_deep (auto) and audrey_fast (fast-only).
    if (
        FAST_PATH_ENABLED
        and req.model in {"audrey_deep", "audrey_fast"}
        and not stream_prepare_only
    ):
        _timeline(
            s,
            stage="fast_path",
            message="Evaluating fast path",
            details={"mode": s["audrey_mode"]},
        )
        r = await FAST_GRAPH.ainvoke(s)
        r.setdefault("timeline", s.get("timeline", []))
        r["audrey_mode"] = s["audrey_mode"]
        r["needs_fresh_data"] = s["needs_fresh_data"]

        if r.get("use_fast_path") and r.get("result_text") and not r.get("escalated"):
            r["latency_ms"] = int((time.time() - s["started_at"]) * 1000)
            if CACHE_ENABLED and not r.get("needs_fresh_data", False):
                cache.put(msgs_raw, req.model, temp, r["result_text"], **cache_kwargs)
            if req.model == "audrey_fast":
                state.update_audrey_fast_health(
                    selected_model=r.get("selected_model", r.get("fast_model", "none")),
                    success=True,
                    reason=r.get("route_reason", "fast_only:completed"),
                )
            logger.info(
                json.dumps(
                    {
                        "rid": r["request_id"],
                        "model": r["requested_model"],
                        "type": r.get("task_type"),
                        "conf": r.get("confidence"),
                        "selected": r.get("selected_model"),
                        "path": (
                            "fast-only"
                            if req.model == "audrey_fast"
                            else "fast+react"
                        ),
                        "react_rounds": r.get("react_rounds", 0),
                        "reflection": r.get("reflection_result", {}).get(
                            "quality", "n/a"
                        ),
                        "search": r.get("search_performed", False),
                        "search_query": _log_safe_query(
                            r.get("search_query", "")
                        ),
                        "tools": [
                            t.get("tool", "") for t in r.get("tools_used", [])
                        ],
                        "ms": r["latency_ms"],
                    }
                )
            )
            _timeline(
                r,
                stage="complete",
                message="Completed on fast path",
                status="success",
                details={"selected_model": r.get("selected_model", r.get("fast_model", ""))},
            )
            return r

        if req.model == "audrey_fast":
            r["latency_ms"] = int((time.time() - s["started_at"]) * 1000)
            r["selected_model"] = r.get("selected_model", r.get("fast_model", "none"))
            r["result_text"] = r.get("result_text") or _FAST_ONLY_FALLBACK_TEXT
            r["prompt_tokens"] = r.get(
                "prompt_tokens", estimate_tokens(flatten_messages(msgs_raw))
            )
            r["completion_tokens"] = estimate_tokens(r["result_text"])
            r["escalated"] = False
            state.update_audrey_fast_health(
                selected_model=r.get("selected_model", "none"),
                success=False,
                reason=r.get("route_reason", "fast_only:fallback"),
            )
            logger.info(
                json.dumps(
                    {
                        "rid": r["request_id"],
                        "model": r["requested_model"],
                        "type": r.get("task_type"),
                        "conf": r.get("confidence"),
                        "selected": r.get("selected_model"),
                        "path": "fast-only",
                        "fallback": True,
                        "reason": r.get("route_reason", ""),
                        "ms": r["latency_ms"],
                    }
                )
            )
            _timeline(
                r,
                stage="complete",
                message="Fast-only request fell back to user-visible failure message",
                status="warning",
            )
            return r

        # Fast path was skipped, failed, or escalated — fall through to deep panel
        s = {**r, "started_at": time.time()}
        s.setdefault("timeline", r.get("timeline", []))
        if s.get("escalated"):
            logger.info(
                "Escalated from fast→deep: %s", s.get("route_reason", "")
            )
            _timeline(
                s,
                stage="escalation",
                message="Escalated from fast path to deep panel",
                status="warning",
                details={"reason": s.get("route_reason", "")},
            )

    # audrey_fast is fast-only and must never run the deep panel.
    if req.model == "audrey_fast":
        elapsed_ms = int((time.time() - s["started_at"]) * 1000)
        state.update_audrey_fast_health(
            selected_model="none",
            success=False,
            reason="fast_only:disabled_or_unavailable",
        )
        logger.info(
            "audrey_fast fast-only fallback (fast_path_enabled=%s, prepare_only=%s)",
            FAST_PATH_ENABLED,
            stream_prepare_only,
        )
        _timeline(
            s,
            stage="complete",
            message="Fast-only request returned fallback message",
            status="warning",
        )
        return {
            "request_id": s["request_id"],
            "requested_model": req.model,
            "result_text": _FAST_ONLY_FALLBACK_TEXT,
            "selected_model": "none",
            "task_type": s.get("task_type", "general"),
            "confidence": s.get("confidence", 0.0),
            "route_reason": "fast_only:disabled_or_unavailable",
            "latency_ms": elapsed_ms,
            "prompt_tokens": estimate_tokens(flatten_messages(msgs_raw)),
            "completion_tokens": estimate_tokens(_FAST_ONLY_FALLBACK_TEXT),
            "search_performed": False,
            "use_fast_path": False,
            "escalated": False,
            "cache_hit": False,
            "audrey_mode": s.get("audrey_mode", mode),
            "needs_fresh_data": s.get("needs_fresh_data", False),
            "timeline": s.get("timeline", []),
        }

    # Deep panel path (single classify pass, even after fast->deep escalation).
    _timeline(
        s,
        stage="deep_path",
        message="Running deep-panel pipeline",
    )
    r = await _run_deep_sequence(s, include_synthesis=not stream_prepare_only)
    r["latency_ms"] = int((time.time() - s["started_at"]) * 1000)

    if (
        CACHE_ENABLED
        and not stream_prepare_only
        and r.get("result_text")
        and not r.get("needs_fresh_data", False)
    ):
        cache.put(msgs_raw, req.model, temp, r["result_text"], **cache_kwargs)
    r["cache_hit"] = bool(r.get("cache_hit", False))
    _timeline(
        r,
        stage="complete",
        message="Completed on deep panel",
        status="success",
        details={"selected_model": r.get("selected_model", r.get("synthesizer", ""))},
    )

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
                "search_query": _log_safe_query(r.get("search_query", "")),
                "tools": [t.get("tool", "") for t in r.get("tools_used", [])],
                "escalated": r.get("escalated", False),
                "mode": r.get("audrey_mode", mode),
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
    if req.temperature is not None and not (0 <= req.temperature <= 2):
        raise HTTPException(400, "temperature must be between 0 and 2")
    if req.top_p is not None and not (0 <= req.top_p <= 1):
        raise HTTPException(400, "top_p must be between 0 and 1")
    if req.frequency_penalty is not None and not (-2 <= req.frequency_penalty <= 2):
        raise HTTPException(400, "frequency_penalty must be between -2 and 2")
    if req.presence_penalty is not None and not (-2 <= req.presence_penalty <= 2):
        raise HTTPException(400, "presence_penalty must be between -2 and 2")
    if req.max_tokens is not None and req.max_tokens <= 0:
        raise HTTPException(400, "max_tokens must be positive")
    if req.max_tokens is not None and req.max_tokens > 128000:
        raise HTTPException(400, "max_tokens too large")

    # ── Streaming ────────────────────────────────────────────────────────
    if req.stream:

        async def _stream():
            ct = int(time.time())
            request_id = str(uuid.uuid4())
            rid = f"chatcmpl-{request_id.replace('-', '')[:24]}"
            logger.info(
                "Streaming request started rid=%s model=%s messages=%d",
                request_id,
                req.model,
                len(req.messages),
            )
            try:
                if EMIT_STATUS_UPDATES:
                    yield _sc(rid, ct, req.model, "🔍 Analyzing...\n")

                init_state = build_initial_state(
                    request_id=request_id,
                    requested_model=req.model,
                    messages=[m.model_dump() for m in req.messages],
                    audrey_mode=normalize_audrey_mode(req.audrey_mode),
                    stream=True,
                    temperature=(
                        req.temperature
                        if req.temperature is not None
                        else DEFAULT_TEMPERATURE
                    ),
                    max_tokens=req.max_tokens,
                    top_p=req.top_p,
                    stop=req.stop,
                    frequency_penalty=req.frequency_penalty,
                    presence_penalty=req.presence_penalty,
                )
                init_state = _apply_mode_overrides(init_state)
                init_state["needs_fresh_data"] = is_time_sensitive_query(
                    get_last_user_text(init_state["messages"])
                )
                start_evt = _timeline(
                    init_state,
                    stage="request_received",
                    message="Streaming request accepted",
                    details={"mode": init_state.get("audrey_mode", "balanced")},
                )
                if EMIT_TIMELINE_EVENTS:
                    yield _sc_event(rid, ct, req.model, start_evt)

                def _emit_timeline_event(
                    state_obj: dict[str, Any],
                    *,
                    stage: str,
                    message: str,
                    status: str = "info",
                    details: dict[str, Any] | None = None,
                ) -> str | None:
                    event = _timeline(
                        state_obj,
                        stage=stage,
                        message=message,
                        status=status,
                        details=details,
                    )
                    if EMIT_TIMELINE_EVENTS:
                        return _sc_event(rid, ct, req.model, event)
                    return None

                # Classify (also initializes search fields and original_messages)
                if EMIT_STATUS_UPDATES:
                    yield _sc(rid, ct, req.model, "🧭 Routing request...\n")

                classified = None
                async for chunk, result in _await_stream_stage(
                    node_classify(init_state),
                    rid=rid,
                    created=ct,
                    model_name=req.model,
                    request_id=request_id,
                    stage="classify",
                    heartbeat_text="Still analyzing request",
                ):
                    if chunk is not None:
                        yield chunk
                    else:
                        classified = result

                if classified is None:
                    raise RuntimeError("Classification stage completed without a result")
                classification_evt = _emit_timeline_event(
                    classified,
                    stage="classify",
                    message="Request classified",
                    details={
                        "task_type": classified.get("task_type"),
                        "confidence": classified.get("confidence"),
                        "use_fast_path": bool(classified.get("use_fast_path")),
                    },
                )
                if classification_evt:
                    yield classification_evt

                logger.info(
                    "Streaming route decided rid=%s path=%s task_type=%s conf=%.2f fast_model=%s",
                    request_id,
                    (
                        "fast+react"
                        if classified.get("use_fast_path")
                        and classified.get("fast_model")
                        else "fast-only-fallback"
                        if req.model == "audrey_fast"
                        else "deep"
                    ),
                    classified.get("task_type"),
                    classified.get("confidence", 0.0),
                    classified.get("fast_model", ""),
                )

                # Decide path — search is handled by tool-calling during generation
                if classified.get("use_fast_path") and classified.get("fast_model"):
                    if EMIT_STATUS_UPDATES:
                        yield _sc(
                            rid,
                            ct,
                            req.model,
                            "⚡ Running fast path with tools/search...\n",
                        )

                    fast_result = None
                    async for chunk, result in _await_stream_stage(
                        FAST_GRAPH.ainvoke(classified),
                        rid=rid,
                        created=ct,
                        model_name=req.model,
                        request_id=request_id,
                        stage="fast_graph",
                        heartbeat_text="Still running fast path (tools/search)",
                    ):
                        if chunk is not None:
                            yield chunk
                        else:
                            fast_result = result

                    if fast_result is None:
                        raise RuntimeError("Fast graph completed without a result")

                    if (
                        fast_result.get("use_fast_path")
                        and fast_result.get("result_text")
                        and not fast_result.get("escalated")
                    ):
                        if req.model == "audrey_fast":
                            state.update_audrey_fast_health(
                                selected_model=fast_result.get(
                                    "selected_model",
                                    fast_result.get("fast_model", "none"),
                                ),
                                success=True,
                                reason=fast_result.get(
                                    "route_reason", "fast_only:completed"
                                ),
                            )
                        if EMIT_STATUS_UPDATES:
                            yield _sc(
                                rid,
                                ct,
                                req.model,
                                f"⚡ Fast path complete: {fast_result.get('selected_model', fast_result.get('fast_model', '?'))}\n\n",
                            )
                            if fast_result.get("search_performed"):
                                search_query = str(
                                    fast_result.get("search_query", "")
                                ).strip()
                                if search_query:
                                    yield _sc(
                                        rid,
                                        ct,
                                        req.model,
                                        f"🌐 Web search used: {search_query}\n",
                                    )
                                else:
                                    yield _sc(rid, ct, req.model, "🌐 Web search used\n")
                        if EMIT_ROUTING_BANNER:
                            yield _sc(rid, ct, req.model, banner(fast_result))
                        fast_complete_evt = _emit_timeline_event(
                            fast_result,
                            stage="complete",
                            message="Completed on fast path",
                            status="success",
                            details={
                                "selected_model": fast_result.get(
                                    "selected_model",
                                    fast_result.get("fast_model", ""),
                                )
                            },
                        )
                        if fast_complete_evt:
                            yield fast_complete_evt

                        content = fast_result["result_text"]
                        if content:
                            for i in range(0, len(content), _STREAM_RESULT_CHUNK_CHARS):
                                yield _sc(
                                    rid,
                                    ct,
                                    req.model,
                                    content[i : i + _STREAM_RESULT_CHUNK_CHARS],
                                )
                        if EMIT_TRUST_SIGNALS:
                            yield _sc_event(
                                rid,
                                ct,
                                req.model,
                                {
                                    "stage": "trust",
                                    "status": "info",
                                    "message": "Trust signals",
                                    "details": build_trust_signals(fast_result),
                                },
                            )

                        yield _sc_stop(rid, ct, req.model)
                        yield _SSE_DONE
                        return

                    if req.model == "audrey_fast":
                        state.update_audrey_fast_health(
                            selected_model=fast_result.get("selected_model", "none"),
                            success=False,
                            reason=fast_result.get("route_reason", "fast_only:fallback"),
                        )
                        fallback_evt = _emit_timeline_event(
                            fast_result,
                            stage="complete",
                            message="Fast-only request returned fallback message",
                            status="warning",
                        )
                        if fallback_evt:
                            yield fallback_evt
                        yield _sc(rid, ct, req.model, f"{_FAST_ONLY_FALLBACK_TEXT}\n")
                        yield _sc_stop(rid, ct, req.model)
                        yield _SSE_DONE
                        return

                    # Fast path was skipped/failed/escalated in audrey_deep, so continue to deep panel.
                    classified = fast_result
                    if classified.get("escalated"):
                        esc_evt = _emit_timeline_event(
                            classified,
                            stage="escalation",
                            message="Escalated from fast path to deep panel",
                            status="warning",
                            details={"reason": classified.get("route_reason", "")},
                        )
                        if esc_evt:
                            yield esc_evt
                elif req.model == "audrey_fast":
                    state.update_audrey_fast_health(
                        selected_model=classified.get("fast_model", "none"),
                        success=False,
                        reason=classified.get("route_reason", "fast_only:fallback"),
                    )
                    fallback_evt = _emit_timeline_event(
                        classified,
                        stage="complete",
                        message="Fast-only request returned fallback message",
                        status="warning",
                    )
                    if fallback_evt:
                        yield fallback_evt
                    yield _sc(rid, ct, req.model, f"{_FAST_ONLY_FALLBACK_TEXT}\n")
                    yield _sc_stop(rid, ct, req.model)
                    yield _SSE_DONE
                    return

                # Deep panel (audrey_deep)
                deep_evt = _emit_timeline_event(
                    classified,
                    stage="deep_path",
                    message="Running deep-panel pipeline",
                )
                if deep_evt:
                    yield deep_evt
                if EMIT_STATUS_UPDATES:
                    yield _sc(rid, ct, req.model, "📋 Planning approach...\n")

                planned = None
                async for chunk, result in _await_stream_stage(
                    node_plan_panel(classified),
                    rid=rid,
                    created=ct,
                    model_name=req.model,
                    request_id=request_id,
                    stage="plan",
                    heartbeat_text="Still planning deep-panel approach",
                ):
                    if chunk is not None:
                        yield chunk
                    else:
                        planned = result

                if planned is None:
                    raise RuntimeError("Planning stage completed without a result")
                plan_evt = _emit_timeline_event(
                    planned,
                    stage="plan",
                    message="Deep-panel plan ready",
                    details={
                        "workers": len(planned.get("deep_workers", [])),
                        "planned_sub_tasks": len(planned.get("sub_tasks") or []),
                    },
                )
                if plan_evt:
                    yield plan_evt

                if EMIT_STATUS_UPDATES and planned.get("sub_tasks"):
                    yield _sc(
                        rid,
                        ct,
                        req.model,
                        f"📋 Planning: {len(planned['sub_tasks'])} sub-tasks\n",
                    )
                if EMIT_STATUS_UPDATES:
                    yield _sc(
                        rid,
                        ct,
                        req.model,
                        f"🧠 Generating drafts from {len(planned.get('deep_workers', []))} models...\n",
                    )

                generated = None
                async for chunk, result in _await_stream_stage(
                    node_parallel_generate(planned),
                    rid=rid,
                    created=ct,
                    model_name=req.model,
                    request_id=request_id,
                    stage="parallel_generate",
                    heartbeat_text="Still generating worker drafts",
                ):
                    if chunk is not None:
                        yield chunk
                    else:
                        generated = result

                if generated is None:
                    raise RuntimeError(
                        "Parallel generation stage completed without a result"
                    )
                generate_evt = _emit_timeline_event(
                    generated,
                    stage="generate",
                    message="Worker drafts generated",
                    details={"outputs": len(generated.get("worker_outputs", []))},
                )
                if generate_evt:
                    yield generate_evt

                if EMIT_STATUS_UPDATES:
                    yield _sc(rid, ct, req.model, "🧩 Preparing synthesis...\n")

                prepared = None
                async for chunk, result in _await_stream_stage(
                    node_prepare_synthesis(generated),
                    rid=rid,
                    created=ct,
                    model_name=req.model,
                    request_id=request_id,
                    stage="prepare_synthesis",
                    heartbeat_text="Still preparing synthesis",
                ):
                    if chunk is not None:
                        yield chunk
                    else:
                        prepared = result

                if prepared is None:
                    raise RuntimeError(
                        "Synthesis preparation stage completed without a result"
                    )
                prep_evt = _emit_timeline_event(
                    prepared,
                    stage="prepare_synthesis",
                    message="Synthesis context prepared",
                )
                if prep_evt:
                    yield prep_evt
                if EMIT_TRUST_SIGNALS:
                    yield _sc_event(
                        rid,
                        ct,
                        req.model,
                        {
                            "stage": "trust",
                            "status": "info",
                            "message": "Trust signals",
                            "details": build_trust_signals(prepared),
                        },
                    )

                async for chunk in stream_synthesis(prepared, rid=rid, created=ct):
                    yield chunk
            except Exception as e:
                logger.exception(
                    "Streaming request failed rid=%s model=%s: %s",
                    request_id,
                    req.model,
                    e,
                )
                error_text = str(e).strip() or "internal error"
                yield _sc(rid, ct, req.model, f"[Error: {error_text[:240]}]\n")
                yield _sc_stop(rid, ct, req.model)
                yield _SSE_DONE

        return StreamingResponse(
            _stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    # ── Non-streaming ────────────────────────────────────────────────────
    final = await run_graph_dispatch(req)
    content = final["result_text"]
    if EMIT_ROUTING_BANNER:
        content = banner(final) + content
    audrey_meta: dict[str, Any] = {
        "mode": final.get("audrey_mode", normalize_audrey_mode(req.audrey_mode)),
        "timeline": final.get("timeline", []),
        "needs_fresh_data": bool(final.get("needs_fresh_data", False)),
    }
    if EMIT_TRUST_SIGNALS:
        audrey_meta["trust"] = build_trust_signals(final)
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
            "audrey": audrey_meta,
        }
    )
