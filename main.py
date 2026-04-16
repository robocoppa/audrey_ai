"""
Audrey — Auto-Router

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
from helpers import (
    append_timeline_event,
    build_initial_state,
    build_trust_signals,
    estimate_tokens,
    flatten_messages,
    get_last_user_text,
    is_time_sensitive_query,
    normalize_audrey_mode,
)
from models import ChatCompletionRequest
from pipeline import (
    node_adaptive_escalate,
    node_classify,
    node_parallel_generate,
    node_plan_panel,
    node_prepare_synthesis,
    node_react_agent,
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


async def _orchestrate(
    s: dict[str, Any],
    *,
    req_model: str,
    include_synthesis: bool = True,
) -> AsyncGenerator[tuple[str, dict[str, Any], dict[str, Any]], None]:
    """Shared pipeline orchestration.

    Yields ``(stage, state, hints)`` before and after each step.
    ``stage`` is ``"starting:<name>"`` or ``"done:<name>"``.
    ``hints`` carries per-stage metadata for UX (heartbeat style, etc.).

    Consumers:
      - ``run_graph_dispatch`` ignores ``starting:`` yields.
      - ``_stream()`` emits SSE status/heartbeat between stages.
    """

    # ── Classify (runs exactly once) ────────────────────────────────────
    yield ("starting:classify", s, {})
    s = await node_classify(s)
    _timeline(
        s,
        stage="classify",
        message="Request classified",
        details={
            "task_type": s.get("task_type"),
            "confidence": s.get("confidence"),
            "use_fast_path": bool(s.get("use_fast_path")),
        },
    )
    yield ("done:classify", s, {})

    # ── Fast path (audrey_deep auto + audrey_fast) ──────────────────────
    fast_eligible = (
        FAST_PATH_ENABLED
        and req_model in {"audrey_deep", "audrey_fast"}
    )
    if fast_eligible and s.get("use_fast_path") and s.get("fast_model"):
        yield ("starting:fast_react", s, {})
        s = await node_react_agent(s)
        s = await node_adaptive_escalate(s)
        _timeline(
            s,
            stage="fast_react",
            message="Fast path completed",
            details={
                "selected_model": s.get("selected_model", s.get("fast_model", "")),
                "escalated": bool(s.get("escalated")),
            },
        )
        yield ("done:fast_react", s, {})

        if s.get("use_fast_path") and s.get("result_text") and not s.get("escalated"):
            yield ("done:complete_fast", s, {})
            return

        if req_model == "audrey_fast":
            yield ("done:fast_only_fallback", s, {})
            return

        if s.get("escalated"):
            _timeline(
                s,
                stage="escalation",
                message="Escalated from fast path to deep panel",
                status="warning",
                details={"reason": s.get("route_reason", "")},
            )
            yield ("done:escalation", s, {})

    elif req_model == "audrey_fast":
        yield ("done:fast_only_fallback", s, {})
        return

    # ── Deep panel ──────────────────────────────────────────────────────
    yield ("starting:plan", s, {})
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
    yield ("done:plan", s, {})

    yield ("starting:generate", s, {"heartbeat_style": "dots"})
    s = await node_parallel_generate(s)
    _timeline(
        s,
        stage="generate",
        message="Worker drafts generated",
        details={"outputs": len(s.get("worker_outputs", []))},
    )
    yield ("done:generate", s, {})

    yield ("starting:prepare_synthesis", s, {})
    s = await node_prepare_synthesis(s)
    _timeline(s, stage="prepare_synthesis", message="Synthesis context prepared")
    yield ("done:prepare_synthesis", s, {})

    if include_synthesis:
        yield ("starting:synthesize", s, {})
        s = await node_synthesize(s)
        _timeline(
            s,
            stage="synthesize",
            message="Drafts synthesized into final answer",
            details={"selected_model": s.get("selected_model", s.get("synthesizer", ""))},
        )
        yield ("done:synthesize", s, {})

        s = await node_reflect_deep(s)
        rr = s.get("reflection_result", {})
        if rr:
            _timeline(
                s,
                stage="reflection",
                message="Final answer evaluated by reflection gate",
                details={"quality": rr.get("quality", "n/a")},
            )
        yield ("done:reflect", s, {})


def _format_worker_event(
    rid: str,
    created: int,
    model_name: str,
    event: dict[str, Any],
) -> list[str]:
    """Render a worker progress event as SSE chunks for the client."""
    chunks: list[str] = []
    etype = event.get("type")
    wn = str(event.get("model", "?"))
    if etype == "worker_started":
        if EMIT_STATUS_UPDATES:
            chunks.append(_sc(rid, created, model_name, f"\n▶ Running: {wn}\n"))
        if EMIT_TIMELINE_EVENTS:
            chunks.append(_sc_event(rid, created, model_name, {
                "stage": "worker_started",
                "status": "info",
                "message": f"Worker {wn} started",
                "details": {"model": wn, "sub_task": event.get("sub_task", "")},
            }))
    elif etype == "worker_finished":
        status = event.get("status", "success")
        ms = int(event.get("elapsed_ms", 0) or 0)
        secs = ms / 1000.0
        if EMIT_STATUS_UPDATES:
            icon = "✅" if status == "success" else "⚠"
            chunks.append(_sc(
                rid, created, model_name,
                f"{icon} Finished: {wn} ({secs:.1f}s)\n",
            ))
        if EMIT_TIMELINE_EVENTS:
            chunks.append(_sc_event(rid, created, model_name, {
                "stage": "worker_finished",
                "status": "success" if status == "success" else "warning",
                "message": f"Worker {wn} {status}",
                "details": {
                    "model": wn,
                    "elapsed_ms": ms,
                    "sub_task": event.get("sub_task", ""),
                    "error": event.get("error", ""),
                },
            }))
    return chunks


async def _await_stream_stage(
    awaitable: Awaitable[Any],
    *,
    rid: str,
    created: int,
    model_name: str,
    request_id: str,
    stage: str,
    heartbeat_text: str,
    heartbeat_style: str = "line",
    progress_queue: asyncio.Queue | None = None,
) -> AsyncGenerator[tuple[str | None, Any], None]:
    started = time.monotonic()
    heartbeat_ticks = 0
    heartbeat_interval_s = STREAM_HEARTBEAT_SECONDS
    if heartbeat_style == "dots":
        heartbeat_interval_s = 5
    logger.info(
        "Streaming stage started rid=%s stage=%s model=%s",
        request_id,
        stage,
        model_name,
    )

    def _drain_queue() -> list[str]:
        out: list[str] = []
        if progress_queue is None:
            return out
        while True:
            try:
                evt = progress_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            out.extend(_format_worker_event(rid, created, model_name, evt))
        return out

    if heartbeat_interval_s <= 0:
        result = await awaitable
        elapsed_ms = int((time.monotonic() - started) * 1000)
        logger.info(
            "Streaming stage finished rid=%s stage=%s ms=%d",
            request_id,
            stage,
            elapsed_ms,
        )
        for chunk in _drain_queue():
            yield (chunk, _STREAM_STAGE_DONE)
        yield None, result
        return

    task = asyncio.create_task(awaitable)
    try:
        while True:
            try:
                result = await asyncio.wait_for(
                    asyncio.shield(task),
                    timeout=heartbeat_interval_s,
                )
                elapsed_ms = int((time.monotonic() - started) * 1000)
                logger.info(
                    "Streaming stage finished rid=%s stage=%s ms=%d",
                    request_id,
                    stage,
                    elapsed_ms,
                )
                for chunk in _drain_queue():
                    yield (chunk, _STREAM_STAGE_DONE)
                if EMIT_STATUS_UPDATES and heartbeat_style == "dots":
                    yield (_sc(rid, created, model_name, "\n"), _STREAM_STAGE_DONE)
                yield None, result
                return
            except asyncio.TimeoutError:
                # Drain any worker progress events that landed during this tick
                # so the client sees them promptly, not at stage end.
                for chunk in _drain_queue():
                    yield (chunk, _STREAM_STAGE_DONE)
                heartbeat_ticks += 1
                elapsed_s = heartbeat_ticks * heartbeat_interval_s
                actual_elapsed_s = max(1, int(time.monotonic() - started))
                logger.info(
                    "Streaming stage pending rid=%s stage=%s elapsed_s=%d displayed_s=%d",
                    request_id,
                    stage,
                    actual_elapsed_s,
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
                    should_emit_status = True
                    if heartbeat_style == "dots":
                        chunk_text = "."
                        yield (
                            _sc(rid, created, model_name, chunk_text),
                            _STREAM_STAGE_DONE,
                        )
                        continue
                    if heartbeat_style == "append_seconds":
                        # Keep very long stages readable: first ping, then every ~2 min.
                        throttle_ticks = max(
                            1,
                            120 // max(1, heartbeat_interval_s),
                        )
                        should_emit_status = (
                            heartbeat_ticks == 1
                            or heartbeat_ticks % throttle_ticks == 0
                        )
                    if should_emit_status:
                        chunk_text = f"⏳ {heartbeat_text} ({elapsed_s}s elapsed)\n"
                        yield (
                            _sc(rid, created, model_name, chunk_text),
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
    from config import (
        MODEL_REGISTRY,
        DEEP_PANEL_MIXED,
        DEEP_PANEL_CLOUD,
        DEEP_PANEL_LOCAL,
        DEEP_PANEL_CODE,
    )

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
        for panel_set in [
            DEEP_PANEL_MIXED,
            DEEP_PANEL_CLOUD,
            DEEP_PANEL_LOCAL,
            DEEP_PANEL_CODE,
        ]:
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

    try:
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
    finally:
        if state.ollama_session and not state.ollama_session.closed:
            await state.ollama_session.close()
        if state.ext_session and not state.ext_session.closed:
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

def _build_request_state(req, *, request_id: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build initial state and cache kwargs from a request. Shared by both paths."""
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
        request_id=request_id or str(uuid.uuid4()),
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
    return s, cache_kwargs


def _finalize_fast_only(s: dict[str, Any]) -> dict[str, Any]:
    """Populate fields for an audrey_fast fallback response."""
    s["selected_model"] = s.get("selected_model", s.get("fast_model", "none"))
    s["result_text"] = s.get("result_text") or _FAST_ONLY_FALLBACK_TEXT
    s.setdefault("prompt_tokens", estimate_tokens(flatten_messages(s["messages"])))
    s["completion_tokens"] = estimate_tokens(s["result_text"])
    s["escalated"] = False
    state.update_audrey_fast_health(
        selected_model=s.get("selected_model", "none"),
        success=False,
        reason=s.get("route_reason", "fast_only:fallback"),
    )
    _timeline(s, stage="complete", message="Fast-only fallback", status="warning")
    return s


def _log_completion(r: dict[str, Any], *, path: str) -> None:
    log_entry: dict[str, Any] = {
        "rid": r.get("request_id"),
        "model": r.get("requested_model"),
        "type": r.get("task_type"),
        "conf": r.get("confidence"),
        "selected": r.get("selected_model", r.get("synthesizer")),
        "path": path,
        "ms": r.get("latency_ms"),
    }
    if path in ("fast+react", "fast-only"):
        log_entry["react_rounds"] = r.get("react_rounds", 0)
        log_entry["search"] = r.get("search_performed", False)
        log_entry["search_query"] = _log_safe_query(r.get("search_query", ""))
        log_entry["tools"] = [t.get("tool", "") for t in r.get("tools_used", [])]
    else:
        log_entry["planned"] = bool(r.get("sub_tasks"))
        log_entry["reflection"] = (r.get("reflection_result") or {}).get("quality", "n/a")
        log_entry["search"] = r.get("search_performed", False)
        log_entry["search_query"] = _log_safe_query(r.get("search_query", ""))
        log_entry["tools"] = [t.get("tool", "") for t in r.get("tools_used", [])]
        log_entry["escalated"] = r.get("escalated", False)
        log_entry["mode"] = r.get("audrey_mode")
    logger.info(json.dumps(log_entry))


async def run_graph_dispatch(req) -> dict[str, Any]:
    s, cache_kwargs = _build_request_state(req)
    msgs_raw = s["messages"]
    temp = s["temperature"]
    _timeline(s, stage="request_received", message="Request accepted",
              details={"mode": s["audrey_mode"], "stream": False})

    if CACHE_ENABLED and not s["needs_fresh_data"]:
        cached = cache.get(msgs_raw, req.model, temp, **cache_kwargs)
        if cached is not None:
            logger.info("Cache hit for model=%s", req.model)
            s.update({
                "result_text": cached, "selected_model": "cache",
                "task_type": "cached", "confidence": 1.0,
                "route_reason": "Cache hit", "latency_ms": 0,
                "prompt_tokens": estimate_tokens(flatten_messages(msgs_raw)),
                "completion_tokens": estimate_tokens(cached),
                "search_performed": False, "use_fast_path": False,
                "escalated": False, "cache_hit": True,
            })
            _timeline(s, stage="cache", message="Response served from cache", status="success")
            return s
    elif CACHE_ENABLED and s["needs_fresh_data"]:
        _timeline(s, stage="cache", message="Cache bypassed for freshness-sensitive query",
                  details={"reason": "time_sensitive_query"})

    last_stage = ""
    async for stage_name, st, _hints in _orchestrate(s, req_model=req.model):
        s = st
        last_stage = stage_name

    s["latency_ms"] = int((time.time() - s["started_at"]) * 1000)

    if last_stage == "done:complete_fast":
        if CACHE_ENABLED and not s.get("needs_fresh_data", False) and s.get("result_text"):
            cache.put(msgs_raw, req.model, temp, s["result_text"], **cache_kwargs)
        if req.model == "audrey_fast":
            state.update_audrey_fast_health(
                selected_model=s.get("selected_model", s.get("fast_model", "none")),
                success=True, reason=s.get("route_reason", "fast_only:completed"),
            )
        _timeline(s, stage="complete", message="Completed on fast path", status="success",
                  details={"selected_model": s.get("selected_model", s.get("fast_model", ""))})
        _log_completion(s, path="fast-only" if req.model == "audrey_fast" else "fast+react")
        return s

    if last_stage == "done:fast_only_fallback":
        _finalize_fast_only(s)
        _log_completion(s, path="fast-only")
        return s

    # Deep panel completed
    if CACHE_ENABLED and s.get("result_text") and not s.get("needs_fresh_data", False):
        cache.put(msgs_raw, req.model, temp, s["result_text"], **cache_kwargs)
    s["cache_hit"] = bool(s.get("cache_hit", False))
    _timeline(s, stage="complete", message="Completed on deep panel", status="success",
              details={"selected_model": s.get("selected_model", s.get("synthesizer", ""))})
    _log_completion(s, path="deep")
    return s


# ── SSE stage-to-UX mapping ──────────────────────────────────────────────────

_STAGE_STATUS: dict[str, tuple[str, str, str]] = {
    "starting:classify":           ("🧭 Routing request...\n", "classify", "dots"),
    "starting:fast_react":         ("", "fast_react", "dots"),
    "starting:plan":               ("📋 Planning approach...\n", "plan", "dots"),
    "starting:generate":           ("🧠 Generating worker drafts", "parallel_generate", "dots"),
    "starting:prepare_synthesis":  ("🧩 Preparing synthesis...\n", "prepare_synthesis", "dots"),
    "starting:synthesize":         ("", "synthesize", "dots"),
}


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
            request_id = str(uuid.uuid4())
            ct = int(time.time())
            rid = f"chatcmpl-{request_id.replace('-', '')[:24]}"
            mn = req.model
            logger.info("Streaming request started rid=%s model=%s messages=%d",
                        request_id, mn, len(req.messages))

            def _evt(st, *, stage, message, status="info", details=None):
                event = _timeline(st, stage=stage, message=message, status=status, details=details)
                if EMIT_TIMELINE_EVENTS:
                    return _sc_event(rid, ct, mn, event)
                return None

            try:
                if EMIT_STATUS_UPDATES:
                    yield _sc(rid, ct, mn, "🔍 Analyzing...\n")

                init_state, _ = _build_request_state(req, request_id=request_id)
                start_evt = _evt(init_state, stage="request_received",
                                 message="Streaming request accepted",
                                 details={"mode": init_state.get("audrey_mode", "balanced")})
                if start_evt:
                    yield start_evt

                orch = _orchestrate(init_state, req_model=mn, include_synthesis=False)
                s = init_state
                last_stage = ""
                pending_heartbeat: tuple[str, str, str] | None = None

                async def _next_stage():
                    return await orch.__anext__()

                while True:
                    # Advance the orchestrator — with heartbeat wrapping
                    # when the previous yield was a starting: stage.
                    try:
                        if pending_heartbeat is not None:
                            hb_text, hb_stage, hb_style = pending_heartbeat
                            pending_heartbeat = None
                            # Only the parallel_generate stage currently emits
                            # per-worker progress events via this queue.
                            stage_progress_q = (
                                s.get("worker_progress_queue")
                                if hb_stage == "parallel_generate"
                                else None
                            )
                            async for chunk_or_result in _await_stream_stage(
                                _next_stage(),
                                rid=rid,
                                created=ct,
                                model_name=mn,
                                request_id=request_id,
                                stage=hb_stage,
                                heartbeat_text=hb_text or f"Processing {hb_stage}...",
                                heartbeat_style=hb_style,
                                progress_queue=stage_progress_q,
                            ):
                                sse_chunk, value = chunk_or_result
                                if sse_chunk is not None:
                                    yield sse_chunk
                                if value is not _STREAM_STAGE_DONE:
                                    stage_name, st, hints = value
                        else:
                            stage_name, st, hints = await _next_stage()
                    except StopAsyncIteration:
                        break

                    s = st
                    last_stage = stage_name

                    # ── Before each stage: emit status, set up heartbeat ──
                    if stage_name.startswith("starting:"):
                        status_text, hb_stage, hb_style = _STAGE_STATUS.get(
                            stage_name, ("", stage_name.split(":", 1)[1], "dots"),
                        )
                        if status_text and EMIT_STATUS_UPDATES:
                            yield _sc(rid, ct, mn, status_text)

                        if stage_name == "starting:fast_react":
                            fm = str(s.get("fast_model", "")).strip()
                            if EMIT_STATUS_UPDATES and fm:
                                yield _sc(rid, ct, mn, f"⚡ Running fast model: {fm} (tools/search)\n")
                            if EMIT_ROUTING_BANNER and fm:
                                yield _sc(rid, ct, mn, banner(s, running_model=fm))

                        if stage_name == "starting:generate":
                            dw = [str(w).strip() for w in s.get("deep_workers", []) if str(w).strip()]
                            if s.get("sub_tasks") and EMIT_STATUS_UPDATES:
                                yield _sc(rid, ct, mn, f"📋 Planning: {len(s['sub_tasks'])} sub-tasks\n")
                            if EMIT_ROUTING_BANNER and dw:
                                yield _sc(rid, ct, mn, banner(s, running_model=", ".join(dw)))
                            # Wire up a progress queue so workers can report
                            # started/finished events while running in parallel.
                            s["worker_progress_queue"] = asyncio.Queue()

                        pending_heartbeat = (status_text, hb_stage, hb_style)
                        continue

                    # ── After classify ──────────────────────────────
                    if stage_name == "done:classify":
                        evt = _evt(s, stage="classify", message="Request classified",
                                   details={"task_type": s.get("task_type"),
                                            "confidence": s.get("confidence"),
                                            "use_fast_path": bool(s.get("use_fast_path"))})
                        if evt:
                            yield evt

                    # ── Fast path completed successfully ────────────
                    elif stage_name == "done:complete_fast":
                        sel = s.get("selected_model", s.get("fast_model", "?"))
                        if req.model == "audrey_fast":
                            state.update_audrey_fast_health(
                                selected_model=sel, success=True,
                                reason=s.get("route_reason", "fast_only:completed"))
                        if EMIT_STATUS_UPDATES:
                            yield _sc(rid, ct, mn, f"✅ Fast model finished: {sel}\n\n")
                            if s.get("search_performed"):
                                sq = str(s.get("search_query", "")).strip()
                                yield _sc(rid, ct, mn,
                                          f"🌐 Web search used: {sq}\n" if sq else "🌐 Web search used\n")
                        if EMIT_ROUTING_BANNER:
                            yield _sc(rid, ct, mn, banner(s, finished_model=sel))
                        evt = _evt(s, stage="complete", message="Completed on fast path",
                                   status="success", details={"selected_model": sel})
                        if evt:
                            yield evt
                        content = s.get("result_text", "")
                        for i in range(0, len(content), _STREAM_RESULT_CHUNK_CHARS):
                            yield _sc(rid, ct, mn, content[i:i + _STREAM_RESULT_CHUNK_CHARS])
                        if EMIT_TRUST_SIGNALS:
                            yield _sc_event(rid, ct, mn, {"stage": "trust", "status": "info",
                                                          "message": "Trust signals",
                                                          "details": build_trust_signals(s)})
                        yield _sc_stop(rid, ct, mn)
                        yield _SSE_DONE
                        return

                    # ── Fast-only fallback ──────────────────────────
                    elif stage_name == "done:fast_only_fallback":
                        _finalize_fast_only(s)
                        evt = _evt(s, stage="complete", message="Fast-only fallback", status="warning")
                        if evt:
                            yield evt
                        yield _sc(rid, ct, mn, f"{_FAST_ONLY_FALLBACK_TEXT}\n")
                        yield _sc_stop(rid, ct, mn)
                        yield _SSE_DONE
                        return

                    # ── Escalation ──────────────────────────────────
                    elif stage_name == "done:escalation":
                        evt = _evt(s, stage="escalation",
                                   message="Escalated from fast path to deep panel", status="warning",
                                   details={"reason": s.get("route_reason", "")})
                        if evt:
                            yield evt

                    # ── Deep: plan done ─────────────────────────────
                    elif stage_name == "done:plan":
                        evt = _evt(s, stage="plan", message="Deep-panel plan ready",
                                   details={"workers": len(s.get("deep_workers", [])),
                                            "planned_sub_tasks": len(s.get("sub_tasks") or [])})
                        if evt:
                            yield evt

                    # ── Deep: generate done ─────────────────────────
                    elif stage_name == "done:generate":
                        s.pop("worker_progress_queue", None)
                        evt = _evt(s, stage="generate", message="Worker drafts generated",
                                   details={"outputs": len(s.get("worker_outputs", []))})
                        if evt:
                            yield evt
                        fw = [str(o.get("model", "")).strip()
                              for o in s.get("worker_outputs", [])
                              if str(o.get("model", "")).strip()]
                        fw_text = ", ".join(dict.fromkeys(fw)) or "none"
                        if EMIT_STATUS_UPDATES:
                            yield _sc(rid, ct, mn, f"\n✅ All workers finished: {fw_text}\n")
                        if EMIT_ROUTING_BANNER and fw_text != "none":
                            yield _sc(rid, ct, mn, banner(s, finished_model=fw_text))

                    # ── Deep: synthesis prepared ────────────────────
                    elif stage_name == "done:prepare_synthesis":
                        evt = _evt(s, stage="prepare_synthesis", message="Synthesis context prepared")
                        if evt:
                            yield evt
                        if EMIT_TRUST_SIGNALS:
                            yield _sc_event(rid, ct, mn, {"stage": "trust", "status": "info",
                                                          "message": "Trust signals",
                                                          "details": build_trust_signals(s)})

                # Orchestrator finished without early return → deep panel with streaming synthesis.
                async for chunk in stream_synthesis(s, rid=rid, created=ct):
                    yield chunk

            except Exception as e:
                logger.exception("Streaming request failed rid=%s model=%s: %s", request_id, mn, e)
                error_text = str(e).strip() or "internal error"
                yield _sc(rid, ct, mn, f"[Error: {error_text[:240]}]\n")
                yield _sc_stop(rid, ct, mn)
                yield _SSE_DONE

        return StreamingResponse(
            _stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
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
