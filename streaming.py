"""
Audrey — SSE streaming helpers.

Banner builder, chunk formatter, and async generators for fast-path
and deep-panel streaming.
"""

import json
import logging
import uuid
import time
from typing import Any, AsyncGenerator, Dict

from config import EMIT_ROUTING_BANNER, EMIT_STATUS_UPDATES
from health import note_model_failure, note_model_success
from helpers import role_prompt
from ollama import ollama_chat_stream
from agents import _REACT_SYSTEM

logger = logging.getLogger("audrey.streaming")


# ── SSE chunk helper ─────────────────────────────────────────────────────────

def _sc(rid: str, created: int, model_name: str, text: str) -> str:
    return (
        f"data: {json.dumps({'id': rid, 'object': 'chat.completion.chunk', 'created': created, 'model': model_name, 'choices': [{'index': 0, 'delta': {'content': text}, 'finish_reason': None}]})}\n\n"
    )


# ── Banner builder ───────────────────────────────────────────────────────────

def banner(s: Dict[str, Any]) -> str:
    sel = s.get("selected_model") or s.get("synthesizer") or "?"
    path = "fast+react" if s.get("use_fast_path") else "deep"
    esc = " | ESCALATED" if s.get("escalated") else ""
    plan = " | planned" if s.get("sub_tasks") else ""
    react = (
        f" | react×{s.get('react_rounds', 0)}" if s.get("react_rounds") else ""
    )
    refl = ""
    rr = s.get("reflection_result", {})
    if rr.get("quality"):
        refl = f" | refl:{rr['quality']}"

    sr = ""
    if s.get("search_performed"):
        sq = s.get("search_query", "")
        sr = f' | 🌐 search: "{sq}"' if sq else " | +search"

    tools_str = ""
    tools_log = s.get("tools_used", [])
    if tools_log:
        seen: Dict[str, int] = {}
        for t in tools_log:
            raw_name = t.get("tool", "")
            short = raw_name.split("__", 1)[-1] if "__" in raw_name else raw_name
            seen[short] = seen.get(short, 0) + 1
        parts = [f"{n}×{c}" if c > 1 else n for n, c in seen.items()]
        tools_str = f" | 🔧 tools: {', '.join(parts)}"

    return (
        f"[{s.get('requested_model')} → {sel} | {s.get('task_type')} "
        f"| conf {s.get('confidence', 0):.2f} | {path}{sr}{tools_str}{react}{plan}{refl}{esc} "
        f"| {s.get('route_reason', '')}]\n"
    )


# ── Fast-path streamer ──────────────────────────────────────────────────────

async def stream_fast_path(s: Dict[str, Any]) -> AsyncGenerator[str, None]:
    ct = int(time.time())
    rid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    mn = s["requested_model"]
    model = s["fast_model"]

    if EMIT_STATUS_UPDATES:
        yield _sc(rid, ct, mn, f"⚡ Fast path (ReAct): {model}\n\n")
    if EMIT_ROUTING_BANNER:
        yield _sc(rid, ct, mn, banner(s))

    sys_msg = {
        "role": "system",
        "content": f"{_REACT_SYSTEM}\n\nFocus: {role_prompt(s['task_type'], model)}",
    }
    msgs = [sys_msg, *s["messages"]]

    try:
        async for item in ollama_chat_stream(
            model,
            msgs,
            temperature=s["temperature"],
            max_tokens=s.get("max_tokens"),
            top_p=s.get("top_p"),
            stop=s.get("stop"),
            frequency_penalty=s.get("frequency_penalty"),
            presence_penalty=s.get("presence_penalty"),
        ):
            msg = item.get("message") or {}
            c = msg.get("content", "")
            if c:
                yield (
                    f"data: {json.dumps({'id': rid, 'object': 'chat.completion.chunk', 'created': ct, 'model': mn, 'choices': [{'index': 0, 'delta': {'content': c}, 'finish_reason': None}]})}\n\n"
                )
            if item.get("done"):
                note_model_success(model)
                yield (
                    f"data: {json.dumps({'id': rid, 'object': 'chat.completion.chunk', 'created': ct, 'model': mn, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
                )
                yield "data: [DONE]\n\n"
                return
    except Exception as e:
        note_model_failure(model)
        logger.warning("Fast streaming failed for %s: %s", model, e)
        yield _sc(rid, ct, mn, f"\n\n[Fast path error: {model}. Please retry.]\n")
        yield (
            f"data: {json.dumps({'id': rid, 'object': 'chat.completion.chunk', 'created': ct, 'model': mn, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
        )
        yield "data: [DONE]\n\n"


# ── Deep-panel synthesis streamer ────────────────────────────────────────────

async def stream_synthesis(ps: Dict[str, Any]) -> AsyncGenerator[str, None]:
    ct = int(time.time())
    rid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    mn = ps["requested_model"]
    sy = ps["synthesizer"]
    fb = ps.get("fallback_synthesizer", "")
    ws = ps.get("deep_workers", [])
    sub_tasks = ps.get("sub_tasks")

    if EMIT_STATUS_UPDATES:
        yield _sc(rid, ct, mn, f"🧠 Drafts from {len(ws)} models ({', '.join(ws)})\n")
        if sub_tasks:
            yield _sc(rid, ct, mn, f"📋 Planned: {len(sub_tasks)} sub-tasks\n")
        yield _sc(rid, ct, mn, f"✨ Synthesizing with {sy}...\n\n---\n\n")
    if EMIT_ROUTING_BANNER:
        yield _sc(rid, ct, mn, banner(ps))

    for s in [sy] + ([fb] if fb else []):
        try:
            async for item in ollama_chat_stream(
                s,
                ps["synthesis_messages"],
                temperature=min(ps["temperature"], 0.3),
                max_tokens=ps.get("max_tokens"),
                top_p=ps.get("top_p"),
                stop=ps.get("stop"),
                frequency_penalty=ps.get("frequency_penalty"),
                presence_penalty=ps.get("presence_penalty"),
            ):
                msg = item.get("message") or {}
                c = msg.get("content", "")
                if c:
                    yield (
                        f"data: {json.dumps({'id': rid, 'object': 'chat.completion.chunk', 'created': ct, 'model': mn, 'choices': [{'index': 0, 'delta': {'content': c}, 'finish_reason': None}]})}\n\n"
                    )
                if item.get("done"):
                    note_model_success(s)
                    yield (
                        f"data: {json.dumps({'id': rid, 'object': 'chat.completion.chunk', 'created': ct, 'model': mn, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
                    )
                    yield "data: [DONE]\n\n"
                    return
        except Exception:
            note_model_failure(s)

    yield (
        f"data: {json.dumps({'id': rid, 'object': 'chat.completion.chunk', 'created': ct, 'model': mn, 'choices': [{'index': 0, 'delta': {'content': '[Error. Please try again.]'}, 'finish_reason': 'stop'}]})}\n\n"
    )
    yield "data: [DONE]\n\n"
