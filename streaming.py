"""
Audrey — SSE streaming helpers.

Banner builder, chunk formatter, and async generators for fast-path
and deep-panel streaming.
"""

import json
import logging
import time
import uuid
from typing import Any, AsyncGenerator

import state
from config import EMIT_ROUTING_BANNER, EMIT_STATUS_UPDATES
from health import note_model_failure, note_model_success
from helpers import model_call_kwargs, role_prompt
from ollama import ollama_chat_stream
from agents import _REACT_SYSTEM

logger = logging.getLogger("audrey.streaming")


# ── SSE chunk helper ─────────────────────────────────────────────────────────

_SSE_DONE = "data: [DONE]\n\n"


def _sc(rid: str, created: int, model_name: str, text: str) -> str:
    return (
        f"data: {json.dumps({'id': rid, 'object': 'chat.completion.chunk', 'created': created, 'model': model_name, 'choices': [{'index': 0, 'delta': {'content': text}, 'finish_reason': None}]})}\n\n"
    )


def _sc_stop(rid: str, created: int, model_name: str) -> str:
    return (
        f"data: {json.dumps({'id': rid, 'object': 'chat.completion.chunk', 'created': created, 'model': model_name, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
    )


def _sc_event(rid: str, created: int, model_name: str, event: dict[str, Any]) -> str:
    payload = {
        "id": rid,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_name,
        "choices": [{"index": 0, "delta": {}, "finish_reason": None}],
        "audrey_event": event,
    }
    return f"data: {json.dumps(payload)}\n\n"


def _resolve_sid_ct(
    state: dict[str, Any],
    rid: str | None,
    created: int | None,
) -> tuple[str, int]:
    ct = created if created is not None else int(time.time())
    sid_source = str(state.get("request_id") or uuid.uuid4())
    sid = rid or f"chatcmpl-{sid_source.replace('-', '')[:24]}"
    return sid, ct


# ── Banner builder ───────────────────────────────────────────────────────────

def banner(
    s: dict[str, Any],
    *,
    running_model: str | None = None,
    finished_model: str | None = None,
) -> str:
    """Build a concise routing banner for the chat UI."""
    sel = s.get("selected_model") or s.get("synthesizer") or "?"
    path = "fast+react" if s.get("use_fast_path") else "deep"
    tags = []

    # Show workers that contributed drafts
    worker_outputs = s.get("worker_outputs", [])
    error_count = s.get("worker_error_count", 0)
    worker_models = []
    for wo in worker_outputs:
        m = str(wo.get("model", "")).strip()
        if m and not str(wo.get("content", "")).startswith("[WORKER_ERROR]"):
            worker_models.append(m)
    if worker_models:
        tags.append("✓ " + ", ".join(worker_models))
    if error_count > 0:
        tags.append(f"✗ {error_count} failed")

    run_name = str(running_model or "").strip()
    done_name = str(finished_model or "").strip()
    if run_name:
        tags.append(f"▶ {run_name}")
    if done_name:
        synth_tag = f"✓ synth:{done_name}"
        tags.append(synth_tag)

    detail = f" | {' | '.join(tags)}" if tags else ""

    return (
        f"[{s.get('requested_model')} → {sel} | {path}{detail}]\n"
    )


# ── Fast-path streamer ──────────────────────────────────────────────────────

async def stream_fast_path(
    s: dict[str, Any],
    *,
    rid: str | None = None,
    created: int | None = None,
) -> AsyncGenerator[str, None]:
    sid, ct = _resolve_sid_ct(s, rid, created)
    mn = s["requested_model"]
    model = s.get("fast_model", "")

    if not model:
        logger.warning("stream_fast_path called with empty fast_model — skipping")
        if mn == "audrey_fast":
            state.update_audrey_fast_health(
                selected_model="none",
                success=False,
                reason="fast_only:no_model_selected",
            )
        yield _sc(sid, ct, mn, "[Fast path error: no model selected. Falling back.]\n")
        yield _sc_stop(sid, ct, mn)
        yield _SSE_DONE
        return

    if EMIT_STATUS_UPDATES:
        yield _sc(sid, ct, mn, f"⚡ Running fast model: {model} (ReAct tools/search)\n\n")

    sys_msg = {
        "role": "system",
        "content": f"{_REACT_SYSTEM}\n\nFocus: {role_prompt(s['task_type'], model)}",
    }
    msgs = [sys_msg, *s["messages"]]

    try:
        async for item in ollama_chat_stream(
            model,
            msgs,
            **model_call_kwargs(s),
        ):
            c = (item.get("message") or {}).get("content", "")
            if c:
                yield _sc(sid, ct, mn, c)
            if item.get("done"):
                note_model_success(model)
                s["selected_model"] = model
                if mn == "audrey_fast":
                    state.update_audrey_fast_health(
                        selected_model=model,
                        success=True,
                        reason=s.get("route_reason", "fast_only:completed"),
                    )
                if EMIT_STATUS_UPDATES:
                    yield _sc(sid, ct, mn, f"\n✅ Fast model finished: {model}\n")
                if EMIT_ROUTING_BANNER:
                    yield _sc(sid, ct, mn, banner(s, finished_model=model))
                yield _sc_stop(sid, ct, mn)
                yield _SSE_DONE
                return
    except Exception as e:
        note_model_failure(model)
        logger.warning("Fast streaming failed for %s: %s", model, e)
        if mn == "audrey_fast":
            state.update_audrey_fast_health(
                selected_model=model,
                success=False,
                reason=f"fast_only:stream_error:{str(e)[:120]}",
            )
        yield _sc(sid, ct, mn, f"\n\n[Fast path error: {model}. Please retry.]\n")
        yield _sc_stop(sid, ct, mn)
        yield _SSE_DONE


# ── Deep-panel synthesis streamer ────────────────────────────────────────────

async def stream_synthesis(
    ps: dict[str, Any],
    *,
    rid: str | None = None,
    created: int | None = None,
) -> AsyncGenerator[str, None]:
    sid, ct = _resolve_sid_ct(ps, rid, created)
    mn = ps["requested_model"]
    synth_candidates = [
        str(m).strip()
        for m in ps.get("synthesis_candidates", [])
        if str(m).strip()
    ]
    if not synth_candidates:
        sy = str(ps.get("synthesizer", "")).strip()
        fb = str(ps.get("fallback_synthesizer", "")).strip()
        synth_candidates = [m for m in [sy, fb] if m]
    current_synth = synth_candidates[0] if synth_candidates else "?"

    if EMIT_STATUS_UPDATES:
        yield _sc(sid, ct, mn, f"🧠 Synthesizing with {current_synth}...\n\n---\n\n")

    for i, synth in enumerate(synth_candidates):
        try:
            async for item in ollama_chat_stream(
                synth,
                ps["synthesis_messages"],
                **model_call_kwargs(ps, temperature=min(ps["temperature"], 0.3)),
            ):
                c = (item.get("message") or {}).get("content", "")
                if c:
                    yield _sc(sid, ct, mn, c)
                if item.get("done"):
                    note_model_success(synth)
                    ps["selected_model"] = synth
                    if EMIT_STATUS_UPDATES:
                        yield _sc(sid, ct, mn, f"\n✅ Synthesis model finished: {synth}\n")
                    if EMIT_ROUTING_BANNER:
                        yield _sc(sid, ct, mn, banner(ps, finished_model=synth))
                    yield _sc_stop(sid, ct, mn)
                    yield _SSE_DONE
                    return
        except Exception as e:
            logger.warning("Synthesis failed for %s: %s", synth, e)
            note_model_failure(synth)
            if EMIT_STATUS_UPDATES and i < len(synth_candidates) - 1:
                nxt = synth_candidates[i + 1]
                yield _sc(sid, ct, mn, f"\n[Retrying synthesis with {nxt}]\n\n")

    yield _sc(sid, ct, mn, "[Error. Please try again.]")
    yield _sc_stop(sid, ct, mn)
    yield _SSE_DONE
