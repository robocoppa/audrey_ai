"""
Audrey — SSE streaming helpers.

Banner builder, chunk formatter, and async generators for fast-path
and deep-panel streaming.

Fast-path streaming runs tool rounds non-streaming (tool calls can't be
streamed anyway), then streams the final generation pass from Ollama.
Reflection runs after streaming completes.
"""

import asyncio
import json
import logging
import uuid
import time
from typing import Any, AsyncGenerator, Dict, List, Optional

import state
from config import (
    EMIT_ROUTING_BANNER,
    EMIT_STATUS_UPDATES,
    ESCALATION_CONFIDENCE_CEILING,
    ESCALATION_ENABLED,
    ESCALATION_MIN_LENGTH,
    FAST_PATH_TIMEOUT,
    REFLECTION_ENABLED,
    TOOL_CAPABLE_MODELS,
    TOOLS_ENABLED,
)
from health import note_model_failure, note_model_success
from helpers import estimate_tokens, flatten_messages, get_last_user_text, role_prompt
from ollama import ollama_chat_once, ollama_chat_stream
from agents import _REACT_SYSTEM, reflect_on_response

logger = logging.getLogger("audrey.streaming")


# ── SSE chunk helper ─────────────────────────────────────────────────────────

def _sc(rid: str, created: int, model_name: str, text: str) -> str:
    return (
        f"data: {json.dumps({'id': rid, 'object': 'chat.completion.chunk', 'created': created, 'model': model_name, 'choices': [{'index': 0, 'delta': {'content': text}, 'finish_reason': None}]})}\n\n"
    )


def _stop(rid: str, created: int, model_name: str) -> str:
    return (
        f"data: {json.dumps({'id': rid, 'object': 'chat.completion.chunk', 'created': created, 'model': model_name, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
    )


def _done() -> str:
    return "data: [DONE]\n\n"


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

    review = " | 📝 review" if s.get("is_code_review") else ""

    return (
        f"[{s.get('requested_model')} → {sel} | {s.get('task_type')} "
        f"| conf {s.get('confidence', 0):.2f} | {path}{review}{sr}{tools_str}{react}{plan}{refl}{esc} "
        f"| {s.get('route_reason', '')}]\n"
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Fast-path streamer
# ══════════════════════════════════════════════════════════════════════════════
#
# Strategy:
#   1. Run tool rounds non-streaming (tool calls can't be streamed)
#   2. Stream the final generation from Ollama via ollama_chat_stream
#   3. Collect the streamed text for reflection + cache
#   4. Run reflection after streaming completes (small router call, not visible)
#   5. If reflection says poor quality, set escalated=True so main.py can
#      fall through to deep panel on next request
#
# When the model has no tools or doesn't need them, step 1 is skipped and
# we stream directly — zero added latency vs a plain ollama_chat_stream call.

async def stream_fast_path(s: Dict[str, Any]) -> AsyncGenerator[str, None]:
    ct = int(time.time())
    rid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    mn = s["requested_model"]
    model = s.get("fast_model", "")

    # Guard: if fast_model is blank, yield error and bail
    if not model:
        logger.warning("stream_fast_path called with empty fast_model — skipping")
        yield _sc(rid, ct, mn, "[Fast path error: no model selected.]\n")
        yield _stop(rid, ct, mn)
        yield _done()
        return

    if EMIT_STATUS_UPDATES:
        yield _sc(rid, ct, mn, f"⚡ Fast path (ReAct): {model}\n")

    # Build conversation with system prompt
    sys_msg = {
        "role": "system",
        "content": f"{_REACT_SYSTEM}\n\nFocus: {role_prompt(s['task_type'], model, is_code_review=s.get('is_code_review', False))}",
    }
    msgs = [sys_msg, *s["messages"]]
    tool_calls_log: list = []
    tool_rounds = 0

    model_can_use_tools = (
        TOOLS_ENABLED
        and model in TOOL_CAPABLE_MODELS
        and state.tool_registry
        and state.tool_registry.has_tools
    )

    # ── Tool rounds (non-streaming) ──────────────────────────────────────
    if model_can_use_tools:
        tool_defs = state.tool_registry.tool_definitions

        async def chat_fn(current_msgs):
            return await ollama_chat_once(
                model,
                current_msgs,
                temperature=s["temperature"],
                max_tokens=s.get("max_tokens"),
                top_p=s.get("top_p"),
                stop=s.get("stop"),
                frequency_penalty=s.get("frequency_penalty"),
                presence_penalty=s.get("presence_penalty"),
                tools=tool_defs,
            )

        from config import MAX_TOOL_ROUNDS
        from tool_registry import compress_tool_context

        current = list(msgs)
        for round_num in range(MAX_TOOL_ROUNDS):
            if round_num > 0 and round_num % 2 == 0:
                current = compress_tool_context(current)

            try:
                data = await asyncio.wait_for(
                    chat_fn(current), timeout=FAST_PATH_TIMEOUT
                )
            except Exception as e:
                note_model_failure(model)
                logger.warning("Fast stream tool round failed for %s: %s", model, e)
                yield _sc(rid, ct, mn, f"\n\n[Fast path error: {model}. Please retry.]\n")
                yield _stop(rid, ct, mn)
                yield _done()
                return

            msg = data.get("message", {}) if isinstance(data, dict) else {}
            tool_calls = msg.get("tool_calls")

            if not tool_calls or not isinstance(tool_calls, list):
                # No tool calls on this round
                if round_num == 0:
                    # First call returned no tools — break to streaming path
                    break

                # Had tool rounds, model gave final answer non-streaming.
                # Emit the pre-computed content.
                content = msg.get("content", "")
                if content:
                    tool_rounds = round_num
                    s["react_rounds"] = tool_rounds
                    s["tools_used"] = tool_calls_log
                    s["selected_model"] = model

                    if tool_calls_log and EMIT_STATUS_UPDATES:
                        names = list(dict.fromkeys(t["tool"] for t in tool_calls_log))
                        short = [n.split("__", 1)[-1] if "__" in n else n for n in names]
                        yield _sc(
                            rid, ct, mn,
                            f"🔧 Tools: {', '.join(short)} "
                            f"({tool_rounds} round{'s' if tool_rounds > 1 else ''})\n",
                        )
                    if EMIT_ROUTING_BANNER:
                        yield _sc(rid, ct, mn, banner(s))

                    yield _sc(rid, ct, mn, content)
                    note_model_success(model)

                    s["result_text"] = content
                    s["prompt_tokens"] = estimate_tokens(flatten_messages(s["messages"]))
                    s["completion_tokens"] = estimate_tokens(content)
                    await _post_stream_reflect(s)

                    yield _stop(rid, ct, mn)
                    yield _done()
                    return
                break

            # Process tool calls
            current.append({
                "role": "assistant",
                "content": msg.get("content", ""),
                "tool_calls": tool_calls,
            })
            for tc in tool_calls:
                func = tc.get("function", {}) if isinstance(tc, dict) else {}
                name = func.get("name", "") if isinstance(func, dict) else ""
                args = func.get("arguments", {}) if isinstance(func, dict) else {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                if not isinstance(args, dict):
                    args = {}
                try:
                    result = await state.tool_registry.execute(name, args)
                except Exception as exc:
                    result = json.dumps({"error": str(exc)})
                current.append({"role": "tool", "content": result})
                tool_calls_log.append({
                    "round": round_num + 1,
                    "tool": name,
                    "args_preview": json.dumps(args, default=str)[:100],
                    "result_len": len(result),
                })

            tool_rounds = round_num + 1
            if EMIT_STATUS_UPDATES:
                last_tools = [
                    tc.get("function", {}).get("name", "?")
                    for tc in tool_calls if isinstance(tc, dict)
                ]
                short = [n.split("__", 1)[-1] if "__" in n else n for n in last_tools]
                yield _sc(rid, ct, mn, f"🔧 Round {tool_rounds}: {', '.join(short)}\n")

        # Update msgs to include tool context for the streaming call
        if tool_rounds > 0:
            msgs = current

    # ── Stream the final generation ──────────────────────────────────────
    s["react_rounds"] = tool_rounds
    s["tools_used"] = tool_calls_log
    s["selected_model"] = model

    if EMIT_ROUTING_BANNER:
        yield _sc(rid, ct, mn, banner(s))

    collected: list = []
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
                collected.append(c)
                yield _sc(rid, ct, mn, c)
            if item.get("done"):
                note_model_success(model)
                break
    except Exception as e:
        note_model_failure(model)
        logger.warning("Fast streaming failed for %s: %s", model, e)
        yield _sc(rid, ct, mn, f"\n\n[Fast path error: {model}. Please retry.]\n")
        yield _stop(rid, ct, mn)
        yield _done()
        return

    # ── Post-stream: collect, reflect, finalize ──────────────────────────
    result_text = "".join(collected)
    s["result_text"] = result_text
    s["prompt_tokens"] = estimate_tokens(flatten_messages(s["messages"]))
    s["completion_tokens"] = estimate_tokens(result_text)

    await _post_stream_reflect(s)

    yield _stop(rid, ct, mn)
    yield _done()


# ── Post-stream reflection ───────────────────────────────────────────────────

async def _post_stream_reflect(s: Dict[str, Any]) -> None:
    """Run reflection after streaming completes.  Mutates s in place.

    For streaming, we can't re-generate — the tokens are already sent.
    We log quality and set escalation flags so the banner is accurate and
    main.py can log the reflection result.
    """
    if not REFLECTION_ENABLED or not s.get("result_text"):
        return

    result = s["result_text"]

    # Length-based escalation check
    if (
        ESCALATION_ENABLED
        and len(result.strip()) < ESCALATION_MIN_LENGTH
        and s.get("confidence", 1.0) < ESCALATION_CONFIDENCE_CEILING
    ):
        question_len = len(
            get_last_user_text(s.get("original_messages", s["messages"]))
        )
        if question_len > 50:
            logger.info(
                "Stream reflection: response too short (%d chars) — flagging",
                len(result.strip()),
            )
            s["escalated"] = True
            return

    try:
        reflection = await reflect_on_response(
            s.get("original_messages", s["messages"]),
            result,
        )
        s["reflection_result"] = reflection
        if reflection["quality"] == "poor":
            logger.info("Stream reflection: quality=poor — flagging")
            s["escalated"] = True
    except Exception as e:
        logger.warning("Stream reflection failed: %s — assuming ok", e)


# ══════════════════════════════════════════════════════════════════════════════
#  Deep-panel synthesis streamer
# ══════════════════════════════════════════════════════════════════════════════

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
                    yield _sc(rid, ct, mn, c)
                if item.get("done"):
                    note_model_success(s)
                    yield _stop(rid, ct, mn)
                    yield _done()
                    return
        except Exception:
            note_model_failure(s)

    yield _sc(rid, ct, mn, "[Error. Please try again.]")
    yield _stop(rid, ct, mn)
    yield _done()
