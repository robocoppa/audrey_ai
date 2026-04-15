"""
Audrey — Ollama communication layer.

Builds payloads, sends requests (single-shot and streaming), and provides
model runners with optional tool-calling support.
"""

import json
import logging
import traceback
from typing import Any, AsyncGenerator

import aiohttp

import state
from config import OLLAMA_BASE_URL, TOOL_CAPABLE_MODELS, TOOLS_ENABLED, is_cloud_model
from helpers import timeout_for_model

logger = logging.getLogger("audrey.ollama")


# ── Payload builder ──────────────────────────────────────────────────────────

def build_ollama_payload(
    model: str,
    msgs: list[dict[str, Any]],
    *,
    temperature: float,
    max_tokens: int | None,
    top_p: float | None,
    stop: Any | None,
    stream: bool,
    frequency_penalty: float | None = None,
    presence_penalty: float | None = None,
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    p: dict[str, Any] = {
        "model": model,
        "messages": msgs,
        "stream": stream,
        "options": {"temperature": temperature},
    }
    if max_tokens is not None:
        p["options"]["num_predict"] = max_tokens
    if top_p is not None:
        p["options"]["top_p"] = top_p
    if stop is not None:
        p["options"]["stop"] = stop
    if frequency_penalty is not None:
        p["options"]["frequency_penalty"] = frequency_penalty
    if presence_penalty is not None:
        p["options"]["presence_penalty"] = presence_penalty
    if tools:
        p["tools"] = tools
    return p


# ── Single-shot chat ─────────────────────────────────────────────────────────

async def ollama_chat_once(
    model: str,
    msgs: list[dict[str, Any]],
    *,
    temperature: float,
    max_tokens: int | None,
    top_p: float | None,
    stop: Any | None,
    frequency_penalty: float | None = None,
    presence_penalty: float | None = None,
    tools: list[dict[str, Any]] | None = None,
    request_timeout: int | None = None,
) -> dict[str, Any]:
    pl = build_ollama_payload(
        model, msgs,
        temperature=temperature, max_tokens=max_tokens, top_p=top_p,
        stop=stop, stream=False,
        frequency_penalty=frequency_penalty, presence_penalty=presence_penalty,
        tools=tools,
    )
    tout = request_timeout or timeout_for_model(model)

    async def _do():
        async with state.ollama_session.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=pl,
            timeout=aiohttp.ClientTimeout(total=tout),
        ) as r:
            if r.status != 200:
                raise RuntimeError(f"Ollama {r.status}: {await r.text()}")
            return await r.json()

    if is_cloud_model(model):
        return await _do()
    async with state.gpu_semaphore:
        return await _do()


# ── Streaming chat ───────────────────────────────────────────────────────────

async def ollama_chat_stream(
    model: str,
    msgs: list[dict[str, Any]],
    *,
    temperature: float,
    max_tokens: int | None,
    top_p: float | None,
    stop: Any | None,
    frequency_penalty: float | None = None,
    presence_penalty: float | None = None,
) -> AsyncGenerator[dict[str, Any], None]:
    pl = build_ollama_payload(
        model, msgs,
        temperature=temperature, max_tokens=max_tokens, top_p=top_p,
        stop=stop, stream=True,
        frequency_penalty=frequency_penalty, presence_penalty=presence_penalty,
    )
    tout = timeout_for_model(model)

    async def _do():
        async with state.ollama_session.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=pl,
            timeout=aiohttp.ClientTimeout(total=None, sock_read=tout),
        ) as r:
            if r.status != 200:
                raise RuntimeError(f"Ollama {r.status}: {await r.text()}")
            async for raw in r.content:
                ln = raw.decode().strip()
                if not ln:
                    continue
                try:
                    yield json.loads(ln)
                except json.JSONDecodeError:
                    pass

    if is_cloud_model(model):
        async for item in _do():
            yield item
    else:
        async with state.gpu_semaphore:
            async for item in _do():
                yield item


# ── High-level model runners ─────────────────────────────────────────────────

async def run_model_once(
    model: str,
    msgs: list[dict[str, Any]],
    *,
    temperature: float,
    max_tokens: int | None,
    top_p: float | None,
    stop: Any | None,
    frequency_penalty: float | None = None,
    presence_penalty: float | None = None,
) -> str:
    d = await ollama_chat_once(
        model, msgs,
        temperature=temperature, max_tokens=max_tokens, top_p=top_p,
        stop=stop,
        frequency_penalty=frequency_penalty, presence_penalty=presence_penalty,
    )
    c = d["message"]["content"]
    if not c or not c.strip():
        raise RuntimeError("Empty content")
    return c


async def run_model_with_tools(
    model: str,
    msgs: list[dict[str, Any]],
    *,
    temperature: float,
    max_tokens: int | None,
    top_p: float | None,
    stop: Any | None,
    frequency_penalty: float | None = None,
    presence_penalty: float | None = None,
    max_tool_rounds: int | None = None,
) -> str:
    content, _ = await run_model_with_tools_detailed(
        model,
        msgs,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        stop=stop,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        max_tool_rounds=max_tool_rounds,
    )
    return content


async def run_model_with_tools_detailed(
    model: str,
    msgs: list[dict[str, Any]],
    *,
    temperature: float,
    max_tokens: int | None,
    top_p: float | None,
    stop: Any | None,
    frequency_penalty: float | None = None,
    presence_penalty: float | None = None,
    max_tool_rounds: int | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Run a model with tool-calling support via the tool registry.

    Falls back to run_model_once when:
      - Tools are globally disabled
      - The tool registry has no tools
      - The model is not in the TOOL_CAPABLE_MODELS set
    """
    # ── Gate: skip tools entirely for non-capable models ──
    model_supports_tools = model in TOOL_CAPABLE_MODELS
    if (
        not TOOLS_ENABLED
        or not state.tool_registry
        or not state.tool_registry.has_tools
        or not model_supports_tools
    ):
        if not model_supports_tools and TOOLS_ENABLED:
            logger.debug("Model %s not in TOOL_CAPABLE_MODELS — skipping tools", model)
        content = await run_model_once(
            model, msgs,
            temperature=temperature, max_tokens=max_tokens, top_p=top_p,
            stop=stop,
            frequency_penalty=frequency_penalty, presence_penalty=presence_penalty,
        )
        return content, []

    tool_defs = state.tool_registry.tool_definitions

    async def chat_fn(current_msgs):
        return await ollama_chat_once(
            model, current_msgs,
            temperature=temperature, max_tokens=max_tokens, top_p=top_p,
            stop=stop,
            frequency_penalty=frequency_penalty, presence_penalty=presence_penalty,
            tools=tool_defs,
        )

    try:
        content, _, tool_calls_log = await state.tool_registry.run_with_tools(
            chat_fn,
            msgs,
            max_rounds=max_tool_rounds,
        )
        return content, tool_calls_log
    except Exception as e:
        logger.error(
            "run_model_with_tools failed for %s: %s\n%s",
            model, e, traceback.format_exc(),
        )
        raise
