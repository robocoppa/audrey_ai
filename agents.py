"""
Audrey — agentic features.

Planning (sub-task decomposition), reflection gates, ReAct agent loop,
and adaptive escalation from fast path to deep panel.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any

import state
from classifier import _extract_json
from config import (
    ESCALATION_CONFIDENCE_CEILING,
    ESCALATION_ENABLED,
    ESCALATION_MIN_LENGTH,
    FAST_PATH_TIMEOUT,
    PLANNING_ENABLED,
    PLANNING_MIN_TOKENS,
    REACT_MAX_ROUNDS,
    REFLECTION_ENABLED,
    REFLECTION_MAX_RETRIES,
    ROUTER_MODEL,
    TOOL_CAPABLE_MODELS,
    TOOLS_ENABLED,
)
from health import note_model_failure, note_model_success
from helpers import (
    extract_web_search_info,
    estimate_tokens,
    flatten_messages,
    get_last_user_text,
    model_call_kwargs,
    role_prompt,
)
from ollama import ollama_chat_once, run_model_once, run_model_with_tools

logger = logging.getLogger("audrey.agents")


# ══════════════════════════════════════════════════════════════════════════════
#  Planning — decomposes complex queries for deep workers
# ══════════════════════════════════════════════════════════════════════════════

_PLANNER_SYSTEM = """You are a task planner. Given a user request, decide if it should be
decomposed into focused sub-tasks for parallel expert workers, or handled as a single task.

Rules:
- If the request is simple or has a single clear focus, return: {"plan": "single"}
- If complex (multi-part, comparison, multi-step analysis), decompose into 2-3 sub-tasks.
- Each sub-task should be self-contained and answerable independently.
- Return strict JSON, no markdown fences.

For decomposition, return:
{"plan": "decompose", "sub_tasks": ["sub-task 1 description", "sub-task 2 description"]}"""


async def plan_sub_tasks(
    msgs: list[dict[str, Any]],
    task_type: str,
    *,
    audrey_mode: str = "balanced",
    planning_enabled_override: bool | None = None,
    min_tokens_override: int | None = None,
) -> list[str] | None:
    """Use the router model to optionally decompose a complex query."""
    planning_enabled = (
        planning_enabled_override
        if planning_enabled_override is not None
        else PLANNING_ENABLED
    )
    if audrey_mode == "quick":
        planning_enabled = False
    elif audrey_mode == "research":
        planning_enabled = True

    if not planning_enabled:
        return None

    user_text = get_last_user_text(msgs)
    min_tokens = (
        int(min_tokens_override)
        if min_tokens_override is not None
        else PLANNING_MIN_TOKENS
    )
    if audrey_mode == "research":
        min_tokens = min(min_tokens, 40)

    if estimate_tokens(user_text) < min_tokens:
        return None

    try:
        raw = await run_model_once(
            ROUTER_MODEL,
            [
                {"role": "system", "content": _PLANNER_SYSTEM},
                {
                    "role": "user",
                    "content": f"Task type: {task_type}\n\nRequest:\n{user_text}",
                },
            ],
            temperature=0.0,
            max_tokens=300,
            top_p=0.9,
            stop=None,
        )
        parsed = _extract_json(raw)
        if parsed and parsed.get("plan") == "decompose" and parsed.get("sub_tasks"):
            tasks = parsed["sub_tasks"]
            if isinstance(tasks, list) and 2 <= len(tasks) <= 4:
                logger.info("Planner decomposed into %d sub-tasks", len(tasks))
                return tasks
        return None
    except Exception as e:
        logger.warning("Planning failed: %s — skipping", e)
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  Reflection gate — checks response completeness
# ══════════════════════════════════════════════════════════════════════════════

_REFLECT_SYSTEM = """You are a response quality checker. Given the original question and a
candidate answer, evaluate whether the answer is complete and correct.

Return strict JSON (no fences):
{
  "complete": true/false,
  "quality": "good" | "partial" | "poor",
  "missing": "brief description of what's missing or empty string if complete"
}"""


async def reflect_on_response(
    original_msgs: list[dict[str, Any]],
    response_text: str,
) -> dict[str, Any]:
    """Use the router model to check if the response adequately answers the question."""
    if not REFLECTION_ENABLED:
        return {"complete": True, "quality": "good", "missing": ""}

    # Don't reflect on very long responses — they're almost certainly substantive
    if len(response_text) > 3000:
        return {"complete": True, "quality": "good", "missing": ""}

    try:
        question = get_last_user_text(original_msgs)
        if not question:
            return {"complete": True, "quality": "good", "missing": ""}

        raw = await run_model_once(
            ROUTER_MODEL,
            [
                {"role": "system", "content": _REFLECT_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"Question:\n{question[:500]}\n\n"
                        f"Answer:\n{response_text[:1500]}"
                    ),
                },
            ],
            temperature=0.0,
            max_tokens=150,
            top_p=0.9,
            stop=None,
        )
        parsed = _extract_json(raw)
        if not parsed:
            logger.warning("Reflection parse failure. Raw: %s", raw[:200])
            return {"complete": True, "quality": "good", "missing": ""}
        result = {
            "complete": bool(parsed.get("complete", True)),
            "quality": parsed.get("quality", "good"),
            "missing": str(parsed.get("missing", "")),
        }
        logger.info(
            "Reflection: complete=%s quality=%s missing='%s'",
            result["complete"],
            result["quality"],
            result["missing"][:80],
        )
        return result
    except Exception as e:
        logger.warning("Reflection failed: %s — assuming complete", e)
        return {"complete": True, "quality": "good", "missing": ""}


# ══════════════════════════════════════════════════════════════════════════════
#  ReAct agent loop for fast path
# ══════════════════════════════════════════════════════════════════════════════

_REACT_SYSTEM = """You are a helpful AI assistant with access to tools. Think step by step.

For each request:
1. THINK about what information you need and whether tools would help.
2. If you need data, use the available tools. You can make multiple tool calls across rounds.
3. Once you have enough information, provide your final comprehensive answer.

If web search results are already provided in context, use them directly rather than
searching again unless you need additional information.

Date handling rules:
- Always resolve relative dates (today, tomorrow, yesterday, this year, next year, last year)
  against the current date in system context before using tools.
- When making tool/web search queries for relative-date questions, include absolute
  dates/years (for example, "next year" -> a concrete year).
- If the user asks about time-sensitive facts, prefer fresh tool/web evidence and
  state the resolved absolute date/year in the answer.

Be thorough but efficient — use tools only when they add value.
Do NOT wrap your entire response in a code block or code fence. Use code fences only for actual code snippets. Output clean markdown directly."""


async def run_react_agent(s: dict[str, Any]) -> dict[str, Any]:
    """ReAct loop — think, act, observe, repeat.

    Guards:
      - If use_fast_path is False or fast_model is blank, short-circuit
        (the fast graph always runs this node, so we must bail early when
        classify decided against fast path).
      - Only sends tools if the model is in TOOL_CAPABLE_MODELS.
    """
    # ── Guard: skip if fast path wasn't selected ──
    if not s.get("use_fast_path") or not s.get("fast_model"):
        logger.debug("react_agent skipped: use_fast_path=%s fast_model='%s'",
                      s.get("use_fast_path"), s.get("fast_model"))
        return s

    model = s["fast_model"]
    task_prompt = role_prompt(s["task_type"], model, is_code_review=s.get("is_code_review", False))
    user_text = get_last_user_text(s["messages"]).lower()
    current_year = datetime.now().year
    date_hints: list[str] = []
    if "next year" in user_text:
        date_hints.append(f"'next year' means {current_year + 1}")
    if "this year" in user_text:
        date_hints.append(f"'this year' means {current_year}")
    if "last year" in user_text:
        date_hints.append(f"'last year' means {current_year - 1}")

    date_hint_text = ""
    if date_hints:
        date_hint_text = "\n\nRelative-date resolution for this request: " + "; ".join(date_hints) + "."

    sys_msg = {
        "role": "system",
        "content": f"{_REACT_SYSTEM}\n\nFocus: {task_prompt}{date_hint_text}",
    }
    msgs = [sys_msg, *s["messages"]]

    try:
        model_can_use_tools = (
            TOOLS_ENABLED
            and model in TOOL_CAPABLE_MODELS
            and state.tool_registry
            and state.tool_registry.has_tools
        )

        if model_can_use_tools:
            tool_defs = state.tool_registry.tool_definitions
            max_rounds = s.get("react_max_rounds_override")
            if max_rounds is None:
                max_rounds = REACT_MAX_ROUNDS

            async def chat_fn(current_msgs):
                return await ollama_chat_once(
                    model,
                    current_msgs,
                    **model_call_kwargs(s),
                    tools=tool_defs,
                )

            content, final_msgs, tool_calls_log = await asyncio.wait_for(
                state.tool_registry.run_with_tools(
                    chat_fn,
                    msgs,
                    max_rounds=max(1, int(max_rounds)),
                ),
                timeout=FAST_PATH_TIMEOUT,
            )

            tool_rounds = sum(
                1
                for m in final_msgs
                if m.get("role") == "assistant" and m.get("tool_calls")
            )
            s["react_rounds"] = tool_rounds
            s["tools_used"] = tool_calls_log
            used_web_search, search_query = extract_web_search_info(tool_calls_log)
            s["search_performed"] = used_web_search
            s["search_query"] = search_query
            if tool_calls_log:
                tool_names = list(dict.fromkeys(t["tool"] for t in tool_calls_log))
                logger.info(
                    "ReAct tools used (%d calls): %s",
                    len(tool_calls_log),
                    ", ".join(tool_names),
                )
        else:
            content = await asyncio.wait_for(
                run_model_once(
                    model,
                    msgs,
                    **model_call_kwargs(s),
                ),
                timeout=FAST_PATH_TIMEOUT,
            )
            s["react_rounds"] = 0
            s["search_performed"] = False
            s["search_query"] = ""

        note_model_success(model)
        s["result_text"] = content
        s["selected_model"] = model
        s["prompt_tokens"] = estimate_tokens(flatten_messages(s["messages"]))
        s["completion_tokens"] = estimate_tokens(content)

    except Exception as e:
        note_model_failure(model)
        logger.warning(
            "ReAct agent failed for %s: %s — flagging for escalation", model, e
        )
        s["use_fast_path"] = False
        s["fast_model"] = ""
        s["result_text"] = ""
    return s


# ══════════════════════════════════════════════════════════════════════════════
#  Adaptive escalation — check quality and escalate if needed
# ══════════════════════════════════════════════════════════════════════════════

async def adaptive_escalate(s: dict[str, Any]) -> dict[str, Any]:
    """After fast path, check if response quality warrants escalation."""
    if not s.get("use_fast_path") or not s.get("result_text"):
        s["escalated"] = not s.get("result_text", "")
        return s

    result = s["result_text"]

    # Length-based escalation
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
                "Adaptive escalation: response too short (%d chars) for question (%d chars)",
                len(result.strip()),
                question_len,
            )
            s["use_fast_path"] = False
            s["escalated"] = True
            s["route_reason"] += " → escalated:short_response"
            return s

    # Reflection gate
    reflection_enabled = s.get("reflection_enabled_override")
    if reflection_enabled is None:
        reflection_enabled = REFLECTION_ENABLED

    if reflection_enabled:
        reflection = await reflect_on_response(
            s.get("original_messages", s["messages"]),
            result,
        )
        s["reflection_result"] = reflection

        if not reflection["complete"] or reflection["quality"] == "poor":
            retries = s.get("reflection_retries", 0)
            retries_limit = s.get("reflection_max_retries_override")
            if retries_limit is None:
                retries_limit = REFLECTION_MAX_RETRIES
            if retries < retries_limit:
                logger.info(
                    "Reflection: incomplete (missing: %s) — retry %d",
                    reflection["missing"][:60],
                    retries + 1,
                )
                s["reflection_retries"] = retries + 1

                model = s["fast_model"]
                retry_msgs = [
                    {"role": "system", "content": role_prompt(s["task_type"], model, is_code_review=s.get("is_code_review", False))},
                    *s["messages"],
                    {"role": "assistant", "content": result},
                    {
                        "role": "user",
                        "content": (
                            f"Your answer is incomplete. "
                            f"Missing: {reflection['missing']}. "
                            f"Please provide a more comprehensive response."
                        ),
                    },
                ]
                try:
                    improved = await asyncio.wait_for(
                        run_model_with_tools(
                            model,
                            retry_msgs,
                            max_tool_rounds=REACT_MAX_ROUNDS,
                            **model_call_kwargs(s),
                        ),
                        timeout=FAST_PATH_TIMEOUT,
                    )
                    s["result_text"] = improved
                    s["completion_tokens"] = estimate_tokens(improved)
                    s["route_reason"] += " → reflection_retry"
                    return s
                except Exception as e:
                    logger.warning("Reflection retry failed: %s — escalating", e)

            if reflection["quality"] == "poor":
                logger.info(
                    "Reflection: quality=poor — escalating to deep panel"
                )
                s["use_fast_path"] = False
                s["escalated"] = True
                s["route_reason"] += " → escalated:reflection_poor"
                return s

    s["escalated"] = False
    return s
