"""
Audrey — agentic features.

Planning (sub-task decomposition), reflection gates, ReAct agent loop,
and adaptive escalation from fast path to deep panel.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

import state
from classifier import _extract_json
from config import (
    ESCALATION_CONFIDENCE_CEILING,
    ESCALATION_ENABLED,
    ESCALATION_MIN_LENGTH,
    FAST_PATH_TIMEOUT,
    PLANNING_ENABLED,
    PLANNING_MIN_TOKENS,
    REFLECTION_ENABLED,
    REFLECTION_MAX_RETRIES,
    ROUTER_MODEL,
    TOOL_CAPABLE_MODELS,
    TOOLS_ENABLED,
)
from health import note_model_failure, note_model_success
from helpers import (
    estimate_tokens,
    flatten_messages,
    get_last_user_text,
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
    msgs: List[Dict[str, Any]], task_type: str
) -> Optional[List[str]]:
    """Use the router model to optionally decompose a complex query."""
    if not PLANNING_ENABLED:
        return None

    user_text = get_last_user_text(msgs)
    if estimate_tokens(user_text) < PLANNING_MIN_TOKENS:
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
    original_msgs: List[Dict[str, Any]],
    response_text: str,
) -> Dict[str, Any]:
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

Be thorough but efficient — use tools only when they add value."""


async def run_react_agent(s: Dict[str, Any]) -> Dict[str, Any]:
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

    sys_msg = {
        "role": "system",
        "content": f"{_REACT_SYSTEM}\n\nFocus: {task_prompt}",
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

            content, final_msgs, tool_calls_log = await asyncio.wait_for(
                state.tool_registry.run_with_tools(chat_fn, msgs),
                timeout=FAST_PATH_TIMEOUT,
            )

            tool_rounds = sum(
                1
                for m in final_msgs
                if m.get("role") == "assistant" and m.get("tool_calls")
            )
            s["react_rounds"] = tool_rounds
            s["tools_used"] = tool_calls_log
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
                    temperature=s["temperature"],
                    max_tokens=s.get("max_tokens"),
                    top_p=s.get("top_p"),
                    stop=s.get("stop"),
                    frequency_penalty=s.get("frequency_penalty"),
                    presence_penalty=s.get("presence_penalty"),
                ),
                timeout=FAST_PATH_TIMEOUT,
            )
            s["react_rounds"] = 0

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

async def adaptive_escalate(s: Dict[str, Any]) -> Dict[str, Any]:
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
    if REFLECTION_ENABLED:
        reflection = await reflect_on_response(
            s.get("original_messages", s["messages"]),
            result,
        )
        s["reflection_result"] = reflection

        if not reflection["complete"] or reflection["quality"] == "poor":
            retries = s.get("reflection_retries", 0)
            if retries < REFLECTION_MAX_RETRIES:
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
                            temperature=s["temperature"],
                            max_tokens=s.get("max_tokens"),
                            top_p=s.get("top_p"),
                            stop=s.get("stop"),
                            frequency_penalty=s.get("frequency_penalty"),
                            presence_penalty=s.get("presence_penalty"),
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
