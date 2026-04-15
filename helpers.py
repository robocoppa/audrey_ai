"""
Audrey — helper utilities.

Message manipulation, request state helpers, token estimation, datetime
injection, and worker role prompts.
"""

import time
import re
import json
from datetime import datetime, timezone
from typing import Any

from config import TIMEOUTS, is_cloud_model


AUDREY_MODES = {"quick", "balanced", "research"}

_TIME_SENSITIVE_PATTERNS = [
    re.compile(
        r"\b(today|tonight|yesterday|this week|this month|right now|currently|latest|recent)\b",
        re.I,
    ),
    re.compile(r"\b(who is|who are)\b.{0,30}\b(current|president|ceo|mayor)\b", re.I),
    re.compile(
        r"\b(what is the|what's the)\b.{0,20}\b(price|score|weather|rate|status)\b",
        re.I,
    ),
    re.compile(r"\b(when is|when does|when will)\b", re.I),
    re.compile(r"\b(news|headline|update|announcement|released|launched)\b", re.I),
    re.compile(r"\b(stock|market|crypto|bitcoin)\b.{0,20}\b(price|value|worth)\b", re.I),
    re.compile(r"\b(score|won|lost|beat|defeated|playoff|standings)\b", re.I),
    re.compile(r"\b(election|voted|poll|ballot)\b", re.I),
    re.compile(r"\b20(2[5-9]|[3-9]\d)\b"),
]


# ── Request state / model options ────────────────────────────────────────────

def ensure_state_defaults(state: dict[str, Any]) -> dict[str, Any]:
    """Populate optional state fields used across fast and deep flows."""
    state.setdefault("errors", [])
    state.setdefault("prompt_tokens", 0)
    state.setdefault("completion_tokens", 0)
    state.setdefault("search_performed", False)
    state.setdefault("search_query", "")
    state.setdefault("search_results", [])
    state.setdefault("use_fast_path", False)
    state.setdefault("fast_model", "")
    state.setdefault("sub_tasks", None)
    state.setdefault("worker_error_count", 0)
    state.setdefault("react_rounds", 0)
    state.setdefault("reflection_result", {})
    state.setdefault("reflection_retries", 0)
    state.setdefault("force_strong_synth", False)
    state.setdefault("escalated", False)
    state.setdefault("tools_used", [])
    state.setdefault("synthesis_candidates", [])
    state.setdefault("synthesis_strategy", "configured_first")
    state.setdefault("synthesis_escalation_reason", "")
    state.setdefault("is_code_review", False)
    state.setdefault("audrey_mode", "balanced")
    state.setdefault("timeline", [])
    state.setdefault("cache_hit", False)
    state.setdefault("needs_fresh_data", False)
    state.setdefault("fast_path_confidence", None)
    state.setdefault("force_deep_profile", False)
    state.setdefault("planning_enabled_override", None)
    state.setdefault("planning_min_tokens_override", None)
    state.setdefault("reflection_enabled_override", None)
    state.setdefault("reflection_max_retries_override", None)
    state.setdefault("react_max_rounds_override", None)
    return state


def build_initial_state(
    *,
    request_id: str,
    requested_model: str,
    messages: list[dict[str, Any]],
    audrey_mode: str,
    stream: bool,
    temperature: float,
    max_tokens: int | None,
    top_p: float | None,
    stop: Any | None,
    frequency_penalty: float | None,
    presence_penalty: float | None,
) -> dict[str, Any]:
    """Build the shared initial pipeline state for a request."""
    state = {
        "request_id": request_id,
        "requested_model": requested_model,
        "messages": messages,
        "audrey_mode": normalize_audrey_mode(audrey_mode),
        "stream": stream,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "stop": stop,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
        "started_at": time.time(),
    }
    return ensure_state_defaults(state)


def normalize_audrey_mode(mode: str | None) -> str:
    raw = str(mode or "balanced").strip().lower()
    return raw if raw in AUDREY_MODES else "balanced"


def is_time_sensitive_query(text: str) -> bool:
    candidate = str(text or "").strip()
    if not candidate:
        return False
    for pattern in _TIME_SENSITIVE_PATTERNS:
        if pattern.search(candidate):
            return True
    return False


def append_timeline_event(
    state: dict[str, Any],
    *,
    stage: str,
    message: str,
    status: str = "info",
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    event: dict[str, Any] = {
        "at_ms": int(time.time() * 1000),
        "stage": stage,
        "status": status,
        "message": message,
    }
    if details:
        event["details"] = details
    state.setdefault("timeline", []).append(event)
    return event


def build_trust_signals(state: dict[str, Any]) -> dict[str, Any]:
    tools_used = state.get("tools_used", []) or []
    distinct_tools = sorted(
        {
            str(item.get("tool", "")).split("__", 1)[-1]
            for item in tools_used
            if item.get("tool")
        }
    )
    source_count = 0
    for item in tools_used:
        try:
            source_count += int(item.get("result_url_count", 0) or 0)
        except (TypeError, ValueError):
            continue

    freshness = "cached" if state.get("cache_hit") else (
        "live_web" if state.get("search_performed") else "model_only"
    )
    if (
        state.get("needs_fresh_data")
        and not state.get("search_performed")
        and not state.get("cache_hit")
    ):
        freshness = "potentially_stale"

    path = "deep"
    if state.get("cache_hit"):
        path = "cache"
    elif state.get("requested_model") == "audrey_fast" or state.get("use_fast_path"):
        path = "fast"

    return {
        "path": path,
        "confidence": round(float(state.get("confidence", 0.0) or 0.0), 3),
        "freshness": freshness,
        "search_used": bool(state.get("search_performed")),
        "search_query": str(state.get("search_query", "") or ""),
        "tools_used_count": len(tools_used),
        "tools_used": distinct_tools,
        "source_count": source_count,
        "reflection_quality": str(
            (state.get("reflection_result") or {}).get("quality", "n/a")
        ),
        "synthesis_strategy": str(state.get("synthesis_strategy", "")),
        "synthesis_escalation_reason": str(
            state.get("synthesis_escalation_reason", "")
        ),
        "escalated": bool(state.get("escalated", False)),
        "cache_hit": bool(state.get("cache_hit", False)),
    }


def model_call_kwargs(
    state: dict[str, Any],
    *,
    temperature: float | None = None,
) -> dict[str, Any]:
    """Shared generation kwargs for model calls."""
    return {
        "temperature": state["temperature"] if temperature is None else temperature,
        "max_tokens": state.get("max_tokens"),
        "top_p": state.get("top_p"),
        "stop": state.get("stop"),
        "frequency_penalty": state.get("frequency_penalty"),
        "presence_penalty": state.get("presence_penalty"),
    }


# ── Timeout for a model ──────────────────────────────────────────────────────

def timeout_for_model(name: str, *, is_router: bool = False) -> int:
    if is_router:
        return TIMEOUTS.get("router", 20)
    if is_cloud_model(name):
        return TIMEOUTS.get("cloud", 120)
    m = re.search(r"(\d+(\.\d+)?)b", name.lower())
    if m:
        p = float(m.group(1))
        if p <= 3:
            return TIMEOUTS.get("small", 60)
        if p <= 14:
            return TIMEOUTS.get("medium", 180)
    return TIMEOUTS.get("large", 360)


# ── Token estimation ─────────────────────────────────────────────────────────

def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    ce = len(text) / 3.5
    we = len(text.split()) * 1.3
    cr = sum(1 for c in text if c in "{}[]();=<>|&!@#$%^*~/\\") / max(len(text), 1)
    return max(1, int(ce * 0.7 + we * 0.3 if cr > 0.05 else ce * 0.3 + we * 0.7))


# ── Message helpers ──────────────────────────────────────────────────────────

def flatten_messages(msgs: list[dict[str, Any]]) -> str:
    parts = []
    for m in msgs:
        c = m.get("content", "")
        if isinstance(c, list):
            c = "\n".join(
                i.get("text", "")
                if isinstance(i, dict) and i.get("type") == "text"
                else "[image]"
                if isinstance(i, dict) and i.get("type") == "image_url"
                else str(i)
                for i in c
            )
        parts.append(f"{m.get('role', 'user').upper()}: {c}")
    return "\n\n".join(parts)


def extract_web_search_info(tool_calls_log: list[dict[str, Any]]) -> tuple[bool, str]:
    """Return (used_web_search, query) from tool observability logs."""
    if not tool_calls_log:
        return False, ""

    for call in tool_calls_log:
        tool_name = str(call.get("tool", "")).lower()
        # Primary signal for the bundled custom tools server.
        is_web_search = (
            tool_name == "web_search"
            or tool_name.endswith("__web_search")
            or "web_search" in tool_name
        )
        if not is_web_search:
            continue

        query = str(call.get("query", "")).strip()
        if query:
            return True, query

        # Backward-compatible fallback for older logs without a dedicated query field.
        args_preview = str(call.get("args_preview", "")).strip()
        if not args_preview:
            return True, ""
        try:
            parsed = json.loads(args_preview)
            if isinstance(parsed, dict):
                query = str(parsed.get("query", "")).strip()
                if query:
                    return True, query
        except (json.JSONDecodeError, TypeError):
            pass
        return True, ""

    return False, ""


def has_vision_content(msgs: list[dict[str, Any]]) -> bool:
    for m in msgs:
        c = m.get("content")
        if isinstance(c, list):
            for i in c:
                if isinstance(i, dict) and i.get("type") == "image_url":
                    return True
    return False


def get_last_user_text(msgs: list[dict[str, Any]]) -> str:
    """Extract the text of the last user message."""
    for m in reversed(msgs):
        if m.get("role") == "user":
            c = m.get("content", "")
            if isinstance(c, list):
                for i in c:
                    if isinstance(i, dict) and i.get("type") == "text":
                        return i.get("text", "")
            else:
                return str(c)
    return ""


# ── Datetime injection ───────────────────────────────────────────────────────

def _datetime_system_message() -> dict[str, str]:
    now = datetime.now()
    utcnow = datetime.now(timezone.utc)
    return {
        "role": "system",
        "content": (
            f"Current date and time: {now.strftime('%A, %B %d, %Y at %I:%M %p')} "
            f"(local server time). UTC: {utcnow.strftime('%A, %B %d, %Y at %H:%M UTC')}. "
            f"Use this for any questions involving relative dates like 'today', 'yesterday', "
            f"'this week', 'last month', etc."
        ),
    }


def inject_datetime(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Prepend a datetime system message to the conversation."""
    return [_datetime_system_message()] + messages


# ── Worker role prompts ──────────────────────────────────────────────────────

_WSF = "\n\nStructure: ## Approach\n## Answer\n## Caveats\n"

_WSF_REVIEW = (
    "\n\nStructure: ## Key Issues (ranked by user impact)\n"
    "## Detailed Findings\n## Minor / Low-Priority Notes\n"
)

_CODE_REVIEW_BASE = (
    "You are reviewing code. Prioritize bugs, correctness errors, and behavior "
    "mismatches — things that will break at runtime or produce wrong results for "
    "the user. Rank every finding by user impact, not by generic security severity.\n\n"
    "Demote theoretical hardening (timing attacks on local services, structured "
    "logging preferences, magic-number extraction) to a low-priority section unless "
    "no larger issues exist. For local-first application code, bugs and behavior "
    "matter more than compliance checklists.\n\n"
    "Be specific: cite the function or line, explain the concrete failure mode, "
    "and suggest a fix."
)


def role_prompt(
    task_type: str,
    worker_name: str,
    structured: bool = False,
    *,
    is_code_review: bool = False,
) -> str:
    if is_code_review:
        b = _CODE_REVIEW_BASE
        return b + _WSF_REVIEW if structured else b

    if task_type == "code":
        b = (
            "Focus on implementation, correctness, bugs."
            if "coder" in worker_name
            else "Focus on reasoning, tradeoffs, edge cases."
            if "deepseek" in worker_name
            else "Clearest practical technical answer."
        )
    elif task_type == "reasoning":
        b = (
            "Reason step by step."
            if "deepseek" in worker_name or "cogito" in worker_name
            else "Clarity, practicality, actionable conclusion."
        )
    elif task_type == "vl":
        b = (
            "Interpret visual content accurately."
            if "vl" in worker_name or "llava" in worker_name
            else "Clear explanation from visual analysis."
        )
    else:
        b = "Clearest, most useful answer."
    return b + _WSF if structured else b
