"""
Audrey — helper utilities.

Message manipulation, request state helpers, token estimation, datetime
injection, and worker role prompts.
"""

import time
import re
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from config import TIMEOUTS, is_cloud_model


# ── Request state / model options ────────────────────────────────────────────

def ensure_state_defaults(state: Dict[str, Any]) -> Dict[str, Any]:
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
    state.setdefault("react_rounds", 0)
    state.setdefault("reflection_result", {})
    state.setdefault("reflection_retries", 0)
    state.setdefault("escalated", False)
    state.setdefault("tools_used", [])
    state.setdefault("is_code_review", False)
    return state


def build_initial_state(
    *,
    request_id: str,
    requested_model: str,
    messages: List[Dict[str, Any]],
    stream: bool,
    temperature: float,
    max_tokens: Optional[int],
    top_p: Optional[float],
    stop: Optional[Any],
    frequency_penalty: Optional[float],
    presence_penalty: Optional[float],
) -> Dict[str, Any]:
    """Build the shared initial pipeline state for a request."""
    state = {
        "request_id": request_id,
        "requested_model": requested_model,
        "messages": messages,
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


def model_call_kwargs(
    state: Dict[str, Any],
    *,
    temperature: Optional[float] = None,
) -> Dict[str, Any]:
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

def flatten_messages(msgs: List[Dict[str, Any]]) -> str:
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


def extract_web_search_info(tool_calls_log: List[Dict[str, Any]]) -> tuple[bool, str]:
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


def has_vision_content(msgs: List[Dict[str, Any]]) -> bool:
    for m in msgs:
        c = m.get("content")
        if isinstance(c, list):
            for i in c:
                if isinstance(i, dict) and i.get("type") == "image_url":
                    return True
    return False


def get_last_user_text(msgs: List[Dict[str, Any]]) -> str:
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

def _datetime_system_message() -> Dict[str, str]:
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


def inject_datetime(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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
