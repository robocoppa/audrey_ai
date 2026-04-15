"""
Audrey — LangGraph pipeline nodes and graph builders.

Contains every node function for the fast-path and deep-panel graphs,
plus the graph construction and compilation.
"""

import asyncio
import logging
import traceback
from typing import Any, Dict, List

from langgraph.graph import END, StateGraph

import state
from agents import adaptive_escalate, plan_sub_tasks, reflect_on_response, run_react_agent
from classifier import classify_request, select_fast_model, should_force_deep
from config import (
    DEEP_WORKER_TIMEOUT,
    FAST_PATH_CONFIDENCE,
    FAST_PATH_ENABLED,
    LOW_CONFIDENCE_THRESHOLD,
    MAX_DEEP_WORKERS,
    MAX_DEEP_WORKERS_CLOUD,
    REFLECTION_ENABLED,
    REFLECTION_MAX_RETRIES,
    deep_panel_for_model,
    is_cloud_model,
)
from health import is_model_healthy, note_model_failure, note_model_success
from helpers import (
    estimate_tokens,
    flatten_messages,
    inject_datetime,
    role_prompt,
)
from models import AudreyState
from ollama import run_model_once, run_model_with_tools
from helpers import get_last_user_text

logger = logging.getLogger("audrey.pipeline")


# ══════════════════════════════════════════════════════════════════════════════
#  Node: classify (+ complexity gate)
# ══════════════════════════════════════════════════════════════════════════════

async def node_classify(s):
    """Classify, inject datetime, decide fast vs deep (with complexity gate)."""
    s["messages"] = inject_datetime(s["messages"])
    s["original_messages"] = [m.copy() for m in s["messages"]]
    s.update(await classify_request(s["messages"]))

    # Agentic defaults
    s.setdefault("sub_tasks", None)
    s.setdefault("react_rounds", 0)
    s.setdefault("reflection_result", {})
    s.setdefault("reflection_retries", 0)
    s.setdefault("escalated", False)
    s.setdefault("is_code_review", False)

    # Search is now tool-only; initialize fields for compatibility
    s.setdefault("search_performed", False)
    s.setdefault("search_query", "")
    s.setdefault("search_results", [])

    # Default: no fast path
    s["use_fast_path"] = False
    s["fast_model"] = ""

    # "audrey_deep" tries fast path when confidence is high enough
    requested = s["requested_model"]
    if FAST_PATH_ENABLED and requested == "audrey_deep":
        if s["confidence"] >= FAST_PATH_CONFIDENCE:
            fm = select_fast_model(s["task_type"])
            if fm:
                # ── Complexity gate ──────────────────────────────────
                # Force deep panel when input is large or complex,
                # regardless of confidence.  This is the fix for
                # "paste main.py and ask to assess it" going fast path.
                force_reason = should_force_deep(
                    s["messages"], s["confidence"], s["task_type"]
                )
                if force_reason:
                    s["route_reason"] += f" → forced_deep:{force_reason}"
                    logger.info(
                        "Complexity gate: forcing deep panel — %s", force_reason
                    )
                else:
                    s["use_fast_path"] = True
                    s["fast_model"] = fm
                    s["route_reason"] += f" → fast:{fm}"

    return s


# ══════════════════════════════════════════════════════════════════════════════
#  Node: ReAct agent (fast path)
# ══════════════════════════════════════════════════════════════════════════════

async def node_react_agent(s):
    return await run_react_agent(s)


# ══════════════════════════════════════════════════════════════════════════════
#  Node: adaptive escalation
# ══════════════════════════════════════════════════════════════════════════════

async def node_adaptive_escalate(s):
    return await adaptive_escalate(s)


# ══════════════════════════════════════════════════════════════════════════════
#  Node: plan panel (worker / synthesizer selection)
# ══════════════════════════════════════════════════════════════════════════════

async def node_plan_panel(s):
    tt = s["task_type"]
    if s["confidence"] < LOW_CONFIDENCE_THRESHOLD:
        tt = "general"
        s["task_type"] = "general"
    panel = deep_panel_for_model(s["requested_model"])[tt]

    requested = s["requested_model"]
    all_workers = panel["workers"]

    # ── Worker selection with independent cloud/local caps ──
    # Walk the full config list in order, filling cloud and local buckets
    # independently.  Unhealthy or unavailable models are skipped and
    # later candidates backfill their slot.
    #
    # audrey_cloud:  only cloud workers, capped by MAX_DEEP_WORKERS_CLOUD
    # audrey_local:  only local workers, capped by MAX_DEEP_WORKERS
    # audrey_deep:   both, each capped independently

    if requested == "audrey_cloud":
        max_cloud, max_local = MAX_DEEP_WORKERS_CLOUD, 0
    elif requested == "audrey_local":
        max_cloud, max_local = 0, MAX_DEEP_WORKERS
    else:
        # audrey_deep — mixed: independent caps
        max_cloud, max_local = MAX_DEEP_WORKERS_CLOUD, MAX_DEEP_WORKERS

    selected = []
    cloud_count = 0
    local_count = 0

    for w in all_workers:
        cloud = is_cloud_model(w)

        # Check bucket capacity
        if cloud and cloud_count >= max_cloud:
            continue
        if not cloud and local_count >= max_local:
            continue

        # Health + availability gate
        if not is_model_healthy(w):
            logger.debug("Worker %s skipped: unhealthy (cooldown)", w)
            continue
        if not cloud and w not in state.available_models:
            logger.debug("Worker %s skipped: not in available_models", w)
            continue

        selected.append(w)
        if cloud:
            cloud_count += 1
        else:
            local_count += 1

    # Last resort: if health filtering emptied the list entirely,
    # fall back to the raw config list with the same type caps so we
    # at least try, but don't exceed intended cloud/local balance.
    if not selected:
        logger.warning(
            "All workers unhealthy for %s/%s — falling back to raw config",
            requested, tt,
        )
        fc, fl = 0, 0
        for w in all_workers:
            cloud = is_cloud_model(w)
            if cloud and fc < max_cloud:
                selected.append(w)
                fc += 1
            elif not cloud and fl < max_local:
                selected.append(w)
                fl += 1

    s["deep_workers"] = selected
    s["synthesizer"] = panel["synthesizer"]
    s["fallback_synthesizer"] = panel.get("fallback_synthesizer", "")

    sub_tasks = await plan_sub_tasks(s["messages"], tt)
    s["sub_tasks"] = sub_tasks

    logger.info(
        "Panel plan: %s/%s → %d workers %s  synth=%s",
        requested, tt, len(selected),
        [w for w in selected],
        s["synthesizer"],
    )
    return s


# ══════════════════════════════════════════════════════════════════════════════
#  Node: parallel generate (deep workers)
# ══════════════════════════════════════════════════════════════════════════════

async def node_parallel_generate(s):
    base = s["messages"]
    sub_tasks = s.get("sub_tasks")
    is_review = s.get("is_code_review", False)

    async def one(wn, sub_task=None):
        if sub_task:
            sys_content = (
                f"{role_prompt(s['task_type'], wn, structured=True, is_code_review=is_review)}\n\n"
                f"You are assigned this specific sub-task:\n{sub_task}\n\n"
                f"Focus your answer on this sub-task only. Be thorough and specific."
            )
        else:
            sys_content = role_prompt(s["task_type"], wn, structured=True, is_code_review=is_review)

        sys = {"role": "system", "content": sys_content}
        try:
            t = await asyncio.wait_for(
                run_model_with_tools(
                    wn,
                    [sys, *base],
                    temperature=s["temperature"],
                    max_tokens=s.get("max_tokens"),
                    top_p=s.get("top_p"),
                    stop=s.get("stop"),
                    frequency_penalty=s.get("frequency_penalty"),
                    presence_penalty=s.get("presence_penalty"),
                ),
                timeout=DEEP_WORKER_TIMEOUT,
            )
            note_model_success(wn)
            label = f" [sub-task: {sub_task[:50]}]" if sub_task else ""
            return {
                "model": wn,
                "content": t,
                "sub_task": sub_task or "",
                "label": label,
            }
        except Exception as e:
            note_model_failure(wn)
            logger.warning(
                "Worker %s failed: %s\n%s", wn, e, traceback.format_exc()
            )
            return {
                "model": wn,
                "content": "[WORKER_ERROR] Unable to respond.",
                "sub_task": sub_task or "",
            }

    workers = s["deep_workers"]

    if not workers:
        logger.error("node_parallel_generate called with empty workers list")
        s["worker_outputs"] = [{
            "model": "none",
            "content": "[WORKER_ERROR] No workers available.",
            "sub_task": "",
        }]
        return s

    if sub_tasks and len(sub_tasks) >= 2:
        assignments = []
        for i, task in enumerate(sub_tasks):
            worker = workers[i % len(workers)]
            assignments.append((worker, task))
        logger.info(
            "Planning: %d sub-tasks assigned to %d workers",
            len(assignments),
            len(workers),
        )
    else:
        assignments = [(w, None) for w in workers]

    cloud_tasks = [(w, t) for w, t in assignments if is_cloud_model(w)]
    local_tasks = [(w, t) for w, t in assignments if not is_cloud_model(w)]

    async def run_local():
        results = []
        for w, t in local_tasks:
            results.append(await one(w, t))
        return results

    if cloud_tasks and local_tasks:
        cr, lr = await asyncio.gather(
            asyncio.gather(*[one(w, t) for w, t in cloud_tasks]),
            run_local(),
        )
        outs = list(cr) + lr
    elif cloud_tasks:
        outs = list(await asyncio.gather(*[one(w, t) for w, t in cloud_tasks]))
    elif local_tasks:
        outs = await run_local()
    else:
        outs = []

    valid = [o for o in outs if not o["content"].startswith("[WORKER_ERROR]")]
    s["worker_outputs"] = valid or outs
    return s


# ══════════════════════════════════════════════════════════════════════════════
#  Synthesis
# ══════════════════════════════════════════════════════════════════════════════

_SYNTH_SYS = """You are a synthesis model. Merge expert drafts into one coherent, comprehensive response.
Do NOT mention model names, draft numbers, or that multiple sources were consulted.
Prefer correctness and clarity. Resolve any contradictions by favoring the more detailed answer.
If drafts reference tool results or web search data, integrate that information naturally.
If drafts address different sub-tasks of a complex question, combine them into a unified answer."""

_SYNTH_SYS_REVIEW = """You are a synthesis model merging code review drafts into one clear report.

Ranking rules (follow strictly):
1. Behavior mismatches and correctness bugs come first — things that produce wrong results or crash at runtime.
2. Logic errors, race conditions, data-loss risks, and missing error handling come next.
3. Design issues (poor separation of concerns, tight coupling, missing abstractions) follow.
4. Low-priority section at the end for: theoretical security hardening (timing attacks on localhost, magic-number extraction, structured-logging preferences), style nits, and naming suggestions.

Do NOT lead with security checklist items unless the code has a genuine exploitable vulnerability with a concrete attack path. For local-first application code, bugs and behavior matter more than compliance.

Do NOT mention model names, draft numbers, or that multiple sources were consulted.
Be specific: cite the function or line, explain the concrete failure mode, and suggest a fix.
If drafts contradict on severity, favor the one that identifies a concrete failure scenario over the one that cites a generic best practice."""


def build_synth_msgs(s):
    outputs = s["worker_outputs"]
    is_review = s.get("is_code_review", False)
    sys_prompt = _SYNTH_SYS_REVIEW if is_review else _SYNTH_SYS

    secs = []
    for i, o in enumerate(outputs, 1):
        label = o.get("label", "")
        secs.append(f"── Draft {i} ({o['model']}){label} ──\n{o['content']}")
    return [
        {"role": "system", "content": sys_prompt},
        {
            "role": "user",
            "content": (
                f"Original:\n\n{flatten_messages(s['messages'])}\n\nDrafts:\n\n"
                + "\n\n---\n\n".join(secs)
            ),
        },
    ]


async def node_prepare_synthesis(s):
    s["synthesis_messages"] = build_synth_msgs(s)
    return s


async def node_synthesize(s):
    if "synthesis_messages" not in s:
        s["synthesis_messages"] = build_synth_msgs(s)
    synths = [s["synthesizer"]]
    fb = s.get("fallback_synthesizer", "")
    if fb:
        synths.append(fb)
    for sy in synths:
        try:
            r = await run_model_once(
                sy,
                s["synthesis_messages"],
                temperature=min(s["temperature"], 0.3),
                max_tokens=s.get("max_tokens"),
                top_p=s.get("top_p"),
                stop=s.get("stop"),
                frequency_penalty=s.get("frequency_penalty"),
                presence_penalty=s.get("presence_penalty"),
            )
            s["result_text"] = r
            s["selected_model"] = sy
            s["prompt_tokens"] = estimate_tokens(flatten_messages(s["messages"]))
            s["completion_tokens"] = estimate_tokens(r)
            note_model_success(sy)
            return s
        except Exception as e:
            note_model_failure(sy)
            if sy == synths[-1]:
                raise RuntimeError(f"Synthesizers failed: {e}")
    raise RuntimeError("No synthesizers")


async def node_reflect_deep(s):
    """Reflection gate for deep panel output."""
    if not REFLECTION_ENABLED or not s.get("result_text"):
        return s

    reflection = await reflect_on_response(
        s.get("original_messages", s["messages"]),
        s["result_text"],
    )
    s["reflection_result"] = reflection

    if reflection["quality"] == "poor" and s.get("reflection_retries", 0) < REFLECTION_MAX_RETRIES:
        s["reflection_retries"] = s.get("reflection_retries", 0) + 1
        logger.info("Deep reflection: quality=poor — re-synthesizing with feedback")

        s["synthesis_messages"].append(
            {
                "role": "user",
                "content": (
                    f"The previous synthesis was rated as poor quality. "
                    f"Missing: {reflection['missing']}. "
                    f"Please re-synthesize with more attention to completeness and accuracy."
                ),
            }
        )
        return await node_synthesize(s)

    return s


# ══════════════════════════════════════════════════════════════════════════════
#  Graph builders
# ══════════════════════════════════════════════════════════════════════════════

def build_deep_graph(include_synthesis: bool = True):
    """Build the deep-panel pipeline graph with reflection.

    Search is handled by tool-calling during worker generation, not as
    a separate pipeline node.
    """
    g = StateGraph(AudreyState)
    for n, f in [
        ("classify", node_classify),
        ("plan", node_plan_panel),
        ("parallel", node_parallel_generate),
        ("prep", node_prepare_synthesis),
    ]:
        g.add_node(n, f)
    g.set_entry_point("classify")
    g.add_edge("classify", "plan")
    g.add_edge("plan", "parallel")
    g.add_edge("parallel", "prep")
    if include_synthesis:
        g.add_node("synth", node_synthesize)
        g.add_node("reflect_deep", node_reflect_deep)
        g.add_edge("prep", "synth")
        g.add_edge("synth", "reflect_deep")
        g.add_edge("reflect_deep", END)
    else:
        g.add_edge("prep", END)
    return g.compile()


def _should_run_react(s) -> str:
    """Conditional edge: run ReAct agent only when fast path is active."""
    if s.get("use_fast_path") and s.get("fast_model"):
        return "react_agent"
    return "escalate"


def build_fast_graph():
    """Build the fast path with ReAct agent + adaptive escalation.

    Uses a conditional edge after classify so that react_agent is
    only invoked when classify actually selected a fast model.
    Search is handled by tool-calling inside the ReAct agent.
    """
    g = StateGraph(AudreyState)
    for n, f in [
        ("classify", node_classify),
        ("react_agent", node_react_agent),
        ("escalate", node_adaptive_escalate),
    ]:
        g.add_node(n, f)
    g.set_entry_point("classify")
    g.add_conditional_edges(
        "classify",
        _should_run_react,
        {"react_agent": "react_agent", "escalate": "escalate"},
    )
    g.add_edge("react_agent", "escalate")
    g.add_edge("escalate", END)
    return g.compile()


# Pre-compiled graphs (imported by main.py)
SYNTH_GRAPH = build_deep_graph(True)
PREP_GRAPH = build_deep_graph(False)
FAST_GRAPH = build_fast_graph()
