"""
Audrey — pipeline nodes.

Contains every node function for the fast-path and deep-panel flows,
worker selection, synthesis routing, and reflection.
"""

import asyncio
import logging
import os
import re
import traceback
from typing import Any

import aiohttp

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
    OLLAMA_LOCAL_UNLOAD_BETWEEN_WORKERS,
    REFLECTION_ENABLED,
    REFLECTION_MAX_RETRIES,
    TOOL_SERVER_URLS,
    deep_panel_for_model,
    is_cloud_model,
)
from health import is_model_healthy, note_model_failure, note_model_success
from helpers import (
    extract_web_search_info,
    ensure_state_defaults,
    estimate_tokens,
    flatten_messages,
    get_last_user_text,
    has_vision_content,
    inject_datetime,
    model_call_kwargs,
    role_prompt,
)
from ollama import run_model_once, run_model_with_tools_detailed

logger = logging.getLogger("audrey.pipeline")
_WORKER_ERROR_PREFIX = "[WORKER_ERROR]"


# ══════════════════════════════════════════════════════════════════════════════
#  Mode helpers (audrey_research, audrey_knowledge)
# ══════════════════════════════════════════════════════════════════════════════

_MATH_PREAMBLE = (
    "You are in math mode. For this request:\n"
    "- Show your work step by step. Do not skip algebraic steps.\n"
    "- State definitions, assumptions, domain constraints, and units up front.\n"
    "- Use LaTeX for all math: inline $x$ and display $$...$$. Never use plain ASCII like x^2 when LaTeX is cleaner.\n"
    "- Verify arithmetic. When you compute a numeric answer, sanity-check it by substitution or an alternative method.\n"
    "- For proofs, name the technique (induction, contradiction, construction, etc.) and justify each step.\n"
    "- If the problem is ambiguous, state your interpretation before solving.\n"
    "- End with the final answer on its own line, boxed: $\\boxed{answer}$."
)


_RESEARCH_PREAMBLE = (
    "You are in research mode. For this request:\n"
    "- Treat it as a structured investigation, not a quick answer.\n"
    "- Use web_search and other tools liberally to gather current evidence.\n"
    "- Resolve any relative dates (today, this year) to absolute values before searching.\n"
    "- Cite sources inline with the URL or title, and prefer primary sources.\n"
    "- Cover multiple angles and call out tradeoffs / dissenting views.\n"
    "- End with a concise summary and, when applicable, a recommendation."
)


def _build_knowledge_preamble(chunks: list[dict[str, Any]]) -> str:
    if not chunks:
        return (
            "You are in knowledge mode but no matching documents were found in "
            "the local knowledge base. Tell the user clearly that nothing in "
            "the indexed corpus matched, then offer to answer from general "
            "knowledge if they want."
        )
    excerpts = []
    for i, ch in enumerate(chunks, 1):
        fname = ch.get("filename") or ch.get("source_path", "source")
        coll = ch.get("collection", "") or ""
        content = (ch.get("content") or "").strip()
        header = f"[{i}] {fname}" + (f" ({coll})" if coll else "")
        excerpts.append(f"{header}\n{content}")
    body = "\n\n".join(excerpts)
    return (
        "You are in knowledge mode. Ground your answer in the retrieved excerpts below. "
        "Cite sources inline using the [N] markers that label each excerpt. "
        "If the excerpts do not answer the question, say so — do NOT fabricate details. "
        "Prefer quoting or paraphrasing the excerpts over inventing information.\n\n"
        "Retrieved excerpts:\n\n" + body
    )


def _knowledge_server_url() -> str | None:
    for url in TOOL_SERVER_URLS:
        if "knowledge" in url:
            return url.rstrip("/")
    return None


async def _fetch_knowledge_context(
    query: str, top_k: int = 6
) -> list[dict[str, Any]]:
    """Query the knowledge server's /search_knowledge endpoint."""
    base = _knowledge_server_url()
    if not base:
        logger.warning("audrey_knowledge: no knowledge-server URL configured")
        return []
    try:
        timeout = aiohttp.ClientTimeout(total=15)
        async with aiohttp.ClientSession(timeout=timeout) as sess:
            async with sess.post(
                f"{base}/search_knowledge",
                json={"query": query, "top_k": top_k},
            ) as r:
                if r.status != 200:
                    logger.warning(
                        "knowledge search failed: HTTP %d", r.status
                    )
                    return []
                data = await r.json()
                return data.get("results") or []
    except Exception as e:
        logger.warning("knowledge search error: %s", e)
        return []


def _maybe_prepend_tool_hint(s: dict[str, Any]) -> list[dict[str, Any]]:
    """Prepend a short system nudge when the user invoked a tool-specific
    slash command (e.g. /py, /read). Leaves messages untouched otherwise."""
    hint = s.get("tool_hint_preamble", "")
    if not hint:
        return s["messages"]
    return _prepend_system(s["messages"], hint)


def _prepend_system(
    msgs: list[dict[str, Any]], preamble: str
) -> list[dict[str, Any]]:
    """Prepend or extend the leading system message with a mode preamble."""
    if msgs and msgs[0].get("role") == "system":
        head = msgs[0]
        existing = head.get("content", "")
        if isinstance(existing, list):
            # Rare: multimodal system — just prepend a text part
            return [
                {**head, "content": [{"type": "text", "text": preamble}, *existing]},
                *msgs[1:],
            ]
        combined = (preamble + "\n\n" + (existing or "")).strip()
        return [{**head, "content": combined}, *msgs[1:]]
    return [{"role": "system", "content": preamble}, *msgs]



_AUDREY_CODE_REVIEW_HINT_RE = re.compile(
    r"\b(review|code review|audit|assess|assessment|evaluate|evaluation|critique|"
    r"feedback|improve|improvement|recommendation|recommendations|any issues|"
    r"any bugs|what's wrong|what is wrong|how does this look|clean up|"
    r"refactor suggestions)\b",
    re.I,
)


# ══════════════════════════════════════════════════════════════════════════════
#  Worker selection helpers
# ══════════════════════════════════════════════════════════════════════════════

def worker_limits_for_model(
    requested_model: str,
    *,
    audrey_mode: str = "balanced",
) -> tuple[int, int]:
    """Return the independent cloud/local worker caps for a virtual model."""
    if requested_model == "audrey_cloud":
        max_cloud, max_local = MAX_DEEP_WORKERS_CLOUD, 0
    elif requested_model == "audrey_local":
        max_cloud, max_local = 0, MAX_DEEP_WORKERS
    elif requested_model == "audrey_code":
        # Mixed profile that prefers local coder workers, with limited cloud assist.
        max_cloud, max_local = 1, MAX_DEEP_WORKERS
    else:
        max_cloud, max_local = MAX_DEEP_WORKERS_CLOUD, MAX_DEEP_WORKERS

    if audrey_mode == "quick":
        return min(max_cloud, 1), min(max_local, 1)
    return max_cloud, max_local


def _fits_worker_bucket(
    worker_name: str,
    *,
    cloud_count: int,
    local_count: int,
    max_cloud: int,
    max_local: int,
) -> bool:
    if is_cloud_model(worker_name):
        return cloud_count < max_cloud
    return local_count < max_local


def _is_worker_available(worker_name: str) -> bool:
    if not is_model_healthy(worker_name):
        logger.debug("Worker %s skipped: unhealthy (cooldown)", worker_name)
        return False
    if not is_cloud_model(worker_name) and worker_name not in state.available_models:
        logger.debug("Worker %s skipped: not in available_models", worker_name)
        return False
    return True


def _append_worker(
    selected: list[str],
    worker_name: str,
    *,
    cloud_count: int,
    local_count: int,
) -> tuple[int, int]:
    selected.append(worker_name)
    if is_cloud_model(worker_name):
        return cloud_count + 1, local_count
    return cloud_count, local_count + 1


def select_workers(
    requested_model: str,
    all_workers: list[str],
    *,
    task_type: str,
    audrey_mode: str = "balanced",
) -> list[str]:
    """Select deep workers using caps, health, and availability gates."""
    max_cloud, max_local = worker_limits_for_model(
        requested_model,
        audrey_mode=audrey_mode,
    )

    selected: list[str] = []
    cloud_count = 0
    local_count = 0

    for worker_name in all_workers:
        if not _fits_worker_bucket(
            worker_name,
            cloud_count=cloud_count,
            local_count=local_count,
            max_cloud=max_cloud,
            max_local=max_local,
        ):
            continue
        if not _is_worker_available(worker_name):
            continue
        cloud_count, local_count = _append_worker(
            selected,
            worker_name,
            cloud_count=cloud_count,
            local_count=local_count,
        )

    if selected:
        return selected

    logger.warning(
        "All workers unhealthy for %s/%s — falling back to raw config",
        requested_model,
        task_type,
    )
    cloud_count = 0
    local_count = 0
    for worker_name in all_workers:
        if not _fits_worker_bucket(
            worker_name,
            cloud_count=cloud_count,
            local_count=local_count,
            max_cloud=max_cloud,
            max_local=max_local,
        ):
            continue
        cloud_count, local_count = _append_worker(
            selected,
            worker_name,
            cloud_count=cloud_count,
            local_count=local_count,
        )
    return selected


# ══════════════════════════════════════════════════════════════════════════════
#  Node: classify (+ complexity gate)
# ══════════════════════════════════════════════════════════════════════════════

async def node_classify(s):
    """Classify, inject datetime, decide fast vs deep (with complexity gate)."""
    ensure_state_defaults(s)
    s["messages"] = inject_datetime(s["messages"])
    s["original_messages"] = [m.copy() for m in s["messages"]]
    requested = s["requested_model"]

    if requested == "audrey_code":
        # Explicit code profile: force code panel, but retain review intent.
        review_hint_text = flatten_messages(s["messages"])
        is_review = bool(_AUDREY_CODE_REVIEW_HINT_RE.search(review_hint_text))
        s.update(
            {
                "task_type": "code",
                "confidence": 1.0,
                "needs_vision": False,
                "is_code_review": is_review,
                "route_reason": (
                    "forced:audrey_code:review" if is_review else "forced:audrey_code"
                ),
            }
        )
        s["use_fast_path"] = False
        s["fast_model"] = ""
        s["messages"] = _maybe_prepend_tool_hint(s)
        return s

    if requested == "audrey_research":
        # Force deep panel + reasoning task type, inject research-focused prompt
        # and bump ReAct rounds so workers dig deeper with web search.
        s.update(
            {
                "task_type": "reasoning",
                "confidence": 1.0,
                "needs_vision": has_vision_content(s["messages"]),
                "is_code_review": False,
                "route_reason": "forced:audrey_research",
            }
        )
        s["use_fast_path"] = False
        s["fast_model"] = ""
        s["research_mode"] = True
        s["messages"] = _prepend_system(
            s["messages"], _RESEARCH_PREAMBLE,
        )
        s["messages"] = _maybe_prepend_tool_hint(s)
        return s

    if requested == "audrey_math":
        # Force deep math panel, inject rigorous step-by-step preamble.
        s.update(
            {
                "task_type": "math",
                "confidence": 1.0,
                "needs_vision": has_vision_content(s["messages"]),
                "is_code_review": False,
                "route_reason": "forced:audrey_math",
            }
        )
        s["use_fast_path"] = False
        s["fast_model"] = ""
        s["math_mode"] = True
        s["messages"] = _prepend_system(s["messages"], _MATH_PREAMBLE)
        s["messages"] = _maybe_prepend_tool_hint(s)
        return s

    if requested == "audrey_knowledge" and not s.get("disable_kb"):
        # Force deep panel + pre-fetch knowledge chunks, inject as context
        # so synthesis has retrieved material to cite from.
        s.update(
            {
                "task_type": "general",
                "confidence": 1.0,
                "needs_vision": has_vision_content(s["messages"]),
                "is_code_review": False,
                "route_reason": "forced:audrey_knowledge",
            }
        )
        s["use_fast_path"] = False
        s["fast_model"] = ""
        s["knowledge_mode"] = True
        query = get_last_user_text(s["messages"])
        chunks = await _fetch_knowledge_context(query) if query else []
        s["knowledge_chunks"] = chunks
        preamble = _build_knowledge_preamble(chunks)
        s["messages"] = _prepend_system(s["messages"], preamble)
        s["messages"] = _maybe_prepend_tool_hint(s)
        return s

    s.update(await classify_request(s["messages"]))

    # Per-turn user overrides: /kb (force_kb) fetches KB context even when
    # the selected model isn't audrey_knowledge. Does not force the deep
    # panel — fast path can still run if the classifier picked it.
    if s.get("force_kb") and not s.get("disable_kb") and not s.get("knowledge_mode"):
        query = get_last_user_text(s["messages"])
        chunks = await _fetch_knowledge_context(query) if query else []
        s["knowledge_chunks"] = chunks
        s["knowledge_mode"] = True
        s["messages"] = _prepend_system(s["messages"], _build_knowledge_preamble(chunks))

    # Default: no fast path
    s["use_fast_path"] = False
    s["fast_model"] = ""

    # "audrey_fast" is explicitly fast-only: always attempt a fast model.
    if requested == "audrey_fast":
        if not FAST_PATH_ENABLED:
            s["route_reason"] += " → fast_only:disabled"
            s["messages"] = _maybe_prepend_tool_hint(s)
            return s
        fm = select_fast_model(s["task_type"])
        if fm:
            s["use_fast_path"] = True
            s["fast_model"] = fm
            s["route_reason"] += f" → fast_only:{fm}"
        else:
            s["route_reason"] += " → fast_only:no_healthy_model"
        s["messages"] = _maybe_prepend_tool_hint(s)
        return s

    # "audrey_deep" tries fast path when confidence is high enough
    confidence_threshold = float(
        s.get("fast_path_confidence", FAST_PATH_CONFIDENCE) or FAST_PATH_CONFIDENCE
    )
    if requested == "audrey_deep" and s.get("force_deep_profile"):
        s["route_reason"] += " → mode:research_force_deep"
        s["messages"] = _maybe_prepend_tool_hint(s)
        return s

    if FAST_PATH_ENABLED and requested == "audrey_deep":
        if s.get("needs_vision"):
            # Deep should synthesize across multiple vision workers when
            # the user uploads an image — skip the fast-path shortcut.
            s["route_reason"] += " → forced_deep:vision"
            logger.info("Forcing deep panel: image upload on audrey_deep")
        elif s["confidence"] >= confidence_threshold:
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

    s["messages"] = _maybe_prepend_tool_hint(s)
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
    if s["requested_model"] != "audrey_code" and s["confidence"] < LOW_CONFIDENCE_THRESHOLD:
        tt = "general"
        s["task_type"] = "general"
    panel = deep_panel_for_model(s["requested_model"])[tt]

    requested = s["requested_model"]
    mode = str(s.get("audrey_mode", "balanced"))
    selected = select_workers(
        requested,
        panel["workers"],
        task_type=tt,
        audrey_mode=mode,
    )

    s["deep_workers"] = selected
    s["synthesizer"] = panel["synthesizer"]
    s["fallback_synthesizer"] = panel.get("fallback_synthesizer", "")

    sub_tasks = await plan_sub_tasks(
        s["messages"],
        tt,
        audrey_mode=mode,
        planning_enabled_override=s.get("planning_enabled_override"),
        min_tokens_override=s.get("planning_min_tokens_override"),
    )
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
    progress_queue: asyncio.Queue | None = s.get("worker_progress_queue")
    planned_synth = str(s.get("synthesizer", "")).strip()
    fallback_synth = str(s.get("fallback_synthesizer", "")).strip()
    # Keep any model that may be chosen as the synthesizer resident; unload the
    # rest after their draft to avoid VRAM eviction thrash on tight hosts.
    warm_locals = {
        m for m in (planned_synth, fallback_synth)
        if m and not is_cloud_model(m)
    }

    def _worker_keep_alive(wn: str) -> Any | None:
        if not OLLAMA_LOCAL_UNLOAD_BETWEEN_WORKERS:
            return None
        if is_cloud_model(wn):
            return None
        if wn in warm_locals:
            return None
        return 0

    async def _emit(event: dict[str, Any]) -> None:
        if progress_queue is not None:
            await progress_queue.put(event)

    async def one(wn, sub_task=None):
        started = asyncio.get_event_loop().time()
        await _emit({"type": "worker_started", "model": wn, "sub_task": sub_task or ""})
        if sub_task:
            sys_content = (
                f"{role_prompt(s['task_type'], wn, structured=True, is_code_review=is_review)}\n\n"
                f"You are assigned this specific sub-task:\n{sub_task}\n\n"
                f"Focus your answer on this sub-task only. Be thorough and specific."
            )
        else:
            sys_content = role_prompt(s["task_type"], wn, structured=True, is_code_review=is_review)

        sys = {"role": "system", "content": sys_content}
        ka = _worker_keep_alive(wn)
        call_kwargs = model_call_kwargs(s)
        if ka is not None:
            call_kwargs["keep_alive"] = ka
        try:
            t, tool_calls_log = await asyncio.wait_for(
                run_model_with_tools_detailed(
                    wn,
                    [sys, *base],
                    disable_web_search=bool(s.get("disable_web_search")),
                    disable_kb=bool(s.get("disable_kb")),
                    **call_kwargs,
                ),
                timeout=DEEP_WORKER_TIMEOUT,
            )
            note_model_success(wn)
            elapsed_ms = int((asyncio.get_event_loop().time() - started) * 1000)
            await _emit({
                "type": "worker_finished",
                "model": wn,
                "status": "success",
                "elapsed_ms": elapsed_ms,
                "sub_task": sub_task or "",
            })
            label = f" [sub-task: {sub_task[:50]}]" if sub_task else ""
            return {
                "model": wn,
                "content": t,
                "sub_task": sub_task or "",
                "label": label,
                "tools_used": tool_calls_log,
            }
        except Exception as e:
            note_model_failure(wn)
            elapsed_ms = int((asyncio.get_event_loop().time() - started) * 1000)
            logger.warning(
                "Worker %s failed: %s\n%s", wn, e, traceback.format_exc()
            )
            await _emit({
                "type": "worker_finished",
                "model": wn,
                "status": "error",
                "elapsed_ms": elapsed_ms,
                "sub_task": sub_task or "",
                "error": str(e)[:160],
            })
            return {
                "model": wn,
                "content": f"{_WORKER_ERROR_PREFIX} Unable to respond.",
                "sub_task": sub_task or "",
                "tools_used": [],
            }

    workers = s["deep_workers"]

    if not workers:
        logger.error("node_parallel_generate called with empty workers list")
        s["worker_outputs"] = [{
            "model": "none",
            "content": f"{_WORKER_ERROR_PREFIX} No workers available.",
            "sub_task": "",
        }]
        s["worker_error_count"] = 1
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

    async def run_local_sequential():
        results = []
        for w, t in local_tasks:
            results.append(await one(w, t))
        return results

    if cloud_tasks and local_tasks:
        cr, lr = await asyncio.gather(
            asyncio.gather(*[one(w, t) for w, t in cloud_tasks]),
            run_local_sequential(),
        )
        outs = list(cr) + lr
    elif cloud_tasks:
        outs = list(await asyncio.gather(*[one(w, t) for w, t in cloud_tasks]))
    elif local_tasks:
        outs = await run_local_sequential()
    else:
        outs = []

    valid = [o for o in outs if not str(o.get("content", "")).startswith(_WORKER_ERROR_PREFIX)]
    s["worker_error_count"] = len(outs) - len(valid)
    selected_outputs = valid or outs
    s["worker_outputs"] = selected_outputs

    merged_tool_calls = []
    for out in selected_outputs:
        merged_tool_calls.extend(out.get("tools_used", []))
    s["tools_used"] = merged_tool_calls

    used_web_search, query = extract_web_search_info(merged_tool_calls)
    s["search_performed"] = used_web_search
    s["search_query"] = query
    return s


# ══════════════════════════════════════════════════════════════════════════════
#  Synthesis
# ══════════════════════════════════════════════════════════════════════════════

_SYNTH_SYS = """You are a synthesis model. Merge expert drafts into one coherent, comprehensive response.
Do NOT mention model names, draft numbers, or that multiple sources were consulted.
Prefer correctness and clarity. Resolve any contradictions by favoring the more detailed answer.
Preserve concrete identifications from the drafts — if any draft names a specific person,
place, object, technique, or entity, carry that identification into the final answer rather
than generalizing it away. Only drop an identification if drafts directly contradict it.
If drafts reference tool results or web search data, integrate that information naturally.
If drafts address different sub-tasks of a complex question, combine them into a unified answer.
Start with a direct answer first, then supporting detail.
Do NOT wrap your entire response in a code block or code fence. Use code fences only for actual code snippets. Output clean markdown directly."""

_SYNTH_SYS_REVIEW = """You are a synthesis model merging code review drafts into one clear report.

Required structure (use exactly these sections in order):
1) `## Findings (Critical/High First)`
2) `## Open Questions / Unverified Risks`
3) `## Low-Priority Suggestions`
4) `## Recommended Next Step`

Ranking rules (strict):
- Behavior mismatches and correctness bugs come first.
- Logic errors, race conditions, data-loss risks, and missing error handling come next.
- Design/maintainability concerns follow.
- Theoretical hardening and style suggestions belong in low-priority.

Evidence policy (strict):
- Include only findings that are grounded in concrete evidence from the drafts.
- Treat the latest user-provided code/message as the primary source of truth.
- Ignore stale snippets from earlier conversation turns unless the latest user message explicitly says to review those older snippets.
- Do NOT invent imports, files, line references, runtime traces, or vulnerabilities.
- For each critical/high finding, include exactly these fields:
  `Location:`
  `Evidence:` (an exact code snippet copied from the original code, wrapped in backticks)
  `Failure mode:`
  `Fix:`
- If you cannot provide an exact `Evidence:` snippet from the original code, drop that finding.
- Never mark an item Critical without deterministic impact (crash/data loss/corruption or concrete exploit path).
- If evidence is insufficient for critical/high findings, state exactly: `No confirmed critical/high findings.`
- Put uncertain ideas into `Open Questions / Unverified Risks`, not into critical/high findings.

Output quality rules:
- Keep the report concise and high-signal (max 5 critical/high items).
- Do NOT include broad refactor plans unless tied to a concrete observed problem.
- Do NOT mention model names, draft numbers, or that multiple sources were consulted.
- If drafts contradict on severity, favor the one with a concrete failure scenario over generic best practice advice.

Do NOT wrap your entire response in a code block or code fence. Use code fences only for actual code snippets. Output clean markdown directly."""


_SYNTH_TOKEN_RE = re.compile(r"[a-z0-9]{4,}")
_SYNTH_STOPWORDS = {
    "about", "above", "after", "again", "against", "also", "among", "because",
    "before", "being", "between", "could", "first", "from", "have", "into",
    "just", "like", "more", "most", "only", "other", "over", "same", "some",
    "than", "that", "their", "there", "these", "they", "this", "those", "very",
    "what", "when", "where", "which", "while", "with", "would", "your",
}
_SYNTH_UNCERTAINTY_MARKERS = (
    "not sure",
    "uncertain",
    "i don't know",
    "cannot determine",
    "can't determine",
    "insufficient information",
    "unable to verify",
)
_REVIEW_SPECULATIVE_MARKERS = (
    "might",
    "may",
    "could",
    "likely",
    "possibly",
    "potential",
    "appears to",
    "seems",
    "probably",
)
_REVIEW_SECTION_TITLE_RE = re.compile(
    r"(?im)^(?:#+\s*)?(Findings \(Critical/High First\)|"
    r"Open Questions / Unverified Risks|"
    r"Low-Priority Suggestions|"
    r"Recommended Next Step)\s*$"
)
_REVIEW_FINDING_START_RE = re.compile(r"(?m)^\s*\d+\.\s+")
_REVIEW_CODE_BLOCK_RE = re.compile(r"```(?:[\w.+-]+)?\n(.*?)```", re.S)


def _is_valid_worker_output(output: dict[str, Any]) -> bool:
    content = str(output.get("content", "") or "")
    return bool(content.strip()) and not content.startswith(_WORKER_ERROR_PREFIX)


_VISION_ONLY_PATTERNS = ("llava", "-vl", ":vl", "vl:", "vision")


def _is_vision_only_model(name: str) -> bool:
    """Vision-specialized models that shouldn't synthesize multi-draft text."""
    n = name.lower()
    return any(p in n for p in _VISION_ONLY_PATTERNS)


def _local_worker_models_for_synth(s: dict[str, Any]) -> list[str]:
    outputs = s.get("worker_outputs", [])
    available = {
        str(o.get("model", "")).strip()
        for o in outputs
        if _is_valid_worker_output(o)
        and not is_cloud_model(str(o.get("model", "")))
        and not _is_vision_only_model(str(o.get("model", "")))
    }
    if not available:
        return []

    ordered: list[str] = []
    # Prefer the most recently produced local draft first (more likely warm).
    for output in reversed(outputs):
        n = str(output.get("model", "")).strip()
        if n and n in available and n not in ordered:
            ordered.append(n)

    for name in s.get("deep_workers", []):
        n = str(name).strip()
        if n and n in available and n not in ordered:
            ordered.append(n)
    return ordered


def _draft_keywords(text: str) -> set[str]:
    return {
        token
        for token in _SYNTH_TOKEN_RE.findall(text.lower())
        if token not in _SYNTH_STOPWORDS
    }


def _has_draft_conflict(
    worker_outputs: list[dict[str, Any]],
    sub_tasks: list[str] | None,
) -> bool:
    # When planned sub-tasks exist, draft diversity is expected, not conflict.
    if sub_tasks and len(sub_tasks) >= 2:
        return False

    keyword_sets: list[set[str]] = []
    for output in worker_outputs:
        text = str(output.get("content", "") or "")
        if len(text) < 120:
            continue
        kws = _draft_keywords(text)
        if len(kws) >= 6:
            keyword_sets.append(kws)

    if len(keyword_sets) < 2:
        return False

    min_overlap = 1.0
    compared = 0
    for i in range(len(keyword_sets)):
        for j in range(i + 1, len(keyword_sets)):
            union = keyword_sets[i] | keyword_sets[j]
            if not union:
                continue
            compared += 1
            overlap = len(keyword_sets[i] & keyword_sets[j]) / len(union)
            if overlap < min_overlap:
                min_overlap = overlap

    return compared > 0 and min_overlap < 0.03


def _has_uncertain_draft(worker_outputs: list[dict[str, Any]]) -> bool:
    for output in worker_outputs:
        text = str(output.get("content", "") or "").lower()
        if any(marker in text for marker in _SYNTH_UNCERTAINTY_MARKERS):
            return True
    return False


def _review_section_bounds(text: str, title: str) -> tuple[int, int, int] | None:
    matches = list(_REVIEW_SECTION_TITLE_RE.finditer(text))
    if not matches:
        return None
    for i, m in enumerate(matches):
        if m.group(1).strip().lower() != title.strip().lower():
            continue
        section_start = m.start()
        body_start = m.end()
        section_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        return section_start, body_start, section_end
    return None


def _split_numbered_findings(body: str) -> list[str]:
    starts = [m.start() for m in _REVIEW_FINDING_START_RE.finditer(body)]
    if not starts:
        stripped = body.strip()
        return [stripped] if stripped else []
    blocks: list[str] = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(body)
        chunk = body[start:end].strip()
        if chunk:
            blocks.append(chunk)
    return blocks


def _finding_has_supported_evidence(block: str, source_text: str) -> bool:
    lowered = block.lower()
    if any(marker in lowered for marker in _REVIEW_SPECULATIVE_MARKERS):
        return False
    required_fields = ("location:", "evidence:", "failure mode:", "fix:")
    if not all(field in lowered for field in required_fields):
        return False

    evidence_line = re.search(r"(?im)^\s*evidence:\s*(.+)$", block)
    if not evidence_line:
        return False
    evidence_raw = evidence_line.group(1).strip()
    snippets = re.findall(r"`([^`]{3,200})`", evidence_raw)
    if not snippets:
        snippets = [evidence_raw.strip("`\"' ")]

    source_lower = source_text.lower()
    for snippet in snippets:
        candidate = snippet.strip()
        if len(candidate) < 12:
            continue
        if not (re.search(r"[()\[\]{}=.:_]", candidate) or "\n" in candidate):
            continue
        if candidate.lower() in source_lower:
            return True
    return False


def _enforce_review_evidence(result_text: str, *, source_text: str) -> str:
    bounds = _review_section_bounds(result_text, "Findings (Critical/High First)")
    if not bounds:
        return result_text

    section_start, body_start, section_end = bounds
    findings_body = result_text[body_start:section_end]
    if "no confirmed critical/high findings." in findings_body.lower():
        return result_text

    blocks = _split_numbered_findings(findings_body)
    if not blocks:
        replacement = "\nNo confirmed critical/high findings.\n"
        return result_text[:body_start] + replacement + result_text[section_end:]

    kept: list[str] = []
    removed = 0
    for block in blocks:
        if _finding_has_supported_evidence(block, source_text):
            kept.append(block)
        else:
            removed += 1

    if not kept:
        replacement = "\nNo confirmed critical/high findings.\n"
        if removed:
            logger.info(
                "Review synthesis filter removed %d unsupported finding(s); no findings left",
                removed,
            )
        return result_text[:body_start] + replacement + result_text[section_end:]

    normalized: list[str] = []
    for i, block in enumerate(kept, 1):
        updated = re.sub(r"(?m)^\s*\d+\.\s+", f"{i}. ", block, count=1)
        if not _REVIEW_FINDING_START_RE.search(updated[:20]):
            updated = f"{i}. {updated.lstrip()}"
        normalized.append(updated.strip())

    replacement = "\n" + "\n\n".join(normalized) + "\n"
    if removed:
        logger.info("Review synthesis filter removed %d unsupported finding(s)", removed)
    return result_text[:body_start] + replacement + result_text[section_end:]


def _review_source_text(s: dict[str, Any]) -> str:
    """Return the strict evidence source for code-review synthesis.

    We intentionally prefer only the latest user-provided message/code blocks so
    stale snippets from older turns do not validate fresh findings.
    """
    msgs = s.get("original_messages", s.get("messages", []))
    if not isinstance(msgs, list) or not msgs:
        return ""

    latest_user = get_last_user_text(msgs).strip()
    if not latest_user:
        return ""

    code_blocks = [
        block.strip()
        for block in _REVIEW_CODE_BLOCK_RE.findall(latest_user)
        if block.strip()
    ]
    if code_blocks:
        return "\n\n".join(code_blocks)
    return latest_user


def _synthesis_escalation_reason(s: dict[str, Any]) -> str:
    if s.get("force_strong_synth"):
        return "forced_strong_synth"

    if s.get("is_code_review"):
        return "code_review"

    confidence = float(s.get("confidence", 0.0) or 0.0)
    if confidence < LOW_CONFIDENCE_THRESHOLD:
        return f"low_confidence:{confidence:.2f}"

    worker_error_count = int(s.get("worker_error_count", 0) or 0)
    if worker_error_count > 0:
        return f"worker_errors:{worker_error_count}"

    worker_outputs = [
        o for o in s.get("worker_outputs", [])
        if _is_valid_worker_output(o)
    ]
    deep_workers = [str(w).strip() for w in s.get("deep_workers", []) if str(w).strip()]
    deep_worker_count = len(deep_workers)
    all_workers_local = bool(deep_workers) and all(
        not is_cloud_model(worker) for worker in deep_workers
    )
    if deep_worker_count > 1 and len(worker_outputs) <= 1:
        return "single_valid_draft"

    # For all-local panels, prefer warm-worker synthesis unless a hard signal
    # (low confidence / worker failures) requires strong configured synth first.
    if not all_workers_local and _has_uncertain_draft(worker_outputs):
        return "draft_uncertainty"

    if not all_workers_local and _has_draft_conflict(worker_outputs, s.get("sub_tasks")):
        return "draft_conflict"

    return ""


def resolve_synthesis_candidates(s: dict[str, Any]) -> list[str]:
    primary = str(s.get("synthesizer", "")).strip()
    fallback = str(s.get("fallback_synthesizer", "")).strip()
    deep_workers = [str(w).strip() for w in s.get("deep_workers", []) if str(w).strip()]
    warm_local_workers = _local_worker_models_for_synth(s)
    all_workers_local = bool(deep_workers) and all(not is_cloud_model(w) for w in deep_workers)
    primary_is_local = bool(primary) and not is_cloud_model(primary)
    primary_is_warm = primary in warm_local_workers
    mixed_workers = bool(deep_workers) and not all_workers_local
    mixed_warm_allowed = (
        mixed_workers
        and warm_local_workers
        and primary_is_local
        and not primary_is_warm
    )
    escalation_reason = _synthesis_escalation_reason(s)

    if warm_local_workers and (all_workers_local or mixed_warm_allowed) and not escalation_reason:
        ordered = [*warm_local_workers, primary, fallback]
        strategy = (
            "warm_worker_first"
            if all_workers_local
            else "warm_worker_first_mixed_avoid_new_local_load"
        )
    elif escalation_reason:
        ordered = [primary, *warm_local_workers, fallback]
        strategy = "configured_first_escalated"
    else:
        ordered = [primary, fallback]
        strategy = "configured_first"

    deduped: list[str] = []
    for name in ordered:
        model_name = str(name or "").strip()
        if model_name and model_name not in deduped:
            deduped.append(model_name)

    if not deduped:
        raise RuntimeError("No synthesis candidates configured")

    # Deprioritize models currently in cooldown rather than drop them outright
    # — if every candidate is unhealthy we still need something to try.
    healthy = [m for m in deduped if is_model_healthy(m)]
    cooled = [m for m in deduped if not is_model_healthy(m)]
    candidates = healthy + cooled if healthy else deduped
    demoted = [m for m in cooled if m in candidates[len(healthy):]]

    s["synthesis_candidates"] = candidates
    s["synthesis_strategy"] = strategy
    s["synthesis_escalation_reason"] = escalation_reason
    s["synthesizer"] = candidates[0]
    logger.info(
        "Synthesis routing: strategy=%s reason=%s candidates=%s demoted_unhealthy=%s",
        strategy,
        escalation_reason or "none",
        candidates,
        demoted or "none",
    )
    return candidates


def build_synth_msgs(s):
    outputs = s["worker_outputs"]
    is_review = s.get("is_code_review", False)
    sys_prompt = _SYNTH_SYS_REVIEW if is_review else _SYNTH_SYS

    secs = []
    for i, o in enumerate(outputs, 1):
        label = o.get("label", "")
        secs.append(f"── Draft {i}{label} ──\n{o['content']}")
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
    s["synthesis_candidates"] = resolve_synthesis_candidates(s)
    s["synthesis_messages"] = build_synth_msgs(s)
    return s


async def node_synthesize(s):
    if "synthesis_messages" not in s:
        s["synthesis_messages"] = build_synth_msgs(s)
    synths = s.get("synthesis_candidates") or resolve_synthesis_candidates(s)
    for i, sy in enumerate(synths):
        try:
            r = await run_model_once(
                sy,
                s["synthesis_messages"],
                **model_call_kwargs(s, temperature=min(s["temperature"], 0.3)),
            )
            if s.get("is_code_review"):
                source = _review_source_text(s)
                if not source:
                    source = flatten_messages(s.get("original_messages", s["messages"]))
                r = _enforce_review_evidence(r, source_text=source)
            s["result_text"] = r
            s["selected_model"] = sy
            s["prompt_tokens"] = estimate_tokens(flatten_messages(s["messages"]))
            s["completion_tokens"] = estimate_tokens(r)
            note_model_success(sy)
            return s
        except Exception as e:
            note_model_failure(sy)
            if i == len(synths) - 1:
                raise RuntimeError(f"Synthesizers failed: {e}")
    raise RuntimeError("No synthesizers")


async def node_reflect_deep(s):
    """Reflection gate for deep panel output."""
    reflection_enabled = s.get("reflection_enabled_override")
    if reflection_enabled is None:
        reflection_enabled = REFLECTION_ENABLED
    if not reflection_enabled or not s.get("result_text"):
        return s

    reflection = await reflect_on_response(
        s.get("original_messages", s["messages"]),
        s["result_text"],
        is_code_review=bool(s.get("is_code_review", False)),
    )
    s["reflection_result"] = reflection

    retries_limit = s.get("reflection_max_retries_override")
    if retries_limit is None:
        retries_limit = REFLECTION_MAX_RETRIES
    if reflection["quality"] == "poor" and s.get("reflection_retries", 0) < retries_limit:
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
        previous = str(s.get("selected_model", "")).strip()
        s["force_strong_synth"] = True
        candidates = resolve_synthesis_candidates(s)
        if previous and previous in candidates and len(candidates) > 1:
            reordered = [m for m in candidates if m != previous]
            reordered.append(previous)
            s["synthesis_candidates"] = reordered
            s["synthesizer"] = reordered[0]
            logger.info(
                "Deep reflection retry: deprioritized previous synthesizer %s",
                previous,
            )
        return await node_synthesize(s)

    return s
