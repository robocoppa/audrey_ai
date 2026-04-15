"""
Audrey — request classification and routing.

Keyword pre-filters, router model classification, complexity gate,
and fast-model selection from the model registry.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from config import (
    COMPLEXITY_FORCE_DEEP,
    COMPLEXITY_TOKEN_THRESHOLD,
    MODEL_REGISTRY,
    ROUTER_MODEL,
    is_cloud_model,
)
from health import is_model_healthy
from helpers import (
    estimate_tokens,
    flatten_messages,
    get_last_user_text,
    has_vision_content,
    timeout_for_model,
)
from ollama import run_model_once, ollama_chat_once

import state

logger = logging.getLogger("audrey.classifier")


# ══════════════════════════════════════════════════════════════════════════════
#  Keyword pre-filters
# ══════════════════════════════════════════════════════════════════════════════

# Strong code signals — high confidence, bypass router entirely
_CS = [
    re.compile(r"```"),
    re.compile(r"\b(traceback|stacktrace|segfault|stderr)\b", re.I),
    re.compile(r"(?:^|\n)\s*(?:import |from \S+ import |#include |using namespace )", re.M),
    re.compile(
        r"\b(TypeError|ValueError|KeyError|IndexError|AttributeError|NameError|"
        r"SyntaxError|RuntimeError|NullPointerException|SegmentationFault)\b"
    ),
    re.compile(
        r"(?:^|\n)\s*(?:fn |func |def |class |struct |enum |impl |pub fn |async fn )", re.M
    ),
]

# Weak code signals — need 2+ matches for confidence
_CW = [
    re.compile(r"(?:^|\n)\s*(?:def |class |from \S+ import )", re.M),
    re.compile(
        r"(fix|debug|refactor|implement|write|build|create|optimize)\s.{0,20}"
        r"(code|function|script|bug|error|program|app|module|class|method)",
        re.I,
    ),
    re.compile(
        r"\b(syntax error|compile|runtime error|type error|null pointer|segfault)\b", re.I
    ),
    re.compile(
        r"\b(API|endpoint|REST|GraphQL|SQL|database|query|schema)\b.{0,20}"
        r"(code|implement|write|create|build|design|optimize)",
        re.I,
    ),
    re.compile(
        r"\b(python|javascript|typescript|rust|go|java|c\+\+|ruby|swift|kotlin|"
        r"bash|shell|sql|html|css)\b.{0,30}"
        r"(code|script|function|program|implement|write|fix|error|bug)",
        re.I,
    ),
    re.compile(
        r"\b(docker|kubernetes|nginx|terraform|ansible|git|webpack|npm|pip|cargo)\b"
        r".{0,20}(config|setup|error|fix|issue|problem)",
        re.I,
    ),
    re.compile(r"\b(regex|regexp|regular expression|pattern match)\b", re.I),
    re.compile(
        r"\b(algorithm|data structure|linked list|binary tree|hash map|sorting|recursion)\b"
        r".{0,20}(implement|write|code|build)",
        re.I,
    ),
]

# Reasoning patterns
_RP = [
    re.compile(
        r"(explain|prove|derive|compare|analyze|evaluate|assess|critique|argue)\s"
        r".{0,30}(why|how|whether|if|the|between|difference)",
        re.I,
    ),
    re.compile(r"step[- ]by[- ]step", re.I),
    re.compile(
        r"\b(proof|theorem|hypothesis|tradeoff|pros and cons|advantages|disadvantages)\b",
        re.I,
    ),
    re.compile(
        r"\b(compare|contrast|versus|vs\.?|difference between)\b.{0,40}\b(and|or|vs)\b",
        re.I,
    ),
    re.compile(
        r"\b(what are the|list the|outline the)\b.{0,20}"
        r"\b(implications|consequences|factors|considerations|tradeoffs|risks|benefits)\b",
        re.I,
    ),
    re.compile(r"\b(why does|why is|why are|why do|how does|how do|how is|how are)\b.{10,}", re.I),
    re.compile(
        r"\b(analyze|evaluate|assess|review|examine|investigate)\b.{0,30}"
        r"\b(impact|effect|result|outcome|performance|approach|strategy|method)\b",
        re.I,
    ),
    re.compile(
        r"\b(should I|which is better|what's the best|recommend)\b.{0,30}"
        r"\b(approach|method|strategy|framework|tool|language|option)\b",
        re.I,
    ),
]

# General / simple patterns — high confidence for fast-path handling
_GP = [
    re.compile(
        r"^(hi|hello|hey|good morning|good afternoon|good evening|howdy|sup|yo)\b[!?.]*$", re.I
    ),
    re.compile(
        r"^(thanks|thank you|thx|ty|cheers|got it|ok|okay|cool|great|perfect|nice)\b[!?.]*$", re.I
    ),
    re.compile(r"^(what is|what's|define|tell me about|who is|who was)\s+\w+", re.I),
    re.compile(r"^(translate|summarize|summarise|paraphrase|rewrite|rephrase)\b", re.I),
]

# Review / analysis override — reclassify code → reasoning when the user's
# actual intent is asking for review, suggestions, or analysis of code.
# EXPANDED: added "assess", "assessment", lone "look at/over", "check",
# "what can be improved", and phrasing without leading verbs.
_REVIEW_OVERRIDE = re.compile(
    r"\b(review|suggest|suggestion|feedback|improve|improvement|critique|analyze|"
    r"analyse|opinion|thoughts|what do you think|look over|check over|go over|"
    r"assess|assessment|evaluate|evaluation|audit|refactor suggestions|best practices|"
    r"code review|code quality|clean up|improvements|recommendations|optimize this|"
    r"any issues|anything wrong|what can be better|what can be improved|"
    r"how can .{0,20} improv|give me .{0,20} feedback|tell me .{0,20} about this|"
    r"check this|look at this|go through this|walk through this|"
    r"what do you see|what stands out|spot any|find any|catch any|"
    r"how does this look|how is this|is this good|is this correct|is this right|"
    r"any problems|any bugs|any improvements|room for improvement)\b",
    re.I,
)


# ══════════════════════════════════════════════════════════════════════════════
#  Keyword pre-filter
# ══════════════════════════════════════════════════════════════════════════════

def keyword_prefilter(text: str, *, user_text: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Expanded keyword pre-filter — catches patterns to reduce router calls.

    Args:
        text: The text to classify (last user message preferred).
        user_text: If provided, used for the review/analysis override check.
    """
    override_text = user_text or text
    is_review_request = bool(_REVIEW_OVERRIDE.search(override_text))

    # Strong code signals
    for p in _CS:
        if p.search(text):
            if is_review_request:
                return {
                    "task_type": "reasoning",
                    "confidence": 0.85,
                    "needs_vision": False,
                    "is_code_review": True,
                    "route_reason": "keyword: code+review → reasoning",
                }
            return {
                "task_type": "code",
                "confidence": 0.92,
                "needs_vision": False,
                "is_code_review": False,
                "route_reason": "keyword: strong code signal",
            }
    # Weak code signals (need 2+)
    w = [p for p in _CW if p.search(text)]
    if len(w) >= 2:
        if is_review_request:
            return {
                "task_type": "reasoning",
                "confidence": 0.82,
                "needs_vision": False,
                "is_code_review": True,
                "route_reason": f"keyword: weak code (x{len(w)})+review → reasoning",
            }
        return {
            "task_type": "code",
            "confidence": 0.85,
            "needs_vision": False,
            "is_code_review": False,
            "route_reason": f"keyword: weak code (x{len(w)})",
        }
    # Reasoning patterns
    for p in _RP:
        if p.search(text):
            return {
                "task_type": "reasoning",
                "confidence": 0.80,
                "needs_vision": False,
                "is_code_review": False,
                "route_reason": "keyword: reasoning pattern",
            }
    # General/simple patterns
    for p in _GP:
        if p.search(text):
            return {
                "task_type": "general",
                "confidence": 0.95,
                "needs_vision": False,
                "is_code_review": False,
                "route_reason": "keyword: general/simple",
            }
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  Router model classification
# ══════════════════════════════════════════════════════════════════════════════

_ROUTER_SYSTEM = """You are a request classifier. Your ONLY job is to output a JSON object.
Do NOT explain, do NOT add markdown fences, do NOT think out loud.

Classify the user request into exactly one of these types:
- "code" — writing, debugging, reviewing, or explaining code; technical implementation
- "reasoning" — analysis, comparison, math, logic, step-by-step problem solving, strategy
- "vl" — the request includes or asks about an image
- "general" — conversation, factual questions, creative writing, translation, summaries

Output ONLY this JSON (nothing else before or after):
{"task_type": "...", "confidence": 0.XX, "needs_vision": false, "route_reason": "one short phrase"}

Confidence guide:
- 0.95 = obviously this type, no ambiguity
- 0.85 = clearly this type with minor ambiguity
- 0.70 = likely this type but could be another
- 0.50 = genuinely uncertain

Examples:

User: "Write a Python function to merge two sorted lists"
{"task_type": "code", "confidence": 0.95, "needs_vision": false, "route_reason": "Python implementation request"}

User: "Compare the pros and cons of microservices vs monolith architecture"
{"task_type": "reasoning", "confidence": 0.90, "needs_vision": false, "route_reason": "Architecture comparison/analysis"}

User: "What's in this image?"
{"task_type": "vl", "confidence": 0.95, "needs_vision": true, "route_reason": "Image analysis request"}

User: "Tell me about the history of jazz music"
{"task_type": "general", "confidence": 0.92, "needs_vision": false, "route_reason": "Factual/historical question"}

User: "Debug this error: TypeError: cannot unpack non-sequence NoneType"
{"task_type": "code", "confidence": 0.95, "needs_vision": false, "route_reason": "Debug Python TypeError"}

User: "Should I use React or Vue for my next project? What are the tradeoffs?"
{"task_type": "reasoning", "confidence": 0.85, "needs_vision": false, "route_reason": "Framework comparison with tradeoffs"}

User: "Write me a poem about autumn"
{"task_type": "general", "confidence": 0.92, "needs_vision": false, "route_reason": "Creative writing request"}

User: "Explain why quicksort has O(n log n) average case complexity"
{"task_type": "reasoning", "confidence": 0.88, "needs_vision": false, "route_reason": "Algorithm complexity analysis"}

Now classify the following request. Output ONLY the JSON object:"""


def _extract_json(raw: str) -> Optional[dict]:
    """Robustly extract a JSON object from model output."""
    if not raw or not raw.strip():
        return None

    text = raw.strip()

    # Strip <think>...</think> blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Strip markdown fences
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find the first { ... } block
    brace_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    # Find nested braces
    depth = 0
    start = None
    for i, c in enumerate(text):
        if c == "{":
            if depth == 0:
                start = i
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    start = None

    return None


# ── Keyword fallback (when router model fails) ──────────────────────────────

def _keyword_fallback_classify(text: str) -> Dict[str, Any]:
    """Last-resort classification using simple keyword counts."""
    text_lower = text.lower()

    code_score = 0
    reasoning_score = 0

    code_words = [
        "code", "function", "bug", "error", "debug", "implement", "script",
        "python", "javascript", "rust", "java", "api", "endpoint", "database",
        "sql", "html", "css", "docker", "git", "compile", "syntax", "variable",
        "class", "method", "library", "package", "framework", "deploy", "server",
        "regex", "algorithm", "array", "string", "loop", "recursion",
    ]
    for w in code_words:
        if w in text_lower:
            code_score += 1

    if "```" in text or re.search(r"(?:import |from \S+ import |#include )", text):
        code_score += 5

    reasoning_words = [
        "explain", "why", "compare", "analyze", "evaluate", "tradeoff",
        "pros and cons", "difference", "versus", "better", "worse",
        "advantage", "disadvantage", "implication", "consequence",
        "strategy", "approach", "reasoning", "logic", "proof", "theorem",
        "step by step", "argue", "critique", "assess",
    ]
    for w in reasoning_words:
        if w in text_lower:
            reasoning_score += 1

    if code_score >= 3 or (code_score >= 2 and reasoning_score == 0):
        return {
            "task_type": "code",
            "confidence": 0.70,
            "needs_vision": False,
            "route_reason": f"Keyword fallback (code score={code_score})",
        }
    if reasoning_score >= 2 or (
        reasoning_score >= 1 and code_score == 0 and len(text) > 100
    ):
        return {
            "task_type": "reasoning",
            "confidence": 0.65,
            "needs_vision": False,
            "route_reason": f"Keyword fallback (reasoning score={reasoning_score})",
        }

    return {
        "task_type": "general",
        "confidence": 0.60,
        "needs_vision": False,
        "route_reason": f"Keyword fallback (code={code_score}, reasoning={reasoning_score})",
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Main classifier
# ══════════════════════════════════════════════════════════════════════════════

async def classify_request(msgs: List[Dict[str, Any]]) -> Dict[str, Any]:
    if has_vision_content(msgs):
        return {
            "task_type": "vl",
            "confidence": 0.99,
            "needs_vision": True,
            "is_code_review": False,
            "route_reason": "Image detected.",
        }

    user_text = get_last_user_text(msgs)
    flat = flatten_messages(msgs)
    kw = keyword_prefilter(user_text or flat, user_text=user_text)
    if kw:
        return kw

    # ── Router model classification ──
    rp = [
        {"role": "system", "content": _ROUTER_SYSTEM},
        {"role": "user", "content": flat},
    ]

    # Check review intent for router-classified results too
    is_review = bool(_REVIEW_OVERRIDE.search(user_text or flat))

    for attempt in range(2):
        try:
            raw = (
                await ollama_chat_once(
                    ROUTER_MODEL, rp,
                    temperature=0.0, max_tokens=120, top_p=0.9, stop=None,
                    request_timeout=timeout_for_model(ROUTER_MODEL, is_router=True),
                )
            )["message"]["content"]

            p = _extract_json(raw)
            if p:
                tt = p.get("task_type", "general")
                if tt not in {"code", "reasoning", "vl", "general"}:
                    tt = "general"
                # If the router classifies as code but user wants review,
                # override to reasoning + flag as code review
                code_review = False
                if is_review and tt == "code":
                    tt = "reasoning"
                    code_review = True
                elif is_review and tt == "reasoning":
                    # Router already said reasoning; still flag as code review
                    # if code signals are present in the input
                    if any(p.search(user_text or flat) for p in _CS):
                        code_review = True
                return {
                    "task_type": tt,
                    "confidence": max(0.0, min(float(p.get("confidence", 0.5)), 1.0)),
                    "needs_vision": bool(p.get("needs_vision", False)),
                    "is_code_review": code_review,
                    "route_reason": str(p.get("route_reason", "router")),
                }

            logger.warning(
                "Router parse failure (attempt %d). Raw: %s", attempt + 1, raw[:300]
            )
        except Exception as e:
            logger.warning("Router call failed (attempt %d): %s", attempt + 1, e)

    # Both attempts failed — keyword fallback
    fallback = _keyword_fallback_classify(user_text or flat)
    fallback.setdefault("is_code_review", False)
    # Check if this is a code review even through the fallback path
    if is_review and fallback["task_type"] in ("code", "reasoning"):
        if any(p.search(user_text or flat) for p in _CS):
            fallback["is_code_review"] = True
            if fallback["task_type"] == "code":
                fallback["task_type"] = "reasoning"
    logger.info(
        "Router exhausted — keyword fallback: type=%s conf=%.2f reason=%s",
        fallback["task_type"],
        fallback["confidence"],
        fallback["route_reason"],
    )
    return fallback


# ══════════════════════════════════════════════════════════════════════════════
#  Complexity gate — force deep panel for large / complex inputs
# ══════════════════════════════════════════════════════════════════════════════

def should_force_deep(
    messages: List[Dict[str, Any]],
    confidence: float,
    task_type: str,
) -> Optional[str]:
    """Return a reason string if the input is too complex for fast path, else None.

    Checks:
      1. Total token estimate exceeds COMPLEXITY_TOKEN_THRESHOLD
      2. (Future: multi-file input, conversation depth, etc.)
    """
    if not COMPLEXITY_FORCE_DEEP:
        return None

    input_tokens = estimate_tokens(flatten_messages(messages))
    if input_tokens > COMPLEXITY_TOKEN_THRESHOLD:
        return f"complex_input({input_tokens}tok)"

    return None


# ══════════════════════════════════════════════════════════════════════════════
#  Fast-model selection from model registry
# ══════════════════════════════════════════════════════════════════════════════

def select_fast_model(task_type: str) -> Optional[str]:
    """Pick the highest-priority healthy model for a task type."""
    candidates = MODEL_REGISTRY.get(task_type, MODEL_REGISTRY.get("general", []))
    for entry in sorted(candidates, key=lambda e: e.get("priority", 0), reverse=True):
        name = entry["name"]
        if is_model_healthy(name) and (
            name in state.available_models or is_cloud_model(name)
        ):
            return name
    return None
