import asyncio
import hashlib
import json
import logging
import os
import re
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, List, Literal, Optional, TypedDict

import aiohttp
import yaml
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from langgraph.graph import END, StateGraph
from pydantic import BaseModel

from tool_registry import ToolRegistry

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("orchestrator")

# ── Config ───────────────────────────────────────────────────────────────────
CONFIG_PATH = os.getenv("CONFIG_PATH", "/app/config.yaml")


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


_config = load_config()
MODEL_REGISTRY = _config["model_registry"]
DEEP_PANEL_CLOUD = _config["deep_panel_cloud"]
DEEP_PANEL_LOCAL = _config["deep_panel_local"]
DEEP_PANEL_MIXED = _config["deep_panel"]
TIMEOUTS = _config.get("timeouts", {})
CACHE_CONFIG = _config.get("cache", {})
FAST_PATH_CONFIG = _config.get("fast_path", {})
TOOL_SERVER_URLS: List[str] = _config.get("tool_servers", [])

# ── Env knobs ────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
# ┌─────────────────────────────────────────────────────────────────────────┐
# │ FIX 2: Router model changed from qwen2.5:1.5b → qwen3:4b              │
# └─────────────────────────────────────────────────────────────────────────┘
ROUTER_MODEL = os.getenv("ROUTER_MODEL", "qwen3:4b")
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.2"))
LOW_CONFIDENCE_THRESHOLD = float(os.getenv("LOW_CONFIDENCE_THRESHOLD", "0.60"))
MAX_DEEP_WORKERS = int(os.getenv("MAX_DEEP_WORKERS", "2"))
# ┌─────────────────────────────────────────────────────────────────────────┐
# │ FIX 1: Cloud worker cap — separate from local MAX_DEEP_WORKERS         │
# └─────────────────────────────────────────────────────────────────────────┘
MAX_DEEP_WORKERS_CLOUD = int(os.getenv("MAX_DEEP_WORKERS_CLOUD", "3"))
EMIT_ROUTING_BANNER = os.getenv("EMIT_ROUTING_BANNER", "true").lower() == "true"
GPU_CONCURRENCY = int(os.getenv("GPU_CONCURRENCY", "1"))
EMIT_STATUS_UPDATES = os.getenv("EMIT_STATUS_UPDATES", "true").lower() == "true"
DEEP_WORKER_TIMEOUT = int(
    os.getenv("DEEP_WORKER_TIMEOUT", str(TIMEOUTS.get("deep_worker", 240)))
)
API_KEY = os.getenv("API_KEY", "")
TOOLS_ENABLED = os.getenv("TOOLS_ENABLED", "true").lower() == "true"
SEARCH_BACKEND = os.getenv("SEARCH_BACKEND", "searxng")
SEARXNG_URL = os.getenv("SEARXNG_URL", "http://searxng:8081")
SEARXNG_MAX_RESULTS = int(os.getenv("SEARXNG_MAX_RESULTS", "5"))
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY", "")
BRAVE_MAX_RESULTS = int(os.getenv("BRAVE_MAX_RESULTS", "5"))

# Fast-path config
FAST_PATH_ENABLED = FAST_PATH_CONFIG.get("enabled", True)
FAST_PATH_CONFIDENCE = FAST_PATH_CONFIG.get("confidence_threshold", 0.88)
FAST_PATH_TOOL_MODELS = set(FAST_PATH_CONFIG.get("tool_capable_models", []))
FAST_PATH_TIMEOUT = int(TIMEOUTS.get("fast_path", 180))

# ── Agentic config ───────────────────────────────────────────────────────────
REACT_MAX_ROUNDS = int(os.getenv("REACT_MAX_ROUNDS", "3"))
REFLECTION_ENABLED = os.getenv("REFLECTION_ENABLED", "true").lower() == "true"
PLANNING_ENABLED = os.getenv("PLANNING_ENABLED", "true").lower() == "true"
PLANNING_MIN_TOKENS = int(os.getenv("PLANNING_MIN_TOKENS", "40"))
ESCALATION_ENABLED = os.getenv("ESCALATION_ENABLED", "true").lower() == "true"
ESCALATION_MIN_LENGTH = int(os.getenv("ESCALATION_MIN_LENGTH", "100"))
ESCALATION_CONFIDENCE_CEILING = float(os.getenv("ESCALATION_CONFIDENCE_CEILING", "0.95"))
REFLECTION_MAX_RETRIES = int(os.getenv("REFLECTION_MAX_RETRIES", "1"))

# ── Shared state ─────────────────────────────────────────────────────────────
_ollama_session: Optional[aiohttp.ClientSession] = None
_ext_session: Optional[aiohttp.ClientSession] = None
_tool_registry: Optional[ToolRegistry] = None
_gpu_semaphore = asyncio.Semaphore(GPU_CONCURRENCY)
_available_models: set = set()  # populated at startup


def _is_cloud_model(n):
    return ":cloud" in n


_ALL_VIRTUAL_MODELS = {"audrey_deep", "audrey_local", "audrey_cloud"}


def _deep_panel_for_model(n):
    if n == "audrey_local":
        return DEEP_PANEL_LOCAL
    if n == "audrey_cloud":
        return DEEP_PANEL_CLOUD
    return DEEP_PANEL_MIXED  # "audrey_deep" uses mixed


# ── Datetime injection ───────────────────────────────────────────────────────
from datetime import datetime, timezone


def _datetime_system_message() -> Dict[str, str]:
    """Build a system message with the current date, time, and day of week."""
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


# ── Cache ────────────────────────────────────────────────────────────────────
CACHE_ENABLED = CACHE_CONFIG.get("enabled", True)
CACHE_MAX = CACHE_CONFIG.get("max_entries", 256)
CACHE_TTL = CACHE_CONFIG.get("ttl_seconds", 600)


class LRUCache:
    def __init__(self, mx, ttl):
        self._s: OrderedDict[str, tuple] = OrderedDict()
        self._mx = mx
        self._ttl = ttl
        self._h = 0
        self._m = 0

    def _k(self, msgs, model, temp):
        return hashlib.sha256(
            json.dumps({"m": msgs, "model": model, "t": temp}, sort_keys=True).encode()
        ).hexdigest()

    def get(self, msgs, model, temp):
        k = self._k(msgs, model, temp)
        e = self._s.get(k)
        if e is None:
            self._m += 1
            return None
        ts, txt = e
        if time.time() - ts > self._ttl:
            del self._s[k]
            self._m += 1
            return None
        self._s.move_to_end(k)
        self._h += 1
        return txt

    def put(self, msgs, model, temp, txt):
        k = self._k(msgs, model, temp)
        self._s[k] = (time.time(), txt)
        self._s.move_to_end(k)
        while len(self._s) > self._mx:
            self._s.popitem(last=False)

    @property
    def stats(self):
        return {"hits": self._h, "misses": self._m, "size": len(self._s)}


_cache = LRUCache(CACHE_MAX, CACHE_TTL)

# ── Model health ─────────────────────────────────────────────────────────────
MODEL_HEALTH: Dict[str, Dict[str, Any]] = {}


def health_record(n):
    if n not in MODEL_HEALTH:
        MODEL_HEALTH[n] = {"failures": 0, "last_failure": None, "cooldown_until": 0}
    return MODEL_HEALTH[n]


def is_model_healthy(n):
    return time.time() >= float(health_record(n).get("cooldown_until", 0) or 0)


def note_model_failure(n):
    r = health_record(n)
    r["failures"] += 1
    r["last_failure"] = time.time()
    if r["failures"] >= 2:
        cd = min(60 * (2 ** (r["failures"] - 2)), 1800)
        r["cooldown_until"] = time.time() + cd
        logger.warning("Model %s cooled %ds (failures=%d)", n, cd, r["failures"])


def note_model_success(n):
    r = health_record(n)
    r["failures"] = 0
    r["cooldown_until"] = 0


# ── Timeout / tokens ─────────────────────────────────────────────────────────
def timeout_for_model(n, *, is_router=False):
    if is_router:
        return TIMEOUTS.get("router", 20)
    if _is_cloud_model(n):
        return TIMEOUTS.get("cloud", 120)
    m = re.search(r"(\d+(\.\d+)?)b", n.lower())
    if m:
        p = float(m.group(1))
        if p <= 3:
            return TIMEOUTS.get("small", 60)
        if p <= 14:
            return TIMEOUTS.get("medium", 180)
    return TIMEOUTS.get("large", 360)


def estimate_tokens(text):
    if not text:
        return 0
    ce = len(text) / 3.5
    we = len(text.split()) * 1.3
    cr = sum(1 for c in text if c in "{}[]();=<>|&!@#$%^*~/\\") / max(len(text), 1)
    return max(1, int(ce * 0.7 + we * 0.3 if cr > 0.05 else ce * 0.3 + we * 0.7))


# ── Message helpers ──────────────────────────────────────────────────────────
def flatten_messages(msgs):
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


def has_vision_content(msgs):
    for m in msgs:
        c = m.get("content")
        if isinstance(c, list):
            for i in c:
                if isinstance(i, dict) and i.get("type") == "image_url":
                    return True
    return False


def get_last_user_text(msgs) -> str:
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


# ══════════════════════════════════════════════════════════════════════════════
#  FIX 3: Expanded keyword pre-filters
#  - More code patterns (languages, frameworks, error types)
#  - More reasoning patterns (analysis, comparison, explanation)
#  - New "general" quick-match for simple conversational queries
# ══════════════════════════════════════════════════════════════════════════════

# Strong code signals — high confidence, bypass router entirely
_CS = [
    re.compile(r"```"),
    re.compile(r"\b(traceback|stacktrace|segfault|stderr)\b", re.I),
    re.compile(r"(?:^|\n)\s*(?:import |from \S+ import |#include |using namespace )", re.M),
    re.compile(r"\b(TypeError|ValueError|KeyError|IndexError|AttributeError|NameError|SyntaxError|RuntimeError|NullPointerException|SegmentationFault)\b"),
    re.compile(r"(?:^|\n)\s*(?:fn |func |def |class |struct |enum |impl |pub fn |async fn )", re.M),
]

# Weak code signals — need 2+ matches for confidence
_CW = [
    re.compile(r"(?:^|\n)\s*(?:def |class |from \S+ import )", re.M),
    re.compile(
        r"(fix|debug|refactor|implement|write|build|create|optimize)\s.{0,20}(code|function|script|bug|error|program|app|module|class|method)",
        re.I,
    ),
    re.compile(r"\b(syntax error|compile|runtime error|type error|null pointer|segfault)\b", re.I),
    re.compile(
        r"\b(API|endpoint|REST|GraphQL|SQL|database|query|schema)\b.{0,20}(code|implement|write|create|build|design|optimize)",
        re.I,
    ),
    re.compile(r"\b(python|javascript|typescript|rust|go|java|c\+\+|ruby|swift|kotlin|bash|shell|sql|html|css)\b.{0,30}(code|script|function|program|implement|write|fix|error|bug)", re.I),
    re.compile(r"\b(docker|kubernetes|nginx|terraform|ansible|git|webpack|npm|pip|cargo)\b.{0,20}(config|setup|error|fix|issue|problem)", re.I),
    re.compile(r"\b(regex|regexp|regular expression|pattern match)\b", re.I),
    re.compile(r"\b(algorithm|data structure|linked list|binary tree|hash map|sorting|recursion)\b.{0,20}(implement|write|code|build)", re.I),
]

# Reasoning patterns
_RP = [
    re.compile(
        r"(explain|prove|derive|compare|analyze|evaluate|assess|critique|argue)\s.{0,30}(why|how|whether|if|the|between|difference)",
        re.I,
    ),
    re.compile(r"step[- ]by[- ]step", re.I),
    re.compile(r"\b(proof|theorem|hypothesis|tradeoff|pros and cons|advantages|disadvantages)\b", re.I),
    re.compile(r"\b(compare|contrast|versus|vs\.?|difference between)\b.{0,40}\b(and|or|vs)\b", re.I),
    re.compile(r"\b(what are the|list the|outline the)\b.{0,20}\b(implications|consequences|factors|considerations|tradeoffs|risks|benefits)\b", re.I),
    re.compile(r"\b(why does|why is|why are|why do|how does|how do|how is|how are)\b.{10,}", re.I),
    re.compile(r"\b(analyze|evaluate|assess|review|examine|investigate)\b.{0,30}\b(impact|effect|result|outcome|performance|approach|strategy|method)\b", re.I),
    re.compile(r"\b(should I|which is better|what's the best|recommend)\b.{0,30}\b(approach|method|strategy|framework|tool|language|option)\b", re.I),
]

# ┌─────────────────────────────────────────────────────────────────────────┐
# │ FIX 3: New — general/simple patterns for fast bypass                   │
# │ These catch simple conversational queries that don't need the router.  │
# └─────────────────────────────────────────────────────────────────────────┘
_GP = [
    re.compile(r"^(hi|hello|hey|good morning|good afternoon|good evening|howdy|sup|yo)\b[!?.]*$", re.I),
    re.compile(r"^(thanks|thank you|thx|ty|cheers|got it|ok|okay|cool|great|perfect|nice)\b[!?.]*$", re.I),
    re.compile(r"^(what is|what's|define|tell me about|who is|who was)\s+\w+", re.I),
    re.compile(r"^(translate|summarize|summarise|paraphrase|rewrite|rephrase)\b", re.I),
]

# ┌─────────────────────────────────────────────────────────────────────────┐
# │ FIX 5: Review/analysis override — reclassify code → reasoning when     │
# │ the user's actual question is asking for review, suggestions, or       │
# │ analysis of code rather than writing/debugging code.                   │
# └─────────────────────────────────────────────────────────────────────────┘
_REVIEW_OVERRIDE = re.compile(
    r"\b(review|suggest|suggestion|feedback|improve|improvement|critique|analyze|"
    r"analyse|opinion|thoughts|what do you think|look over|check over|go over|"
    r"assess|evaluate|audit|refactor suggestions|best practices|code review|"
    r"code quality|clean up|improvements|recommendations|optimize this|"
    r"any issues|anything wrong|what can be better|how can .{0,20} improv|"
    r"give me .{0,20} feedback|tell me .{0,20} about this)\b",
    re.I,
)


def keyword_prefilter(text, *, user_text=None):
    """Expanded keyword pre-filter — catches more patterns to reduce router calls.

    Args:
        text: The text to classify (last user message preferred).
        user_text: If provided, used for the review/analysis override check.
                   When code signals come from pasted code but the user's actual
                   question is asking for review/analysis, reclassify as reasoning.
    """
    # Check for review/analysis intent FIRST using the user's actual question.
    # This catches "here's my code, give me suggestions" before the backtick
    # pattern fires and misroutes it as a code-writing task.
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
                    "route_reason": f"Keyword (code+review override): {p.pattern[:40]} → reasoning",
                }
            return {
                "task_type": "code",
                "confidence": 0.92,
                "needs_vision": False,
                "route_reason": f"Keyword (strong code): {p.pattern[:60]}",
            }
    # Weak code signals (need 2+)
    w = [p for p in _CW if p.search(text)]
    if len(w) >= 2:
        if is_review_request:
            return {
                "task_type": "reasoning",
                "confidence": 0.82,
                "needs_vision": False,
                "route_reason": f"Keyword (weak code x{len(w)}+review override) → reasoning",
            }
        return {
            "task_type": "code",
            "confidence": 0.85,
            "needs_vision": False,
            "route_reason": f"Keyword (weak code x{len(w)})",
        }
    # Reasoning patterns
    for p in _RP:
        if p.search(text):
            return {
                "task_type": "reasoning",
                "confidence": 0.80,
                "needs_vision": False,
                "route_reason": f"Keyword (reasoning): {p.pattern[:60]}",
            }
    # General/simple patterns — high confidence so fast path can handle them
    for p in _GP:
        if p.search(text):
            return {
                "task_type": "general",
                "confidence": 0.95,
                "needs_vision": False,
                "route_reason": f"Keyword (general): {p.pattern[:60]}",
            }
    return None


# ── Web search detection ─────────────────────────────────────────────────────
_SP = [
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
    re.compile(
        r"\b(news|headline|update|announcement|released|launched)\b", re.I
    ),
    re.compile(r"\b(stock|market|crypto|bitcoin)\b.{0,20}\b(price|value|worth)\b", re.I),
    re.compile(r"\b(score|won|lost|beat|defeated|playoff|standings)\b", re.I),
    re.compile(r"\b(election|voted|poll|ballot)\b", re.I),
    re.compile(r"\b20(2[5-9]|[3-9]\d)\b"),
    re.compile(r"\b(search|look up|google|find out)\b", re.I),
]
_FP = [
    re.compile(
        r"\b(how (much|many|old|tall|far|long))\b.{0,30}\b(is|are|was|were)\b", re.I
    ),
    re.compile(r"\b(population|capital|founded|born|died)\b.{0,20}\b(of|in|on)\b", re.I),
]


def needs_web_search(text):
    if SEARCH_BACKEND == "brave" and not BRAVE_API_KEY:
        return False
    for p in _SP:
        if p.search(text):
            return True
    for p in _FP:
        if p.search(text):
            return True
    return False


def extract_search_query(text):
    q = re.sub(
        r"^(hey|hi|hello|please|can you|could you|tell me|search for|look up|find|what is|what's)\s+",
        "",
        text.strip(),
        flags=re.I,
    )
    if len(q) > 80:
        q = q[:80].rsplit(" ", 1)[0]
    return q.strip() or text[:80].strip()


# ── Search backends ──────────────────────────────────────────────────────────
async def searxng_search(query):
    try:
        async with _ext_session.get(
            f"{SEARXNG_URL}/search",
            params={"q": query, "format": "json", "safesearch": "0"},
            timeout=aiohttp.ClientTimeout(total=15),
        ) as r:
            if r.status != 200:
                return []
            data = await r.json()
        return [
            {
                "title": i.get("title", ""),
                "url": i.get("url", ""),
                "snippet": i.get("content", ""),
            }
            for i in data.get("results", [])[:SEARXNG_MAX_RESULTS]
        ]
    except Exception as e:
        logger.warning("SearXNG: %s", e)
        return []


async def brave_search(query):
    if not BRAVE_API_KEY:
        return []
    try:
        async with _ext_session.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers={
                "Accept": "application/json",
                "X-Subscription-Token": BRAVE_API_KEY,
            },
            params={"q": query, "count": BRAVE_MAX_RESULTS},
            timeout=aiohttp.ClientTimeout(total=10),
        ) as r:
            if r.status != 200:
                return []
            data = await r.json()
        return [
            {
                "title": i.get("title", ""),
                "url": i.get("url", ""),
                "snippet": i.get("description", ""),
            }
            for i in data.get("web", {}).get("results", [])[:BRAVE_MAX_RESULTS]
        ]
    except Exception as e:
        logger.warning("Brave: %s", e)
        return []


async def web_search(query):
    if SEARCH_BACKEND == "searxng":
        return await searxng_search(query)
    return await brave_search(query)


def format_search_results(results):
    if not results:
        return ""
    parts = ["Here are recent web search results:\n"]
    for i, r in enumerate(results, 1):
        parts.append(
            f"{i}. {r['title']}\n   {r['snippet']}\n   Source: {r['url']}\n"
        )
    parts.append(
        "\nUse these results for an accurate answer. Cite sources when relevant."
    )
    return "\n".join(parts)


# ── Ollama helpers ───────────────────────────────────────────────────────────
def build_ollama_payload(
    model,
    msgs,
    *,
    temperature,
    max_tokens,
    top_p,
    stop,
    stream,
    frequency_penalty=None,
    presence_penalty=None,
    tools=None,
):
    p = {
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


async def ollama_chat_once(
    model,
    msgs,
    *,
    temperature,
    max_tokens,
    top_p,
    stop,
    frequency_penalty=None,
    presence_penalty=None,
    tools=None,
    request_timeout=None,
):
    pl = build_ollama_payload(
        model,
        msgs,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        stop=stop,
        stream=False,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        tools=tools,
    )
    tout = request_timeout or timeout_for_model(model)

    async def _do():
        async with _ollama_session.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=pl,
            timeout=aiohttp.ClientTimeout(total=tout),
        ) as r:
            if r.status != 200:
                raise RuntimeError(f"Ollama {r.status}: {await r.text()}")
            return await r.json()

    if _is_cloud_model(model):
        return await _do()
    async with _gpu_semaphore:
        return await _do()


async def ollama_chat_stream(
    model,
    msgs,
    *,
    temperature,
    max_tokens,
    top_p,
    stop,
    frequency_penalty=None,
    presence_penalty=None,
):
    pl = build_ollama_payload(
        model,
        msgs,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        stop=stop,
        stream=True,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
    )
    tout = timeout_for_model(model)

    async def _do():
        async with _ollama_session.post(
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

    if _is_cloud_model(model):
        async for i in _do():
            yield i
    else:
        async with _gpu_semaphore:
            async for i in _do():
                yield i


# ══════════════════════════════════════════════════════════════════════════════
#  FIX 2: Improved router classification
#  - Better system prompt with few-shot examples
#  - Explicit /no_think instruction for qwen3 (disables chain-of-thought)
#  - FIX 4: Smarter fallback on parse failure
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
    """Robustly extract a JSON object from model output.

    Handles: markdown fences, preamble text, thinking tags, trailing garbage.
    """
    if not raw or not raw.strip():
        return None

    text = raw.strip()

    # Strip <think>...</think> blocks (some models emit these)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Strip markdown fences: ```json ... ``` or ``` ... ```
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find the first { ... } block in the text
    brace_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    # Find nested braces (for cases like {"key": {"nested": ...}})
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


# ┌─────────────────────────────────────────────────────────────────────────┐
# │ FIX 4: Keyword-based fallback classification                           │
# │ When the router model fails to produce valid JSON, instead of          │
# │ defaulting to general@0.45, use keyword heuristics to at least get     │
# │ the task_type right. Also retry once before giving up.                 │
# └─────────────────────────────────────────────────────────────────────────┘

def _keyword_fallback_classify(text: str) -> Dict[str, Any]:
    """Last-resort classification using simple keyword counts.

    Used when the router model fails to produce valid JSON.
    Returns a classification dict with moderate confidence — enough to
    pick the right worker pool but not enough to trigger fast path.
    """
    text_lower = text.lower()

    # Count keyword hits per category
    code_score = 0
    reasoning_score = 0

    # Code indicators
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

    # Check for code blocks or import statements
    if "```" in text or re.search(r"(?:import |from \S+ import |#include )", text):
        code_score += 5

    # Reasoning indicators
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
    if reasoning_score >= 2 or (reasoning_score >= 1 and code_score == 0 and len(text) > 100):
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


async def classify_request(msgs):
    if has_vision_content(msgs):
        return {
            "task_type": "vl",
            "confidence": 0.99,
            "needs_vision": True,
            "route_reason": "Image detected.",
        }
    # ┌─────────────────────────────────────────────────────────────────────────┐
    # │ FIX 5: Run keyword pre-filter on the LAST USER MESSAGE, not the full   │
    # │ flattened conversation. This prevents pasted code blocks from           │
    # │ dominating classification when the user's actual question is different. │
    # │ The full conversation is still passed to the router model for context.  │
    # └─────────────────────────────────────────────────────────────────────────┘
    user_text = get_last_user_text(msgs)
    flat = flatten_messages(msgs)
    # Pre-filter on user's actual question; pass full flat text as fallback
    # for the review override check (in case the user text is very short).
    kw = keyword_prefilter(user_text or flat, user_text=user_text)
    if kw:
        return kw

    # ── Router model classification ──
    rp = [
        {"role": "system", "content": _ROUTER_SYSTEM},
        {"role": "user", "content": flat},
    ]

    # FIX 4: Try up to 2 times before falling back to keyword heuristics
    for attempt in range(2):
        try:
            raw = (
                await ollama_chat_once(
                    ROUTER_MODEL,
                    rp,
                    temperature=0.0,
                    max_tokens=120,
                    top_p=0.9,
                    stop=None,
                    request_timeout=timeout_for_model(ROUTER_MODEL, is_router=True),
                )
            )["message"]["content"]

            p = _extract_json(raw)
            if p:
                tt = p.get("task_type", "general")
                if tt not in {"code", "reasoning", "vl", "general"}:
                    tt = "general"
                return {
                    "task_type": tt,
                    "confidence": max(0.0, min(float(p.get("confidence", 0.5)), 1.0)),
                    "needs_vision": bool(p.get("needs_vision", False)),
                    "route_reason": str(p.get("route_reason", "router")),
                }

            # Parse failed — log and retry (or fall through)
            logger.warning(
                "Router parse failure (attempt %d). Raw: %s", attempt + 1, raw[:300]
            )

        except Exception as e:
            logger.warning("Router call failed (attempt %d): %s", attempt + 1, e)

    # ── FIX 4: Both attempts failed — use keyword fallback instead of blind default ──
    user_text = get_last_user_text(msgs)
    fallback = _keyword_fallback_classify(user_text or flat)
    logger.info(
        "Router exhausted — keyword fallback: type=%s conf=%.2f reason=%s",
        fallback["task_type"], fallback["confidence"], fallback["route_reason"],
    )
    return fallback


# ── Model selection (uses MODEL_REGISTRY for fast path) ─────────────────────
def select_fast_model(task_type: str) -> Optional[str]:
    """Pick the highest-priority healthy model for a task type from the registry."""
    candidates = MODEL_REGISTRY.get(task_type, MODEL_REGISTRY.get("general", []))
    for entry in sorted(candidates, key=lambda e: e.get("priority", 0), reverse=True):
        name = entry["name"]
        if is_model_healthy(name) and (name in _available_models or _is_cloud_model(name)):
            return name
    return None


# ── Worker prompts ───────────────────────────────────────────────────────────
_WSF = "\n\nStructure: ## Approach\n## Answer\n## Caveats\n"


def role_prompt(tt, wn, structured=False):
    if tt == "code":
        b = (
            "Focus on implementation, correctness, bugs."
            if "coder" in wn
            else "Focus on reasoning, tradeoffs, edge cases."
            if "deepseek" in wn
            else "Clearest practical technical answer."
        )
    elif tt == "reasoning":
        b = (
            "Reason step by step."
            if "deepseek" in wn or "cogito" in wn
            else "Clarity, practicality, actionable conclusion."
        )
    elif tt == "vl":
        b = (
            "Interpret visual content accurately."
            if "vl" in wn or "llava" in wn
            else "Clear explanation from visual analysis."
        )
    else:
        b = "Clearest, most useful answer."
    return b + _WSF if structured else b


async def run_model_once(
    model,
    msgs,
    *,
    temperature,
    max_tokens,
    top_p,
    stop,
    frequency_penalty=None,
    presence_penalty=None,
):
    d = await ollama_chat_once(
        model,
        msgs,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        stop=stop,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
    )
    c = d["message"]["content"]
    if not c or not c.strip():
        raise RuntimeError("Empty content")
    return c


# Tool-aware model runner for workers and fast path
async def run_model_with_tools(
    model,
    msgs,
    *,
    temperature,
    max_tokens,
    top_p,
    stop,
    frequency_penalty=None,
    presence_penalty=None,
):
    """Run a model with tool-calling support via the tool registry."""
    if not TOOLS_ENABLED or not _tool_registry or not _tool_registry.has_tools:
        return await run_model_once(
            model,
            msgs,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        )

    tool_defs = _tool_registry.tool_definitions

    async def chat_fn(current_msgs):
        return await ollama_chat_once(
            model,
            current_msgs,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            tools=tool_defs,
        )

    content, _ = await _tool_registry.run_with_tools(chat_fn, msgs)
    return content


# ══════════════════════════════════════════════════════════════════════════════
#  Planning node — decomposes complex queries for deep workers
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


async def plan_sub_tasks(msgs: List[Dict[str, Any]], task_type: str) -> Optional[List[str]]:
    """Use the router model to optionally decompose a complex query into sub-tasks."""
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
                {"role": "user", "content": f"Task type: {task_type}\n\nRequest:\n{user_text}"},
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
            result["complete"], result["quality"], result["missing"][:80],
        )
        return result
    except Exception as e:
        logger.warning("Reflection failed: %s — assuming complete", e)
        return {"complete": True, "quality": "good", "missing": ""}


# ── Pydantic ─────────────────────────────────────────────────────────────────
class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: Any


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: Optional[bool] = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    stop: Optional[Any] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None


class OrchestratorState(TypedDict, total=False):
    request_id: str
    requested_model: str
    messages: List[Dict[str, Any]]
    original_messages: List[Dict[str, Any]]
    stream: bool
    temperature: float
    max_tokens: Optional[int]
    top_p: Optional[float]
    stop: Optional[Any]
    frequency_penalty: Optional[float]
    presence_penalty: Optional[float]
    task_type: str
    confidence: float
    needs_vision: bool
    route_reason: str
    selected_model: str
    fallback_models: List[str]
    fallback_synthesizer: str
    result_text: str
    errors: List[str]
    started_at: float
    latency_ms: int
    deep_workers: List[str]
    worker_outputs: List[Dict[str, str]]
    synthesizer: str
    synthesis_messages: List[Dict[str, Any]]
    prompt_tokens: int
    completion_tokens: int
    search_performed: bool
    search_query: str
    search_results: List[Dict[str, str]]
    # Fast-path fields
    use_fast_path: bool
    fast_model: str
    # Agentic fields
    sub_tasks: Optional[List[str]]
    react_rounds: int
    reflection_result: Dict[str, Any]
    reflection_retries: int
    escalated: bool
    tools_used: List[Dict[str, Any]]


# ══════════════════════════════════════════════════════════════════════════════
#  Pipeline nodes
# ══════════════════════════════════════════════════════════════════════════════

async def node_classify(s):
    """Classify and inject datetime into messages."""
    s["messages"] = inject_datetime(s["messages"])
    s.update(await classify_request(s["messages"]))

    # Initialize agentic fields
    s.setdefault("sub_tasks", None)
    s.setdefault("react_rounds", 0)
    s.setdefault("reflection_result", {})
    s.setdefault("reflection_retries", 0)
    s.setdefault("escalated", False)

    # Decide fast path vs deep panel
    s["use_fast_path"] = False
    s["fast_model"] = ""

    # "audrey_deep" tries fast path when confidence is high enough
    requested = s["requested_model"]
    if FAST_PATH_ENABLED and requested == "audrey_deep":
        if s["confidence"] >= FAST_PATH_CONFIDENCE:
            fm = select_fast_model(s["task_type"])
            if fm:
                s["use_fast_path"] = True
                s["fast_model"] = fm
                s["route_reason"] += f" → fast:{fm}"

    return s


async def web_search_node(s):
    s["search_performed"] = False
    s["search_query"] = ""
    s["search_results"] = []
    s["original_messages"] = [m.copy() for m in s["messages"]]

    flat = flatten_messages(s["messages"])
    if not needs_web_search(flat):
        return s

    last = get_last_user_text(s["messages"])
    if not last:
        return s

    q = extract_search_query(last)
    results = await web_search(q)
    if not results:
        return s

    s["search_performed"] = True
    s["search_query"] = q
    s["search_results"] = results

    ctx = format_search_results(results)
    new = []
    ins = False
    for m in s["messages"]:
        if not ins and m.get("role") != "system":
            new.append({"role": "system", "content": ctx})
            ins = True
        new.append(m)
    if not ins:
        new.append({"role": "system", "content": ctx})
    s["messages"] = new
    return s


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


async def node_react_agent(s):
    """ReAct loop — think, act, observe, repeat."""
    model = s["fast_model"]
    task_prompt = role_prompt(s["task_type"], model)

    sys_msg = {
        "role": "system",
        "content": f"{_REACT_SYSTEM}\n\nFocus: {task_prompt}",
    }
    msgs = [sys_msg, *s["messages"]]

    try:
        if TOOLS_ENABLED and _tool_registry and _tool_registry.has_tools:
            tool_defs = _tool_registry.tool_definitions

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
                _tool_registry.run_with_tools(chat_fn, msgs),
                timeout=FAST_PATH_TIMEOUT,
            )

            tool_rounds = sum(
                1 for m in final_msgs
                if m.get("role") == "assistant" and m.get("tool_calls")
            )
            s["react_rounds"] = tool_rounds

            # ┌─────────────────────────────────────────────────────────┐
            # │ FIX 6: Store tool usage for banner + log observability  │
            # └─────────────────────────────────────────────────────────┘
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

async def node_adaptive_escalate(s):
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
        question_len = len(get_last_user_text(s.get("original_messages", s["messages"])))
        if question_len > 50:
            logger.info(
                "Adaptive escalation: response too short (%d chars) for question (%d chars)",
                len(result.strip()), question_len,
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
                    reflection["missing"][:60], retries + 1,
                )
                s["reflection_retries"] = retries + 1

                model = s["fast_model"]
                retry_msgs = [
                    {"role": "system", "content": role_prompt(s["task_type"], model)},
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
                logger.info("Reflection: quality=poor — escalating to deep panel")
                s["use_fast_path"] = False
                s["escalated"] = True
                s["route_reason"] += " → escalated:reflection_poor"
                return s

    s["escalated"] = False
    return s


# ══════════════════════════════════════════════════════════════════════════════
#  FIX 1: Mode-aware worker cap in node_plan_panel
# ══════════════════════════════════════════════════════════════════════════════

async def node_plan_panel(s):
    tt = s["task_type"]
    if s["confidence"] < LOW_CONFIDENCE_THRESHOLD:
        tt = "general"
        s["task_type"] = "general"
    panel = _deep_panel_for_model(s["requested_model"])[tt]

    # ┌─────────────────────────────────────────────────────────────────────┐
    # │ FIX 1: Use higher worker cap for cloud-only mode                   │
    # │ Cloud workers run truly in parallel (no GPU semaphore), so more    │
    # │ workers = better draft diversity at minimal cost.                   │
    # └─────────────────────────────────────────────────────────────────────┘
    if s["requested_model"] == "audrey_cloud":
        max_workers = MAX_DEEP_WORKERS_CLOUD
    else:
        max_workers = MAX_DEEP_WORKERS

    ws = panel["workers"][:max_workers]
    # Filter to healthy + available models
    healthy = [
        w
        for w in ws
        if is_model_healthy(w) and (w in _available_models or _is_cloud_model(w))
    ]
    ws = healthy or panel["workers"][:max_workers]
    s["deep_workers"] = ws
    s["synthesizer"] = panel["synthesizer"]
    s["fallback_synthesizer"] = panel.get("fallback_synthesizer", "")

    # Plan sub-tasks for complex queries
    sub_tasks = await plan_sub_tasks(s["messages"], tt)
    s["sub_tasks"] = sub_tasks

    return s


async def node_parallel_generate(s):
    """Run parallel workers, optionally with sub-task assignments."""
    base = s["messages"]
    sub_tasks = s.get("sub_tasks")

    async def one(wn, sub_task=None):
        if sub_task:
            sys_content = (
                f"{role_prompt(s['task_type'], wn, structured=True)}\n\n"
                f"You are assigned this specific sub-task:\n{sub_task}\n\n"
                f"Focus your answer on this sub-task only. Be thorough and specific."
            )
        else:
            sys_content = role_prompt(s["task_type"], wn, structured=True)

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
            return {"model": wn, "content": t, "sub_task": sub_task or "", "label": label}
        except Exception as e:
            note_model_failure(wn)
            logger.warning("Worker %s failed: %s", wn, e)
            return {"model": wn, "content": "[WORKER_ERROR] Unable to respond.", "sub_task": sub_task or ""}

    workers = s["deep_workers"]

    if sub_tasks and len(sub_tasks) >= 2:
        assignments = []
        for i, task in enumerate(sub_tasks):
            worker = workers[i % len(workers)]
            assignments.append((worker, task))
        logger.info(
            "Planning: %d sub-tasks assigned to %d workers",
            len(assignments), len(workers),
        )
    else:
        assignments = [(w, None) for w in workers]

    cloud_tasks = [(w, t) for w, t in assignments if _is_cloud_model(w)]
    local_tasks = [(w, t) for w, t in assignments if not _is_cloud_model(w)]

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


_SYNTH_SYS = """You are a synthesis model. Merge expert drafts into one coherent, comprehensive response.
Do NOT mention model names, draft numbers, or that multiple sources were consulted.
Prefer correctness and clarity. Resolve any contradictions by favoring the more detailed answer.
If drafts reference tool results or web search data, integrate that information naturally.
If drafts address different sub-tasks of a complex question, combine them into a unified answer."""


def build_synth_msgs(s):
    outputs = s["worker_outputs"]
    secs = []
    for i, o in enumerate(outputs, 1):
        label = o.get("label", "")
        secs.append(f"── Draft {i} ({o['model']}){label} ──\n{o['content']}")
    return [
        {"role": "system", "content": _SYNTH_SYS},
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
    """Reflection gate for deep panel output. Re-synthesizes if quality is poor."""
    if not REFLECTION_ENABLED or not s.get("result_text"):
        return s

    reflection = await reflect_on_response(
        s.get("original_messages", s["messages"]),
        s["result_text"],
    )
    s["reflection_result"] = reflection

    if reflection["quality"] == "poor" and s.get("reflection_retries", 0) < 1:
        s["reflection_retries"] = s.get("reflection_retries", 0) + 1
        logger.info("Deep reflection: quality=poor — re-synthesizing with feedback")

        s["synthesis_messages"].append({
            "role": "user",
            "content": (
                f"The previous synthesis was rated as poor quality. "
                f"Missing: {reflection['missing']}. "
                f"Please re-synthesize with more attention to completeness and accuracy."
            ),
        })
        return await node_synthesize(s)

    return s


# ── Graph builders ───────────────────────────────────────────────────────────

def build_deep_graph(include_synthesis=True):
    """Build the deep-panel pipeline graph with planning + reflection."""
    g = StateGraph(OrchestratorState)
    for n, f in [
        ("classify", node_classify),
        ("web_search", web_search_node),
        ("plan", node_plan_panel),
        ("parallel", node_parallel_generate),
        ("prep", node_prepare_synthesis),
    ]:
        g.add_node(n, f)
    g.set_entry_point("classify")
    g.add_edge("classify", "web_search")
    g.add_edge("web_search", "plan")
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


def build_fast_graph():
    """Build the fast path with ReAct agent + adaptive escalation."""
    g = StateGraph(OrchestratorState)
    for n, f in [
        ("classify", node_classify),
        ("web_search", web_search_node),
        ("react_agent", node_react_agent),
        ("escalate", node_adaptive_escalate),
    ]:
        g.add_node(n, f)
    g.set_entry_point("classify")
    g.add_edge("classify", "web_search")
    g.add_edge("web_search", "react_agent")
    g.add_edge("react_agent", "escalate")
    g.add_edge("escalate", END)
    return g.compile()


SYNTH_GRAPH = build_deep_graph(True)
PREP_GRAPH = build_deep_graph(False)
FAST_GRAPH = build_fast_graph()


# ══════════════════════════════════════════════════════════════════════════════
#  Startup — model validation
# ══════════════════════════════════════════════════════════════════════════════

async def validate_models():
    """Check which configured models are actually available in Ollama."""
    global _available_models
    try:
        async with _ollama_session.get(
            f"{OLLAMA_BASE_URL}/api/tags",
            timeout=aiohttp.ClientTimeout(total=10),
        ) as r:
            if r.status != 200:
                logger.warning("Cannot reach Ollama for model validation")
                return
            data = await r.json()
        local_models = {m["name"] for m in data.get("models", [])}
        _available_models = local_models
        logger.info("Ollama models available: %d — %s", len(local_models), ", ".join(sorted(local_models)))

        # Warn about configured models that are missing
        configured = set()
        for category in MODEL_REGISTRY.values():
            for entry in category:
                configured.add(entry["name"])
        for panel_set in [DEEP_PANEL_MIXED, DEEP_PANEL_CLOUD, DEEP_PANEL_LOCAL]:
            for cat in panel_set.values():
                for w in cat.get("workers", []):
                    configured.add(w)
                configured.add(cat.get("synthesizer", ""))
                configured.add(cat.get("fallback_synthesizer", ""))
        configured.discard("")

        for name in sorted(configured):
            if _is_cloud_model(name):
                continue
            if name not in local_models:
                logger.warning(
                    "⚠ Configured model NOT in Ollama: %s — requests using it will fail",
                    name,
                )
    except Exception as e:
        logger.warning("Model validation failed: %s", e)


# ══════════════════════════════════════════════════════════════════════════════
#  FastAPI app
# ══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app):
    global _ollama_session, _ext_session, _tool_registry

    _ollama_session = aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=360),
        connector=aiohttp.TCPConnector(limit=20),
    )
    _ext_session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15))

    _tool_registry = ToolRegistry(session=_ext_session)

    if TOOL_SERVER_URLS and TOOLS_ENABLED:
        await _tool_registry.discover(TOOL_SERVER_URLS)
        logger.info(
            "Tools: %d discovered from %d servers",
            _tool_registry.tool_count,
            len(TOOL_SERVER_URLS),
        )

    await validate_models()

    logger.info(
        "Orchestrator ready  router=%s  tools=%s(%d)  search=%s  fast_path=%s  "
        "react=%s  reflect=%s  plan=%s  escalate=%s  "
        "max_workers_local=%d  max_workers_cloud=%d",
        ROUTER_MODEL,
        TOOLS_ENABLED,
        _tool_registry.tool_count,
        SEARCH_BACKEND,
        FAST_PATH_ENABLED,
        REACT_MAX_ROUNDS,
        REFLECTION_ENABLED,
        PLANNING_ENABLED,
        ESCALATION_ENABLED,
        MAX_DEEP_WORKERS,
        MAX_DEEP_WORKERS_CLOUD,
    )

    yield

    await _ollama_session.close()
    await _ext_session.close()


app = FastAPI(
    title="LangGraph Auto-Orchestrator",
    version="6.1.0",
    lifespan=lifespan,
)


async def verify_api_key(req: Request):
    if not API_KEY:
        return
    auth = req.headers.get("Authorization", "")
    if not auth.startswith("Bearer ") or auth[7:] != API_KEY:
        raise HTTPException(401, "Invalid or missing API key")


@app.get("/health")
async def healthcheck():
    try:
        async with _ollama_session.get(
            f"{OLLAMA_BASE_URL}/api/tags",
            timeout=aiohttp.ClientTimeout(total=5),
        ) as r:
            if r.status == 200:
                return {
                    "ok": True,
                    "ollama": "reachable",
                    "router_model": ROUTER_MODEL,
                    "cache": _cache.stats,
                    "tools": {
                        "enabled": TOOLS_ENABLED,
                        "count": _tool_registry.tool_count if _tool_registry else 0,
                        "servers": _tool_registry.server_info if _tool_registry else {},
                    },
                    "fast_path": FAST_PATH_ENABLED,
                    "agentic": {
                        "react_max_rounds": REACT_MAX_ROUNDS,
                        "reflection": REFLECTION_ENABLED,
                        "planning": PLANNING_ENABLED,
                        "escalation": ESCALATION_ENABLED,
                    },
                    "max_workers": {
                        "local": MAX_DEEP_WORKERS,
                        "cloud": MAX_DEEP_WORKERS_CLOUD,
                    },
                    "available_models": len(_available_models),
                }
            return JSONResponse(503, {"ok": False, "ollama": f"status {r.status}"})
    except Exception as e:
        return JSONResponse(503, {"ok": False, "ollama": f"unreachable: {e}"})


@app.get("/v1/models", dependencies=[Depends(verify_api_key)])
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": m, "object": "model", "owned_by": "orchestrator"}
            for m in sorted(_ALL_VIRTUAL_MODELS)
        ],
    }


@app.post("/v1/tools/rediscover", dependencies=[Depends(verify_api_key)])
async def rediscover_tools():
    if _tool_registry and TOOL_SERVER_URLS:
        await _tool_registry.rediscover(TOOL_SERVER_URLS)
        return {
            "tools": _tool_registry.tool_count,
            "names": _tool_registry.tool_names,
            "servers": _tool_registry.server_info,
        }
    return {"tools": 0}


# ── Request runner ───────────────────────────────────────────────────────────
async def run_graph_dispatch(req, *, stream_prepare_only=False):
    msgs_raw = [m.model_dump() for m in req.messages]
    temp = req.temperature if req.temperature is not None else DEFAULT_TEMPERATURE

    # Check cache before running the pipeline
    if CACHE_ENABLED and not req.stream:
        cached = _cache.get(msgs_raw, req.model, temp)
        if cached is not None:
            logger.info("Cache hit for model=%s", req.model)
            return {
                "request_id": str(uuid.uuid4()),
                "requested_model": req.model,
                "result_text": cached,
                "selected_model": "cache",
                "task_type": "cached",
                "confidence": 1.0,
                "route_reason": "Cache hit",
                "latency_ms": 0,
                "prompt_tokens": estimate_tokens(flatten_messages(msgs_raw)),
                "completion_tokens": estimate_tokens(cached),
                "search_performed": False,
                "use_fast_path": False,
                "escalated": False,
            }

    s = {
        "request_id": str(uuid.uuid4()),
        "requested_model": req.model,
        "messages": msgs_raw,
        "stream": bool(req.stream),
        "temperature": temp,
        "max_tokens": req.max_tokens,
        "top_p": req.top_p,
        "stop": req.stop,
        "frequency_penalty": req.frequency_penalty,
        "presence_penalty": req.presence_penalty,
        "errors": [],
        "started_at": time.time(),
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "search_performed": False,
        "search_query": "",
        "search_results": [],
        "use_fast_path": False,
        "fast_model": "",
        # Agentic state
        "sub_tasks": None,
        "react_rounds": 0,
        "reflection_result": {},
        "reflection_retries": 0,
        "escalated": False,
        "tools_used": [],
    }

    # Try fast path first for audrey_deep
    if (
        FAST_PATH_ENABLED
        and req.model == "audrey_deep"
        and not stream_prepare_only
    ):
        r = await FAST_GRAPH.ainvoke(s)

        if r.get("use_fast_path") and r.get("result_text") and not r.get("escalated"):
            r["latency_ms"] = int((time.time() - s["started_at"]) * 1000)
            if CACHE_ENABLED:
                _cache.put(msgs_raw, req.model, temp, r["result_text"])
            logger.info(
                json.dumps({
                    "rid": r["request_id"],
                    "model": r["requested_model"],
                    "type": r.get("task_type"),
                    "conf": r.get("confidence"),
                    "selected": r.get("selected_model"),
                    "path": "fast+react",
                    "react_rounds": r.get("react_rounds", 0),
                    "reflection": r.get("reflection_result", {}).get("quality", "n/a"),
                    "search": r.get("search_performed", False),
                    "search_query": r.get("search_query", ""),
                    "tools": [t.get("tool", "") for t in r.get("tools_used", [])],
                    "ms": r["latency_ms"],
                })
            )
            return r

        # Fast path was skipped, failed, or escalated — fall through to deep panel
        s = r
        s["started_at"] = time.time()
        if r.get("escalated"):
            logger.info("Escalated from fast→deep: %s", r.get("route_reason", ""))

    # Deep panel path (with planning + reflection)
    g = PREP_GRAPH if stream_prepare_only else SYNTH_GRAPH
    r = await g.ainvoke(s)
    r["latency_ms"] = int((time.time() - s["started_at"]) * 1000)

    if CACHE_ENABLED and not stream_prepare_only and r.get("result_text"):
        _cache.put(msgs_raw, req.model, temp, r["result_text"])

    logger.info(
        json.dumps({
            "rid": r["request_id"],
            "model": r["requested_model"],
            "type": r.get("task_type"),
            "conf": r.get("confidence"),
            "selected": r.get("selected_model", r.get("synthesizer")),
            "path": "deep",
            "planned": bool(r.get("sub_tasks")),
            "reflection": r.get("reflection_result", {}).get("quality", "n/a"),
            "search": r.get("search_performed", False),
            "search_query": r.get("search_query", ""),
            "tools": [t.get("tool", "") for t in r.get("tools_used", [])],
            "escalated": r.get("escalated", False),
            "ms": r["latency_ms"],
        })
    )
    return r


def banner(s):
    sel = s.get("selected_model") or s.get("synthesizer") or "?"
    path = "fast+react" if s.get("use_fast_path") else "deep"
    esc = " | ESCALATED" if s.get("escalated") else ""
    plan = " | planned" if s.get("sub_tasks") else ""
    react = f" | react×{s.get('react_rounds', 0)}" if s.get("react_rounds") else ""
    refl = ""
    rr = s.get("reflection_result", {})
    if rr.get("quality"):
        refl = f" | refl:{rr['quality']}"

    # ┌─────────────────────────────────────────────────────────────────────┐
    # │ FIX 6: Show search query and tool names in the banner              │
    # └─────────────────────────────────────────────────────────────────────┘
    sr = ""
    if s.get("search_performed"):
        sq = s.get("search_query", "")
        sr = f" | 🌐 search: \"{sq}\"" if sq else " | +search"

    tools_str = ""
    tools_log = s.get("tools_used", [])
    if tools_log:
        # Deduplicate tool names preserving order, strip server prefix for readability
        seen = {}
        for t in tools_log:
            raw_name = t.get("tool", "")
            # Strip "servername__" prefix for cleaner display
            short = raw_name.split("__", 1)[-1] if "__" in raw_name else raw_name
            if short not in seen:
                seen[short] = 0
            seen[short] += 1
        parts = [f"{n}×{c}" if c > 1 else n for n, c in seen.items()]
        tools_str = f" | 🔧 tools: {', '.join(parts)}"

    return (
        f"[{s.get('requested_model')} → {sel} | {s.get('task_type')} "
        f"| conf {s.get('confidence', 0):.2f} | {path}{sr}{tools_str}{react}{plan}{refl}{esc} "
        f"| {s.get('route_reason', '')}]\n"
    )


# ── Streaming ────────────────────────────────────────────────────────────────
def _sc(rid, created, mn, text):
    return (
        f"data: {json.dumps({'id': rid, 'object': 'chat.completion.chunk', 'created': created, 'model': mn, 'choices': [{'index': 0, 'delta': {'content': text}, 'finish_reason': None}]})}\n\n"
    )


async def stream_fast_path(s):
    """Stream output from fast-path ReAct agent."""
    ct = int(time.time())
    rid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    mn = s["requested_model"]
    model = s["fast_model"]

    if EMIT_STATUS_UPDATES:
        yield _sc(rid, ct, mn, f"⚡ Fast path (ReAct): {model}\n\n")
    if EMIT_ROUTING_BANNER:
        yield _sc(rid, ct, mn, banner(s))

    sys_msg = {
        "role": "system",
        "content": f"{_REACT_SYSTEM}\n\nFocus: {role_prompt(s['task_type'], model)}",
    }
    msgs = [sys_msg, *s["messages"]]

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
                yield (
                    f"data: {json.dumps({'id': rid, 'object': 'chat.completion.chunk', 'created': ct, 'model': mn, 'choices': [{'index': 0, 'delta': {'content': c}, 'finish_reason': None}]})}\n\n"
                )
            if item.get("done"):
                note_model_success(model)
                yield (
                    f"data: {json.dumps({'id': rid, 'object': 'chat.completion.chunk', 'created': ct, 'model': mn, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
                )
                yield "data: [DONE]\n\n"
                return
    except Exception as e:
        note_model_failure(model)
        logger.warning("Fast streaming failed for %s: %s", model, e)
        yield _sc(rid, ct, mn, f"\n\n[Fast path error: {model}. Please retry.]\n")
        yield (
            f"data: {json.dumps({'id': rid, 'object': 'chat.completion.chunk', 'created': ct, 'model': mn, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
        )
        yield "data: [DONE]\n\n"


async def stream_synthesis(ps):
    """Stream the synthesis phase."""
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
                    yield (
                        f"data: {json.dumps({'id': rid, 'object': 'chat.completion.chunk', 'created': ct, 'model': mn, 'choices': [{'index': 0, 'delta': {'content': c}, 'finish_reason': None}]})}\n\n"
                    )
                if item.get("done"):
                    note_model_success(s)
                    yield (
                        f"data: {json.dumps({'id': rid, 'object': 'chat.completion.chunk', 'created': ct, 'model': mn, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
                    )
                    yield "data: [DONE]\n\n"
                    return
        except Exception:
            note_model_failure(s)

    yield (
        f"data: {json.dumps({'id': rid, 'object': 'chat.completion.chunk', 'created': ct, 'model': mn, 'choices': [{'index': 0, 'delta': {'content': '[Error. Please try again.]'}, 'finish_reason': 'stop'}]})}\n\n"
    )
    yield "data: [DONE]\n\n"


# ── Main endpoint ────────────────────────────────────────────────────────────
@app.post("/v1/chat/completions", dependencies=[Depends(verify_api_key)])
async def chat_completions(req: ChatCompletionRequest):
    if req.model not in _ALL_VIRTUAL_MODELS:
        raise HTTPException(400, f"Unknown model: {req.model}")
    if not req.messages:
        raise HTTPException(400, "messages empty")
    if not any(m.role == "user" for m in req.messages):
        raise HTTPException(400, "No user message")
    if req.max_tokens and req.max_tokens > 128000:
        raise HTTPException(400, "max_tokens too large")

    # ── Streaming ────────────────────────────────────────────────────────
    if req.stream:

        async def _stream():
            ct = int(time.time())
            rid = f"chatcmpl-{uuid.uuid4().hex[:24]}"

            if EMIT_STATUS_UPDATES:
                yield _sc(rid, ct, req.model, "🔍 Analyzing...\n")

            init_state = {
                "request_id": str(uuid.uuid4()),
                "requested_model": req.model,
                "messages": [m.model_dump() for m in req.messages],
                "stream": True,
                "temperature": (
                    req.temperature
                    if req.temperature is not None
                    else DEFAULT_TEMPERATURE
                ),
                "max_tokens": req.max_tokens,
                "top_p": req.top_p,
                "stop": req.stop,
                "frequency_penalty": req.frequency_penalty,
                "presence_penalty": req.presence_penalty,
                "errors": [],
                "started_at": time.time(),
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "search_performed": False,
                "search_query": "",
                "search_results": [],
                "use_fast_path": False,
                "fast_model": "",
                "sub_tasks": None,
                "react_rounds": 0,
                "reflection_result": {},
                "reflection_retries": 0,
                "escalated": False,
                "tools_used": [],
            }

            # Classify
            classified = await node_classify(init_state)
            # Web search
            searched = await web_search_node(classified)

            if searched.get("search_performed"):
                yield _sc(rid, ct, req.model, "🌐 Search completed\n")

            # Decide path
            if searched.get("use_fast_path") and searched.get("fast_model"):
                async for chunk in stream_fast_path(searched):
                    yield chunk
            else:
                # Deep panel
                planned = await node_plan_panel(searched)
                if planned.get("sub_tasks"):
                    yield _sc(rid, ct, req.model, f"📋 Planning: {len(planned['sub_tasks'])} sub-tasks\n")
                generated = await node_parallel_generate(planned)
                prepared = await node_prepare_synthesis(generated)
                async for chunk in stream_synthesis(prepared):
                    yield chunk

        return StreamingResponse(_stream(), media_type="text/event-stream")

    # ── Non-streaming ────────────────────────────────────────────────────
    final = await run_graph_dispatch(req)
    content = final["result_text"]
    if EMIT_ROUTING_BANNER:
        content = banner(final) + content
    return JSONResponse({
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": final.get("prompt_tokens", 0),
            "completion_tokens": final.get("completion_tokens", 0),
            "total_tokens": (
                final.get("prompt_tokens", 0) + final.get("completion_tokens", 0)
            ),
        },
    })
