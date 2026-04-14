"""
Audrey — configuration loading and environment knobs.

Reads config.yaml and environment variables.  Every tunable constant lives here
so the rest of the codebase can just ``from config import X``.
"""

import os
from typing import Any, Dict, List, Set

import yaml

# ── Config file ──────────────────────────────────────────────────────────────
CONFIG_PATH = os.getenv("CONFIG_PATH", "/app/config.yaml")


def load_config() -> Dict[str, Any]:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


_config = load_config()

MODEL_REGISTRY: Dict[str, List[Dict[str, Any]]] = _config["model_registry"]
DEEP_PANEL_CLOUD: Dict[str, Any] = _config["deep_panel_cloud"]
DEEP_PANEL_LOCAL: Dict[str, Any] = _config["deep_panel_local"]
DEEP_PANEL_MIXED: Dict[str, Any] = _config["deep_panel"]
TIMEOUTS: Dict[str, int] = _config.get("timeouts", {})
CACHE_CONFIG: Dict[str, Any] = _config.get("cache", {})
FAST_PATH_CONFIG: Dict[str, Any] = _config.get("fast_path", {})
TOOL_SERVER_URLS: List[str] = _config.get("tool_servers", [])
AGENTIC_CONFIG: Dict[str, Any] = _config.get("agentic", {})
SEARCH_CONFIG: Dict[str, Any] = _config.get("search", {})
TOOLS_CONFIG: Dict[str, Any] = _config.get("tools", {})

# ── Agentic sub-sections (for cleaner access below) ─────────────────────────
_AGENTIC_REACT: Dict[str, Any] = AGENTIC_CONFIG.get("react", {})
_AGENTIC_PLANNING: Dict[str, Any] = AGENTIC_CONFIG.get("planning", {})
_AGENTIC_REFLECTION: Dict[str, Any] = AGENTIC_CONFIG.get("reflection", {})
_AGENTIC_ESCALATION: Dict[str, Any] = AGENTIC_CONFIG.get("escalation", {})

# ── Environment knobs ────────────────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
ROUTER_MODEL = os.getenv("ROUTER_MODEL", "qwen3:4b")
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.2"))
LOW_CONFIDENCE_THRESHOLD = float(os.getenv("LOW_CONFIDENCE_THRESHOLD", "0.60"))
MAX_DEEP_WORKERS = int(os.getenv("MAX_DEEP_WORKERS", "2"))
MAX_DEEP_WORKERS_CLOUD = int(os.getenv("MAX_DEEP_WORKERS_CLOUD", "3"))
EMIT_ROUTING_BANNER = os.getenv("EMIT_ROUTING_BANNER", "true").lower() == "true"
GPU_CONCURRENCY = int(os.getenv("GPU_CONCURRENCY", "1"))
EMIT_STATUS_UPDATES = os.getenv("EMIT_STATUS_UPDATES", "true").lower() == "true"
DEEP_WORKER_TIMEOUT = int(
    os.getenv("DEEP_WORKER_TIMEOUT", str(TIMEOUTS.get("deep_worker", 240)))
)
API_KEY = os.getenv("API_KEY", "")
TOOLS_ENABLED = os.getenv(
    "TOOLS_ENABLED", str(TOOLS_CONFIG.get("enabled", True))
).lower() in ("true", "1", "yes")
SEARCH_BACKEND = os.getenv("SEARCH_BACKEND", SEARCH_CONFIG.get("backend", "searxng"))
SEARXNG_URL = os.getenv("SEARXNG_URL", "http://searxng:8080")
SEARXNG_MAX_RESULTS = int(os.getenv(
    "SEARXNG_MAX_RESULTS", str(SEARCH_CONFIG.get("max_results", 5))
))
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY", "")
BRAVE_MAX_RESULTS = int(os.getenv("BRAVE_MAX_RESULTS", "5"))

# ── Fast-path knobs ──────────────────────────────────────────────────────────
FAST_PATH_ENABLED = FAST_PATH_CONFIG.get("enabled", True)
FAST_PATH_CONFIDENCE = FAST_PATH_CONFIG.get("confidence_threshold", 0.88)
FAST_PATH_TOOL_MODELS: Set[str] = set(FAST_PATH_CONFIG.get("tool_capable_models", []))
FAST_PATH_TIMEOUT = int(TIMEOUTS.get("fast_path", 180))

# ── Complexity gate ──────────────────────────────────────────────────────────
# Force deep panel when input exceeds this token estimate, regardless of
# confidence.  Prevents large code pastes / doc reviews from fast-pathing.
COMPLEXITY_FORCE_DEEP = os.getenv("COMPLEXITY_FORCE_DEEP", "true").lower() == "true"
COMPLEXITY_TOKEN_THRESHOLD = int(os.getenv("COMPLEXITY_TOKEN_THRESHOLD", "500"))

# ── Agentic knobs (YAML defaults, env var overrides) ────────────────────────
REACT_MAX_ROUNDS = int(os.getenv(
    "REACT_MAX_ROUNDS", str(_AGENTIC_REACT.get("max_rounds", 3))
))
REACT_COMPRESS_AFTER = int(os.getenv(
    "COMPRESS_AFTER_ROUNDS", str(_AGENTIC_REACT.get("compress_after_rounds", 2))
))
REACT_COMPRESS_MAX_CHARS = int(os.getenv(
    "COMPRESS_MAX_RESULT_CHARS", str(_AGENTIC_REACT.get("compress_max_result_chars", 2000))
))
REFLECTION_ENABLED = os.getenv(
    "REFLECTION_ENABLED", str(_AGENTIC_REFLECTION.get("enabled", True))
).lower() in ("true", "1", "yes")
REFLECTION_MAX_RETRIES = int(os.getenv(
    "REFLECTION_MAX_RETRIES", str(_AGENTIC_REFLECTION.get("max_retries", 1))
))
PLANNING_ENABLED = os.getenv(
    "PLANNING_ENABLED", str(_AGENTIC_PLANNING.get("enabled", True))
).lower() in ("true", "1", "yes")
PLANNING_MIN_TOKENS = int(os.getenv(
    "PLANNING_MIN_TOKENS", str(_AGENTIC_PLANNING.get("min_tokens", 40))
))
ESCALATION_ENABLED = os.getenv(
    "ESCALATION_ENABLED", str(_AGENTIC_ESCALATION.get("enabled", True))
).lower() in ("true", "1", "yes")
ESCALATION_MIN_LENGTH = int(os.getenv(
    "ESCALATION_MIN_LENGTH", str(_AGENTIC_ESCALATION.get("min_response_length", 100))
))
ESCALATION_CONFIDENCE_CEILING = float(os.getenv(
    "ESCALATION_CONFIDENCE_CEILING", str(_AGENTIC_ESCALATION.get("confidence_ceiling", 0.95))
))
MAX_TOOL_ROUNDS = int(os.getenv(
    "MAX_TOOL_ROUNDS", str(TOOLS_CONFIG.get("max_tool_rounds", 5))
))

# ── Cache knobs ──────────────────────────────────────────────────────────────
CACHE_ENABLED = CACHE_CONFIG.get("enabled", True)
CACHE_MAX = CACHE_CONFIG.get("max_entries", 256)
CACHE_TTL = CACHE_CONFIG.get("ttl_seconds", 600)

# ── Virtual models ───────────────────────────────────────────────────────────
ALL_VIRTUAL_MODELS = {"audrey_deep", "audrey_local", "audrey_cloud"}


def is_cloud_model(name: str) -> bool:
    return ":cloud" in name


def deep_panel_for_model(name: str) -> Dict[str, Any]:
    if name == "audrey_local":
        return DEEP_PANEL_LOCAL
    if name == "audrey_cloud":
        return DEEP_PANEL_CLOUD
    return DEEP_PANEL_MIXED  # "audrey_deep" uses mixed
