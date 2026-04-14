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
TOOLS_ENABLED = os.getenv("TOOLS_ENABLED", "true").lower() == "true"
SEARCH_BACKEND = os.getenv("SEARCH_BACKEND", "searxng")
SEARXNG_URL = os.getenv("SEARXNG_URL", "http://searxng:8080")
SEARXNG_MAX_RESULTS = int(os.getenv("SEARXNG_MAX_RESULTS", "5"))
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

# ── Agentic knobs ────────────────────────────────────────────────────────────
REACT_MAX_ROUNDS = int(os.getenv("REACT_MAX_ROUNDS", "3"))
REFLECTION_ENABLED = os.getenv("REFLECTION_ENABLED", "true").lower() == "true"
PLANNING_ENABLED = os.getenv("PLANNING_ENABLED", "true").lower() == "true"
PLANNING_MIN_TOKENS = int(os.getenv("PLANNING_MIN_TOKENS", "40"))
ESCALATION_ENABLED = os.getenv("ESCALATION_ENABLED", "true").lower() == "true"
ESCALATION_MIN_LENGTH = int(os.getenv("ESCALATION_MIN_LENGTH", "100"))
ESCALATION_CONFIDENCE_CEILING = float(
    os.getenv("ESCALATION_CONFIDENCE_CEILING", "0.95")
)
REFLECTION_MAX_RETRIES = int(os.getenv("REFLECTION_MAX_RETRIES", "1"))

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
