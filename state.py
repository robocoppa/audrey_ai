"""
Audrey — shared mutable state.

Holds the async sessions, tool registry, GPU semaphore, and available-model
set.  Initialized by the FastAPI lifespan in main.py; imported read-only
by every other module.
"""

import asyncio
import time
from typing import Any

import aiohttp

from config import GPU_CONCURRENCY
from tool_registry import ToolRegistry

# Async HTTP sessions (set in main.lifespan)
ollama_session: aiohttp.ClientSession | None = None
ext_session: aiohttp.ClientSession | None = None

# Tool registry (set in main.lifespan)
tool_registry: ToolRegistry | None = None

# GPU concurrency gate — serialises local-model inference
gpu_semaphore: asyncio.Semaphore = asyncio.Semaphore(GPU_CONCURRENCY)

# Models actually present in Ollama (populated by validate_models)
available_models: set[str] = set()

# Last observed audrey_fast routing outcome (used by /health).
audrey_fast_health: dict[str, Any] = {
    "selected_model": "none",
    "last_status": "unknown",
    "last_reason": "no audrey_fast requests yet",
    "updated_at": None,
}


def update_audrey_fast_health(*, selected_model: str, success: bool, reason: str) -> None:
    audrey_fast_health["selected_model"] = selected_model or "none"
    audrey_fast_health["last_status"] = "success" if success else "failure"
    audrey_fast_health["last_reason"] = reason or ""
    audrey_fast_health["updated_at"] = int(time.time())
