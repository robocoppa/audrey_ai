"""
Audrey — shared mutable state.

Holds the async sessions, tool registry, GPU semaphore, and available-model
set.  Initialized by the FastAPI lifespan in main.py; imported read-only
by every other module.
"""

import asyncio
from typing import Optional, Set

import aiohttp

from config import GPU_CONCURRENCY
from tool_registry import ToolRegistry

# Async HTTP sessions (set in main.lifespan)
ollama_session: Optional[aiohttp.ClientSession] = None
ext_session: Optional[aiohttp.ClientSession] = None

# Tool registry (set in main.lifespan)
tool_registry: Optional[ToolRegistry] = None

# GPU concurrency gate — serialises local-model inference
gpu_semaphore: asyncio.Semaphore = asyncio.Semaphore(GPU_CONCURRENCY)

# Models actually present in Ollama (populated by validate_models)
available_models: Set[str] = set()
