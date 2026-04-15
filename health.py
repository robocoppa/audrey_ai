"""
Audrey — model health tracking.

Records failures per model and applies exponential-backoff cooldowns so
unhealthy models are temporarily skipped during routing.
"""

import logging
import time
from typing import Any

logger = logging.getLogger("audrey.health")

MODEL_HEALTH: dict[str, dict[str, Any]] = {}


def health_record(name: str) -> dict[str, Any]:
    if name not in MODEL_HEALTH:
        MODEL_HEALTH[name] = {"failures": 0, "last_failure": None, "cooldown_until": 0}
    return MODEL_HEALTH[name]


def is_model_healthy(name: str) -> bool:
    return time.time() >= float(health_record(name).get("cooldown_until", 0) or 0)


def note_model_failure(name: str) -> None:
    r = health_record(name)
    r["failures"] += 1
    r["last_failure"] = time.time()
    if r["failures"] >= 2:
        cd = min(60 * (2 ** (r["failures"] - 2)), 1800)
        r["cooldown_until"] = time.time() + cd
        logger.warning("Model %s cooled %ds (failures=%d)", name, cd, r["failures"])


def note_model_success(name: str) -> None:
    r = health_record(name)
    r["failures"] = 0
    r["cooldown_until"] = 0
