"""
Audrey — request/response models and pipeline state.
"""

from typing import Any, Literal

from pydantic import BaseModel


# ── OpenAI-compatible request models ─────────────────────────────────────────

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: Any


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    stream: bool | None = False
    audrey_mode: Literal["quick", "balanced", "research"] | None = "balanced"
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    stop: Any | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
