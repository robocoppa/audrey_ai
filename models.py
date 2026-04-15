"""
Audrey — request/response models and pipeline state.
"""

from typing import Any, Literal, TypedDict

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


# ── LangGraph pipeline state ─────────────────────────────────────────────────

class AudreyState(TypedDict, total=False):
    request_id: str
    requested_model: str
    messages: list[dict[str, Any]]
    original_messages: list[dict[str, Any]]
    stream: bool
    temperature: float
    max_tokens: int | None
    top_p: float | None
    stop: Any | None
    frequency_penalty: float | None
    presence_penalty: float | None
    task_type: str
    confidence: float
    needs_vision: bool
    route_reason: str
    selected_model: str
    fallback_models: list[str]
    fallback_synthesizer: str
    result_text: str
    errors: list[str]
    started_at: float
    latency_ms: int
    deep_workers: list[str]
    worker_outputs: list[dict[str, str]]
    worker_error_count: int
    synthesizer: str
    synthesis_candidates: list[str]
    synthesis_strategy: str
    synthesis_escalation_reason: str
    synthesis_messages: list[dict[str, Any]]
    prompt_tokens: int
    completion_tokens: int
    search_performed: bool
    search_query: str
    search_results: list[dict[str, str]]
    # Fast-path fields
    use_fast_path: bool
    fast_model: str
    # Agentic fields
    sub_tasks: list[str] | None
    react_rounds: int
    reflection_result: dict[str, Any]
    reflection_retries: int
    force_strong_synth: bool
    escalated: bool
    tools_used: list[dict[str, Any]]
    # UX metadata fields
    audrey_mode: str
    timeline: list[dict[str, Any]]
    cache_hit: bool
    needs_fresh_data: bool
    fast_path_confidence: float | None
    force_deep_profile: bool
    planning_enabled_override: bool | None
    planning_min_tokens_override: int | None
    reflection_enabled_override: bool | None
    reflection_max_retries_override: int | None
    react_max_rounds_override: int | None
