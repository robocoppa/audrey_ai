"""
Audrey — request/response models and pipeline state.
"""

from typing import Any, Dict, List, Literal, Optional, TypedDict

from pydantic import BaseModel


# ── OpenAI-compatible request models ─────────────────────────────────────────

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


# ── LangGraph pipeline state ─────────────────────────────────────────────────

class AudreyState(TypedDict, total=False):
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
