"""
Audrey — user-facing slash commands.

A lightweight preprocessor that lets end-users control tool usage and sources
on a per-turn basis without switching virtual models.

Commands are detected as the first token of the last user message. The token
is stripped from the message before it reaches the classifier. Flags are
recorded on pipeline state and consumed by the web-search gate, the KB
fetcher, and a tool-nudge preamble that steers the ReAct agent toward a
specific tool when the user asked for it.

Natural-language aliases (full-phrase, anchored to the prompt start) are
handled by ``natural_language_flags`` so ``search the web for...`` etc. also
set the same flags.
"""

import re
from typing import Any

# ── Slash commands ───────────────────────────────────────────────────────────
#
# Each command has a canonical form and may have aliases. The ``flags`` dict
# is merged into pipeline state. ``tool_hint`` is an optional string that
# becomes part of a nudge preamble, telling the model which tool to prefer.

SLASH_COMMANDS: dict[str, dict[str, Any]] = {
    # ── Source priority ─────────────────────────────────────────────────
    "/web":      {"flags": {"source_priority": "web_primary", "force_web_search": True}},
    "/kb":       {"flags": {"source_priority": "kb_primary",  "force_kb": True}},
    "/both":     {"flags": {"source_priority": "both",        "force_web_search": True, "force_kb": True}},

    # ── Source disable ──────────────────────────────────────────────────
    "/nosearch": {"flags": {"disable_web_search": True}},
    "/noweb":    {"flags": {"disable_web_search": True}},
    "/nokb":     {"flags": {"disable_kb": True}},

    # ── Tool nudges ─────────────────────────────────────────────────────
    "/remember": {"flags": {}, "tool_hint": "memory_store", "description": "Save a fact to memory"},
    "/recall":   {"flags": {}, "tool_hint": "memory_search", "description": "Search saved memories"},
    "/py":       {"flags": {}, "tool_hint": "run_python", "description": "Run Python code"},
    "/python":   {"flags": {}, "tool_hint": "run_python"},
    "/sql":      {"flags": {}, "tool_hint": "sql_query", "description": "Run a SQL query"},
    "/read":     {"flags": {}, "tool_hint": "read_file", "description": "Read a workspace file"},
    "/fetch":    {"flags": {"force_web_search": False}, "tool_hint": "fetch_url", "description": "Fetch and summarize a URL"},
    "/stats":    {"flags": {}, "tool_hint": "system_stats", "description": "Server health and GPU stats"},
    "/sources":  {"flags": {}, "tool_hint": "list_sources", "description": "List indexed knowledge-base docs"},
}


# ── Natural-language triggers ────────────────────────────────────────────────
#
# Matched against the *start* of the user prompt (after stripping greetings).
# Only applied when the user didn't already use a slash command.

_NL_TRIGGERS: list[tuple[re.Pattern[str], dict[str, Any]]] = [
    # Web primary
    (re.compile(r"^\s*(please\s+)?(search|look (it )?up)\s+(the\s+)?(web|online|internet)\b", re.I),
     {"source_priority": "web_primary", "force_web_search": True}),
    (re.compile(r"^\s*(please\s+)?google\s+", re.I),
     {"source_priority": "web_primary", "force_web_search": True}),
    (re.compile(r"^\s*(please\s+)?check\s+(the\s+)?(web|online|internet)\b", re.I),
     {"source_priority": "web_primary", "force_web_search": True}),

    # KB primary
    (re.compile(r"^\s*(please\s+)?(search|look\s+(in|up)|check)\s+(my\s+)?(notes|docs|knowledge[\s-]?base)\b", re.I),
     {"source_priority": "kb_primary", "force_kb": True}),
    (re.compile(r"^\s*what\s+do\s+my\s+(notes|docs)\s+say\b", re.I),
     {"source_priority": "kb_primary", "force_kb": True}),

    # Both
    (re.compile(r"^\s*(search|check)\s+(everywhere|both\s+(my\s+)?(notes|docs)\s+and\s+(the\s+)?web)\b", re.I),
     {"source_priority": "both", "force_web_search": True, "force_kb": True}),

    # Explicit disables (anywhere in prompt, not just start)
]

_NL_DISABLE_TRIGGERS: list[tuple[re.Pattern[str], dict[str, Any]]] = [
    (re.compile(r"\bdon'?t\s+search\s+the\s+(web|internet|online)\b", re.I),
     {"disable_web_search": True}),
    (re.compile(r"\bskip\s+the\s+web\b", re.I),
     {"disable_web_search": True}),
    (re.compile(r"\bjust\s+(my\s+)?(notes|knowledge[\s-]?base)\b", re.I),
     {"disable_web_search": True, "source_priority": "kb_primary", "force_kb": True}),
    (re.compile(r"\bdon'?t\s+search\s+my\s+(notes|docs|knowledge[\s-]?base)\b", re.I),
     {"disable_kb": True}),
    (re.compile(r"\bjust\s+the\s+web\b", re.I),
     {"disable_kb": True, "source_priority": "web_primary", "force_web_search": True}),
]


# ── Tool-hint preambles ──────────────────────────────────────────────────────
#
# Short system-level nudges appended when the user invoked a tool-specific
# slash command. The goal is to steer the ReAct loop toward the right tool
# without forcing it (the model can still decline if irrelevant).

_TOOL_HINT_PREAMBLES: dict[str, str] = {
    "memory_store":  "The user wants you to save information to memory. Use the memory_store tool.",
    "memory_search": "The user wants you to recall saved memories. Use memory_search or memory_recall.",
    "run_python":    "The user wants you to execute Python. Use the run_python tool, then explain the result.",
    "sql_query":     "The user wants a SQL query run against the local database. Use sql_query (and sql_schema if you need to inspect tables first).",
    "read_file":     "The user wants you to read a workspace file. Use read_file (or read_document for PDFs/DOCX/HTML).",
    "fetch_url":     "The user wants you to fetch a URL. Use the fetch_url tool, then summarize or answer from the content.",
    "system_stats":  "The user wants server health info. Call system_stats and report CPU/memory/GPU.",
    "list_sources":  "The user wants to know what documents are indexed. Call list_sources on the knowledge server.",
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _get_last_user_text_and_set(
    messages: list[dict[str, Any]],
) -> tuple[str, callable]:
    """Return (text, setter) for the last user message's text content.

    Supports both plain-string content and list-of-parts multimodal content.
    The setter mutates messages in place with the new text.
    """
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if m.get("role") != "user":
            continue
        c = m.get("content", "")
        if isinstance(c, list):
            for j, part in enumerate(c):
                if isinstance(part, dict) and part.get("type") == "text":
                    text = part.get("text", "")

                    def _set(new_text: str, _i=i, _j=j):
                        messages[_i]["content"][_j]["text"] = new_text
                    return text, _set
            return "", lambda _new: None
        text = str(c or "")

        def _set_plain(new_text: str, _i=i):
            messages[_i]["content"] = new_text
        return text, _set_plain
    return "", lambda _new: None


def _parse_slash(text: str) -> tuple[str | None, str]:
    """Return (command, remaining_text) if the prompt starts with a known slash
    command, otherwise (None, text)."""
    stripped = text.lstrip()
    if not stripped.startswith("/"):
        return None, text
    # Command = first whitespace-delimited token, lowercased.
    head, _, rest = stripped.partition(" ")
    cmd = head.lower()
    if cmd not in SLASH_COMMANDS:
        return None, text
    # Preserve any leading whitespace the user had.
    leading_ws = text[: len(text) - len(stripped)]
    return cmd, leading_ws + rest.lstrip()


def _merge_flags(target: dict[str, Any], new: dict[str, Any]) -> None:
    """Merge new flags into target, but don't let later triggers clobber
    explicit disables set by earlier ones."""
    for k, v in new.items():
        if k in ("disable_web_search", "disable_kb") and target.get(k):
            continue  # keep the disable
        target[k] = v


# ── Public API ───────────────────────────────────────────────────────────────

def apply_slash_commands(
    messages: list[dict[str, Any]],
) -> dict[str, Any]:
    """Scan the last user message for slash commands and natural-language
    triggers. Mutates the message content to strip recognized slash tokens.
    Returns a dict of flags to be merged into pipeline state.

    Output flags (all optional):
        source_priority: "web_primary" | "kb_primary" | "both"
        force_web_search: bool
        force_kb: bool
        disable_web_search: bool
        disable_kb: bool
        tool_hint: str (one of _TOOL_HINT_PREAMBLES keys)
        tool_hint_preamble: str  (ready-to-prepend system text)
        slash_command: str  (the command that was matched, for logging)
    """
    flags: dict[str, Any] = {}
    text, set_text = _get_last_user_text_and_set(messages)
    if not text:
        return flags

    # 1. Slash command (takes precedence over NL triggers)
    cmd, remaining = _parse_slash(text)
    if cmd is not None:
        spec = SLASH_COMMANDS[cmd]
        _merge_flags(flags, spec.get("flags", {}))
        hint = spec.get("tool_hint")
        if hint:
            flags["tool_hint"] = hint
        flags["slash_command"] = cmd
        set_text(remaining)
        text = remaining  # also feed stripped text to NL matcher

    # 2. Natural-language anchored triggers (prompt-start)
    if "slash_command" not in flags:
        for pattern, trigger_flags in _NL_TRIGGERS:
            if pattern.search(text):
                _merge_flags(flags, trigger_flags)
                break

    # 3. Natural-language disables (anywhere in prompt, always applied)
    for pattern, trigger_flags in _NL_DISABLE_TRIGGERS:
        if pattern.search(text):
            _merge_flags(flags, trigger_flags)

    # 4. Resolve tool-hint / source-priority preamble
    hint = flags.get("tool_hint")
    priority = flags.get("source_priority", "")
    preamble_parts: list[str] = []
    if hint and hint in _TOOL_HINT_PREAMBLES:
        preamble_parts.append(_TOOL_HINT_PREAMBLES[hint])
    if priority == "web_primary":
        preamble_parts.append(
            "The user has set the web as the primary source for this turn. "
            "Call web_search first. Cite URLs in your answer. The knowledge "
            "base is still available as a secondary source if useful."
        )
    elif priority == "kb_primary":
        preamble_parts.append(
            "The user has set the knowledge base as the primary source for "
            "this turn. Use search_knowledge first and cite [N] markers. The "
            "web is still available as a secondary source if the KB is thin."
        )
    elif priority == "both":
        preamble_parts.append(
            "The user wants you to consult both the knowledge base and the "
            "web. Call search_knowledge and web_search, then reconcile the "
            "findings. Cite KB excerpts with [N] and web sources with URLs."
        )
    if preamble_parts:
        flags["tool_hint_preamble"] = "\n\n".join(preamble_parts)

    return flags


def resolve_web_search(
    *,
    force_web_search: bool = False,
    disable_web_search: bool = False,
    classifier_wants_search: bool = False,
) -> bool:
    """Final decision on whether to perform a web search for this turn.

    Precedence: disable > force > classifier.
    """
    if disable_web_search:
        return False
    if force_web_search:
        return True
    return classifier_wants_search


def resolve_kb(
    *,
    force_kb: bool = False,
    disable_kb: bool = False,
    model_is_knowledge: bool = False,
) -> bool:
    """Final decision on whether to fetch KB context for this turn."""
    if disable_kb:
        return False
    if force_kb:
        return True
    return model_is_knowledge
