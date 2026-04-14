"""
OpenAPI Tool Registry — generic discovery and dispatch for any OpenAPI tool server.

At startup, fetches /openapi.json from each configured server URL, converts
every endpoint into an Ollama-compatible tool definition, and dispatches
model tool calls to the correct server via HTTP.

Adding a new tool = spinning up a new OpenAPI server + adding its URL to config.
Zero code changes in Audrey's main codebase.

v2: Added context compression for multi-round tool loops, tool result tracking,
    and support for the ReAct agent pattern.
"""

import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import aiohttp

from config import MAX_TOOL_ROUNDS, REACT_COMPRESS_AFTER, REACT_COMPRESS_MAX_CHARS

logger = logging.getLogger("audrey.registry")

# Re-export under the names used throughout this module
COMPRESS_AFTER_ROUNDS = REACT_COMPRESS_AFTER
COMPRESS_MAX_RESULT_CHARS = REACT_COMPRESS_MAX_CHARS


# ── ToolServer ───────────────────────────────────────────────────────────────


class ToolServer:
    """A single discovered OpenAPI tool server."""

    def __init__(self, name: str, url: str, spec: Dict[str, Any]):
        self.name = name
        self.url = url.rstrip("/")
        self.spec = spec
        self.tools: List[Dict[str, Any]] = []
        self.endpoints: Dict[str, Dict[str, Any]] = {}
        self._parse_spec()

    def _parse_spec(self) -> None:
        for path, methods in self.spec.get("paths", {}).items():
            for method, operation in methods.items():
                if method not in ("get", "post", "put", "patch", "delete"):
                    continue

                op_id = operation.get("operationId")
                if not op_id:
                    op_id = re.sub(r"[^a-zA-Z0-9_]", "_", f"{method}_{path}").strip("_")

                tool_name = f"{self.name}__{op_id}"
                description = (
                    operation.get("summary")
                    or operation.get("description")
                    or op_id
                )

                # ── Build parameters schema ──
                parameters: Dict[str, Any] = {
                    "type": "object",
                    "properties": {},
                }

                # Request body (POST / PUT)
                body_schema = (
                    operation.get("requestBody", {})
                    .get("content", {})
                    .get("application/json", {})
                    .get("schema", {})
                )
                if body_schema:
                    body_schema = self._resolve_ref(body_schema)
                    parameters = {
                        "type": body_schema.get("type", "object"),
                        "properties": body_schema.get("properties", {}),
                    }
                    if body_schema.get("required"):
                        parameters["required"] = body_schema["required"]

                # Query / path parameters
                for param in operation.get("parameters", []):
                    p_name = param.get("name", "")
                    p_schema = param.get("schema", {"type": "string"})
                    parameters.setdefault("properties", {})[p_name] = {
                        "type": p_schema.get("type", "string"),
                        "description": param.get("description", ""),
                    }
                    if param.get("required"):
                        parameters.setdefault("required", []).append(p_name)

                self.tools.append({
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": f"[{self.name}] {description}",
                        "parameters": parameters,
                    },
                })
                self.endpoints[tool_name] = {
                    "method": method,
                    "path": path,
                    "has_body": bool(body_schema),
                }

        logger.info(
            "Parsed %s (%s): %d tools — %s",
            self.name, self.url, len(self.tools),
            ", ".join(e.rsplit("__", 1)[-1] for e in self.endpoints),
        )

    def _resolve_ref(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        ref = schema.get("$ref", "")
        if ref.startswith("#/components/schemas/"):
            resolved = (
                self.spec.get("components", {})
                .get("schemas", {})
                .get(ref.split("/")[-1], {})
            )
            if resolved:
                return resolved
        return schema


# ── Context compression helpers ──────────────────────────────────────────────


def _truncate_tool_result(result_text: str, max_chars: int) -> str:
    """Truncate a long tool result, keeping head and tail for context."""
    if len(result_text) <= max_chars:
        return result_text
    half = max_chars // 2
    return (
        result_text[:half]
        + f"\n\n... [truncated {len(result_text) - max_chars} chars] ...\n\n"
        + result_text[-half:]
    )


def compress_tool_context(
    messages: List[Dict[str, Any]],
    *,
    preserve_last_n: int = 2,
    max_result_chars: int = COMPRESS_MAX_RESULT_CHARS,
) -> List[Dict[str, Any]]:
    """Compress older tool call/result pairs into a summary to save context space.

    Keeps the original system + user messages intact, compresses intermediate
    assistant/tool message pairs into a single summary, and preserves the most
    recent `preserve_last_n` tool exchange rounds.

    This prevents context window exhaustion during multi-round tool use.
    """
    # Split messages into: preamble (system/user), tool_exchanges, recent
    preamble = []
    exchanges = []
    i = 0

    # Collect preamble (everything before first assistant with tool_calls)
    while i < len(messages):
        msg = messages[i]
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            break
        preamble.append(msg)
        i += 1

    # Collect tool exchange groups: each group = [assistant_with_tools, tool_result, ...]
    current_group = []
    while i < len(messages):
        msg = messages[i]
        if msg.get("role") == "assistant" and msg.get("tool_calls") and current_group:
            exchanges.append(current_group)
            current_group = [msg]
        else:
            current_group.append(msg)
        i += 1
    if current_group:
        exchanges.append(current_group)

    # If few enough exchanges, just truncate large tool results
    if len(exchanges) <= preserve_last_n:
        compressed = list(preamble)
        for group in exchanges:
            for msg in group:
                if msg.get("role") == "tool":
                    compressed.append({
                        "role": "tool",
                        "content": _truncate_tool_result(
                            msg.get("content", ""), max_result_chars
                        ),
                    })
                else:
                    compressed.append(msg)
        return compressed

    # Summarize older exchanges into a compact system message
    old_exchanges = exchanges[:-preserve_last_n]
    recent_exchanges = exchanges[-preserve_last_n:]

    summary_parts = []
    for group in old_exchanges:
        for msg in group:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                calls = msg["tool_calls"]
                for tc in calls:
                    func = tc.get("function", {})
                    name = func.get("name", "unknown")
                    args = func.get("arguments", {})
                    if isinstance(args, dict):
                        args_str = json.dumps(args, default=str)[:100]
                    else:
                        args_str = str(args)[:100]
                    summary_parts.append(f"  Called: {name}({args_str})")
            elif msg.get("role") == "tool":
                content = msg.get("content", "")
                # Extract just the key info from tool results
                try:
                    parsed = json.loads(content)
                    if isinstance(parsed, dict):
                        if "error" in parsed:
                            summary_parts.append(f"  Result: ERROR — {parsed['error'][:100]}")
                        elif "results" in parsed:
                            summary_parts.append(
                                f"  Result: {len(parsed['results'])} results returned"
                            )
                        else:
                            # Compact representation
                            keys = list(parsed.keys())[:5]
                            summary_parts.append(f"  Result: {{{', '.join(keys)}...}}")
                    else:
                        summary_parts.append(f"  Result: {str(parsed)[:150]}")
                except (json.JSONDecodeError, TypeError):
                    summary_parts.append(f"  Result: {content[:150]}")

    summary_msg = {
        "role": "system",
        "content": (
            "Previous tool interactions (summarized):\n"
            + "\n".join(summary_parts)
            + "\n\nThe detailed results from these calls have already been processed."
        ),
    }

    # Reconstruct: preamble + summary + recent exchanges (with truncated results)
    result = list(preamble) + [summary_msg]
    for group in recent_exchanges:
        for msg in group:
            if msg.get("role") == "tool":
                result.append({
                    "role": "tool",
                    "content": _truncate_tool_result(
                        msg.get("content", ""), max_result_chars
                    ),
                })
            else:
                result.append(msg)

    logger.info(
        "Compressed context: %d msgs → %d msgs (%d old exchanges summarized)",
        len(messages), len(result), len(old_exchanges),
    )
    return result


# ── ToolRegistry ─────────────────────────────────────────────────────────────


class ToolRegistry:
    """Discovers and manages multiple OpenAPI tool servers."""

    def __init__(self, session: Optional[aiohttp.ClientSession] = None) -> None:
        self.servers: Dict[str, ToolServer] = {}
        self._session: Optional[aiohttp.ClientSession] = session
        self._all_tools: List[Dict[str, Any]] = []
        self._server_configs: List[Dict[str, Any]] = []

    def set_session(self, session: aiohttp.ClientSession) -> None:
        self._session = session

    # ── Properties expected by main.py ───────────────────────────────────

    @property
    def tool_definitions(self) -> List[Dict[str, Any]]:
        return self._all_tools

    @property
    def has_tools(self) -> bool:
        return len(self._all_tools) > 0

    @property
    def tool_count(self) -> int:
        return len(self._all_tools)

    @property
    def tool_names(self) -> List[str]:
        return [t["function"]["name"] for t in self._all_tools]

    @property
    def server_info(self) -> Dict[str, Any]:
        return {
            name: {"url": s.url, "tools": len(s.tools)}
            for name, s in self.servers.items()
        }

    def status(self) -> Dict[str, Any]:
        return {
            "servers": self.server_info,
            "total_tools": self.tool_count,
        }

    # ── Discovery ────────────────────────────────────────────────────────

    @staticmethod
    def _normalize_configs(
        raw: List[Union[str, Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Accept both plain URL strings and {url, name} dicts from config."""
        out = []
        for item in raw:
            if isinstance(item, str):
                out.append({"url": item, "name": ""})
            elif isinstance(item, dict):
                out.append(item)
            else:
                logger.warning("Skipping unrecognized tool_server entry: %r", item)
        return out

    async def discover(
        self, server_configs: List[Union[str, Dict[str, Any]]]
    ) -> None:
        normalized = self._normalize_configs(server_configs)
        self._server_configs = normalized
        self.servers.clear()
        self._all_tools.clear()

        for cfg in normalized:
            url = cfg.get("url", "").rstrip("/")
            name = cfg.get("name", "")
            if not url:
                logger.warning("Tool server config missing url: %r", cfg)
                continue
            if not name:
                name = re.sub(
                    r"[^a-zA-Z0-9]", "_",
                    url.split("//")[-1].split(":")[0],
                ).strip("_") or "server"

            try:
                spec = await self._fetch_spec(url)
                if spec:
                    server = ToolServer(name, url, spec)
                    if server.tools:
                        self.servers[name] = server
                        self._all_tools.extend(server.tools)
                else:
                    logger.warning("No OpenAPI spec at %s — skipped", url)
            except Exception as exc:
                logger.warning("Discovery failed for %s (%s): %s", name, url, exc)

        logger.info(
            "Registry: %d server(s), %d tool(s)",
            len(self.servers), len(self._all_tools),
        )

    async def _fetch_spec(self, url: str) -> Optional[Dict[str, Any]]:
        assert self._session, "HTTP session not set — call set_session() first"
        for path in ("/openapi.json", "/openapi.yaml", "/.well-known/openapi.json"):
            try:
                async with self._session.get(
                    f"{url}{path}", timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status != 200:
                        continue
                    ct = resp.content_type or ""
                    if "yaml" in ct or path.endswith(".yaml"):
                        import yaml as _yaml
                        return _yaml.safe_load(await resp.text())
                    return await resp.json()
            except Exception:
                continue
        return None

    async def rediscover(
        self,
        server_configs: Optional[List[Union[str, Dict[str, Any]]]] = None,
    ) -> None:
        """Re-discover tools. Uses stored configs if none provided."""
        configs = server_configs if server_configs is not None else [
            {"url": s.url, "name": n} for n, s in self.servers.items()
        ]
        if not configs and self._server_configs:
            configs = self._server_configs
        if configs:
            await self.discover(configs)

    # ── Dispatch ─────────────────────────────────────────────────────────

    async def execute(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        for server in self.servers.values():
            if tool_name in server.endpoints:
                return await self._call(
                    server, server.endpoints[tool_name], dict(arguments),
                )
        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    async def _call(
        self,
        server: ToolServer,
        endpoint: Dict[str, Any],
        arguments: Dict[str, Any],
    ) -> str:
        assert self._session
        method = endpoint["method"]
        url = f"{server.url}{endpoint['path']}"

        # Path parameters: /files/{path} → /files/readme.txt
        for key in list(arguments):
            ph = f"{{{key}}}"
            if ph in url:
                url = url.replace(ph, str(arguments.pop(key)))

        try:
            kw: Dict[str, Any] = {"timeout": aiohttp.ClientTimeout(total=30)}
            if method in ("post", "put", "patch"):
                kw["json"] = arguments
            elif arguments:
                kw["params"] = {k: str(v) for k, v in arguments.items()}

            async with self._session.request(method, url, **kw) as resp:
                text = await resp.text()
                if resp.status >= 400:
                    logger.warning(
                        "Tool HTTP %d: %s %s", resp.status, method.upper(), url,
                    )
                    return json.dumps({
                        "error": f"HTTP {resp.status}",
                        "detail": text[:500],
                    })
                return text
        except Exception as exc:
            logger.exception("Tool dispatch failed: %s %s", method.upper(), url)
            return json.dumps({"error": f"Dispatch failed: {exc}"})

    # ── Tool-Calling Loop (with context compression) ─────────────────────

    async def run_with_tools(
        self,
        chat_fn: Callable,
        messages: List[Dict[str, Any]],
        *,
        compress_enabled: bool = True,
    ) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Run a model with tool calling.  Loops until the model responds with
        plain text or MAX_TOOL_ROUNDS is reached.

        Returns: (content, final_messages, tool_calls_log)
            - content: the model's final text response
            - final_messages: the full conversation including tool exchanges
            - tool_calls_log: list of dicts tracking each tool invocation
              for observability (round, tool name, args preview, result length)

        Improvement: compresses context after COMPRESS_AFTER_ROUNDS to prevent
        context window exhaustion during multi-round tool use.

        chat_fn:  async (messages) → Ollama response dict.
                  The caller must include tools=registry.tool_definitions in
                  the payload built by chat_fn.
        """
        current = list(messages)
        tool_calls_log: List[Dict[str, Any]] = []  # Track for observability

        for round_num in range(MAX_TOOL_ROUNDS):
            # Compress context if we've done enough rounds
            if (
                compress_enabled
                and round_num > 0
                and round_num % COMPRESS_AFTER_ROUNDS == 0
            ):
                current = compress_tool_context(current)

            data = await chat_fn(current)
            msg = data["message"]
            tool_calls = msg.get("tool_calls")

            if not tool_calls:
                content = msg.get("content", "")
                if not content or not content.strip():
                    raise RuntimeError("Model returned empty content")
                return content, current, tool_calls_log

            logger.info(
                "Tool round %d: %d call(s)", round_num + 1, len(tool_calls),
            )

            current.append({
                "role": "assistant",
                "content": msg.get("content", ""),
                "tool_calls": tool_calls,
            })

            for tc in tool_calls:
                func = tc.get("function", {})
                name = func.get("name", "")
                args = func.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}

                logger.info(
                    "  → %s(%s)", name, json.dumps(args, default=str)[:200],
                )
                result = await self.execute(name, args)
                current.append({"role": "tool", "content": result})

                # Track for observability
                tool_calls_log.append({
                    "round": round_num + 1,
                    "tool": name,
                    "args_preview": json.dumps(args, default=str)[:100],
                    "result_len": len(result),
                })

        logger.warning("Max tool rounds (%d) reached", MAX_TOOL_ROUNDS)
        # Final compression before the last attempt
        if compress_enabled:
            current = compress_tool_context(current)
        data = await chat_fn(current)
        return (
            data["message"].get("content", "") or "[Tool rounds exhausted]",
            current,
            tool_calls_log,
        )
