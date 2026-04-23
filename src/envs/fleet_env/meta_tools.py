"""
Meta-tools for on-demand tool schema retrieval.

Instead of dumping all tool schemas into the system prompt (which can consume
50K+ tokens for envs like budget with 192 tools), this module provides:

1. A compact tool index/summary for the system prompt (~500-800 tokens)
2. Meta-tools that the agent calls to discover and retrieve schemas on demand

The architecture mirrors context_manager.py: pure Python, in-memory per episode,
locally intercepted in the training harness (no MCP call).

Usage in SkyRL-Fleet env.py:
    from envs.fleet_env import MetaToolHandler, ToolIndex

    # At reset (replace full schema dump):
    index = ToolIndex(tools_list)
    handler = MetaToolHandler(index)
    summary = index.build_summary()
    meta_schemas = handler.get_tool_schemas()

    # At step (intercept meta-tool calls):
    if handler.is_meta_tool(tool_call["name"]):
        result = handler.execute(tool_call["name"], tool_call["arguments"])
"""

import json
import re
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from .meta_tools_summary import SummaryRecord


# --------------------------------------------------------------------------- #
# Meta-tool definitions (OpenAI function-calling format)
# --------------------------------------------------------------------------- #

META_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_tools",
            "description": (
                "Fallback only. Every tool's name, parameter signature, purpose, "
                "and return shape is already in the system-prompt summary, so "
                "call tools directly. Use this only when the summary line for "
                "a tool is ambiguous or you are looking for an alternate by "
                "keyword. Returns the matching tools' full summary records."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Keyword to match against tool names, use descriptions, and return shapes",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_tool_schema",
            "description": (
                "Fallback only. Returns the full JSON schema for one tool: "
                "nested object sub-properties, regex patterns, enum values "
                "that were dropped from the inline summary, and full per-"
                "parameter descriptions. Use only when the inline summary's "
                "parameter signature is not enough to compose a correct call."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Exact tool name to retrieve schema for",
                    }
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_service_tools",
            "description": (
                "Fallback only. Returns compact cards for every tool in a "
                "service bucket (e.g. 'emails', 'calendar', 'ramp'). Use only "
                "when narrowing from a general task area and you want to see "
                "all tools in one service without scanning the full summary."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "service": {
                        "type": "string",
                        "description": "Service prefix to list tools for",
                    }
                },
                "required": ["service"],
            },
        },
    },
]

META_TOOL_NAMES = {t["function"]["name"] for t in META_TOOLS}


# --------------------------------------------------------------------------- #
# ToolIndex — indexes tool schemas and builds compact summaries
# --------------------------------------------------------------------------- #


_VERB_PREFIXES = {
    "get", "set", "list", "create", "update", "delete", "search",
    "add", "remove", "send", "view", "move", "copy", "find", "check",
    "cancel", "accept", "decline", "forward", "reply", "archive",
    "mark", "load", "export", "import", "submit", "close", "open",
    "flag", "save", "unsave", "follow", "unfollow", "subscribe",
    "unsubscribe", "edit", "vote", "post", "fetch", "read", "write",
    "insert", "upsert", "draft", "download", "upload", "start", "stop",
    "pause", "resume", "reset", "configure",
}


def _extract_service_prefix(tool_name: str) -> str:
    """Extract the service prefix from a tool name.

    Handles two naming conventions:
    1. Real service prefix: 'ramp_execute_query' -> 'ramp',
       'hubspot_relay-create-company' -> 'hubspot'.
    2. Verb_domain convention: 'list_emails' -> 'emails',
       'get_email_attachment' -> 'email', 'create_calendar_event' -> 'calendar'.

    For single-service envs where tools have no separator, returns None.
    """
    # Split on underscore or hyphen
    parts = re.split(r"[_\-]", tool_name)
    if len(parts) < 2:
        return None

    first = parts[0].lower()
    if first not in _VERB_PREFIXES:
        # First token is a real service prefix (e.g., 'ramp', 'hubspot')
        return parts[0]

    # First token is a verb; use second token as domain (e.g. list_emails -> emails).
    return parts[1].lower()


def _extract_description(tool: Dict[str, Any]) -> str:
    """Extract a clean one-line description for compact summaries."""
    func = tool.get("function", tool)
    desc = func.get("description", "")
    # Strip the [Service (Alias)] prefix that Fleet tools use
    desc = re.sub(r"^\[.*?\]\s*", "", desc)
    # Take first line only (many Fleet descriptions have multi-line details)
    desc = desc.split("\n")[0].strip()
    # Take first sentence only
    if ". " in desc:
        desc = desc[: desc.index(". ") + 1]
    # Cap length
    if len(desc) > 120:
        desc = desc[:117] + "..."
    return desc


def _extract_full_description(tool: Dict[str, Any]) -> str:
    """Extract the full description for search results.

    Preserves behavioral details that distinguish similar tools.
    Capped at 300 chars to prevent search results from becoming too large.
    """
    func = tool.get("function", tool)
    desc = func.get("description", "")
    desc = re.sub(r"^\[.*?\]\s*", "", desc)
    desc = desc.strip()
    if len(desc) > 300:
        desc = desc[:297] + "..."
    return desc


_ENUM_INLINE_MAX_VALUES = 6
_ENUM_INLINE_MAX_CHARS = 80


def _format_enum(values: List[Any]) -> Optional[str]:
    """Render an enum list as 'a'|'b'|'c' if small enough to inline, else None.

    Prevents pathological cases like 50-value enums from blowing up the summary.
    """
    if not values or len(values) > _ENUM_INLINE_MAX_VALUES:
        return None
    rendered = "|".join(
        f"'{v}'" if isinstance(v, str) else str(v) for v in values
    )
    if len(rendered) > _ENUM_INLINE_MAX_CHARS:
        return None
    return rendered


def _extract_param_signature(tool: Dict[str, Any]) -> str:
    """Extract a compact parameter signature from a tool schema.

    Returns a string like "folder?: 'inbox'|'sent'|'drafts', limit?: integer"
    so the model can call the tool directly without fetching the full schema.
    Small enums are inlined because the 9B otherwise hallucinates values
    (e.g. calling search with type='post' when the allowed set is 'link' |
    'comment' | 'sr' | 'user') and burns turns on correction.
    """
    func = tool.get("function", tool)
    params = func.get("parameters", {})
    props = params.get("properties", {})
    required = set(params.get("required", []))

    if not props:
        return ""

    parts = []
    for name, spec in props.items():
        suffix = "" if name in required else "?"
        enum_repr = _format_enum(spec.get("enum") or [])
        if enum_repr:
            parts.append(f"{name}{suffix}: {enum_repr}")
        else:
            ptype = spec.get("type", "any")
            parts.append(f"{name}{suffix}: {ptype}")
    return ", ".join(parts)


class ToolIndex:
    """Indexes tool schemas for on-demand retrieval.

    Stores full schemas in memory (not context), provides fast lookup
    by name and keyword search, and generates compact summaries.
    """

    def __init__(
        self,
        tools: List[Dict[str, Any]],
        summary_records: Optional[Dict[str, "SummaryRecord"]] = None,
    ):
        """Build index from raw tool definitions.

        Args:
            tools: Tool dicts in OpenAI function-calling format.
            summary_records: Optional LLM-generated records keyed by tool name.
                When present, the matching tool renders with the richer
                three-line form in the system prompt and search returns the
                full record. Tools without a record fall back to the parser
                line. See :mod:`meta_tools_summary`.
        """
        self._tools_by_name: Dict[str, Dict[str, Any]] = {}
        self._services: Dict[str, List[str]] = defaultdict(list)
        self._descriptions: Dict[str, str] = {}  # one-liner for summaries
        self._full_descriptions: Dict[str, str] = {}  # full text for search results
        self._param_signatures: Dict[str, str] = {}  # compact param signatures
        self._ungrouped: List[str] = []
        self._summary_records: Dict[str, "SummaryRecord"] = dict(summary_records or {})

        for tool in tools:
            func = tool.get("function", tool)
            name = func.get("name", "")
            if not name:
                continue

            self._tools_by_name[name] = tool
            self._descriptions[name] = _extract_description(tool)
            self._full_descriptions[name] = _extract_full_description(tool)
            self._param_signatures[name] = _extract_param_signature(tool)

            service = _extract_service_prefix(name)
            if service is not None:
                self._services[service].append(name)
            else:
                self._ungrouped.append(name)

        # If most tools are ungrouped, put them all in one bucket
        if len(self._ungrouped) > len(self._tools_by_name) * 0.5:
            self._services["tools"] = (
                self._ungrouped
                + [n for names in self._services.values() for n in names]
            )
            self._services = defaultdict(list, {"tools": self._services["tools"]})
            self._ungrouped = []
        elif self._ungrouped:
            # A few ungrouped tools alongside real services — add as "other"
            self._services["other"] = self._ungrouped

    @property
    def tool_count(self) -> int:
        return len(self._tools_by_name)

    @property
    def service_names(self) -> List[str]:
        return sorted(self._services.keys())

    def get_schema(self, name: str) -> Optional[Dict[str, Any]]:
        """Get the full schema for a tool by exact name."""
        return self._tools_by_name.get(name)

    def search(self, query: str, limit: int = 25) -> List[Dict[str, str]]:
        """Search tools by keyword and return summary records.

        When LLM summary records are available for a tool, the search matches
        against name + use + returns fields and the returned entry includes
        those fields so the caller can act without a follow-up schema fetch.
        Tools without a summary record fall back to matching name + full
        description and return name/description/params.
        """
        query_lower = query.lower()
        query_words = [w for w in re.split(r"[\s_\-]+", query_lower) if len(w) >= 2]
        scored: List[tuple] = []

        for name in self._tools_by_name:
            rec = self._summary_records.get(name)
            if rec is not None:
                haystacks = [name.lower(), rec.use.lower(), rec.returns.lower()]
            else:
                haystacks = [name.lower(), self._full_descriptions.get(name, "").lower()]
            score = 0
            for hay in haystacks:
                if query_lower and query_lower in hay:
                    score += 4 if hay is haystacks[0] else 2
            if score == 0 and query_words:
                score = sum(1 for w in query_words if any(w in h for h in haystacks))
            if score > 0:
                scored.append((score, name))

        scored.sort(key=lambda x: (-x[0], x[1]))
        out: List[Dict[str, str]] = []
        for _, name in scored[:limit]:
            rec = self._summary_records.get(name)
            if rec is not None:
                out.append({
                    "name": name,
                    "params_signature": rec.params_signature,
                    "use": rec.use,
                    "returns": rec.returns,
                })
            else:
                out.append({
                    "name": name,
                    "description": self._full_descriptions.get(name, ""),
                    "params": self._param_signatures.get(name, ""),
                })
        return out

    def list_service(self, service: str) -> List[Dict[str, str]]:
        """List all tools in a service bucket as compact cards.

        When a tool has an LLM summary record, its card includes
        ``{name, params_signature, use}``. Otherwise it falls back to
        ``{name, description, params}`` from the parser.
        """
        tool_names = self._services.get(service)
        if not tool_names:
            service_lower = service.lower()
            for svc, names in self._services.items():
                if svc.lower() == service_lower:
                    tool_names = names
                    break
        if not tool_names:
            return []

        cards: List[Dict[str, str]] = []
        for n in sorted(tool_names):
            rec = self._summary_records.get(n)
            if rec is not None:
                cards.append({
                    "name": n,
                    "params_signature": rec.params_signature,
                    "use": rec.use,
                })
            else:
                cards.append({
                    "name": n,
                    "description": self._descriptions.get(n, ""),
                    "params": self._param_signatures.get(n, ""),
                })
        return cards

    def build_summary(self) -> str:
        """Build the per-tool summary inserted into the system prompt.

        Every tool is rendered. Tools with an LLM summary record render as
        three lines (``name(params)`` / ``use:`` / ``out:``). Tools without
        a record fall back to the parser's single-line form. Services are
        listed as level-3 headers when the env has more than one service.
        """
        llm_count = len(self._summary_records)
        head_note = (
            f"{self.tool_count} tools available. Call any tool directly using "
            f"the parameter signature shown."
        )
        if llm_count < self.tool_count:
            head_note += (
                f" ({llm_count} shown with extended summary, "
                f"{self.tool_count - llm_count} with parser fallback.)"
            )
        lines = [head_note + "\n"]

        service_keys = sorted(
            self._services.keys(),
            key=lambda s: (-len(self._services[s]), s),
        )
        show_service_headers = len(service_keys) > 1
        for service in service_keys:
            tool_names = sorted(self._services[service])
            if show_service_headers:
                lines.append(f"### {service} ({len(tool_names)})")
            for name in tool_names:
                lines.append(self._render_tool(name))
            lines.append("")
        return "\n".join(lines)

    def _render_tool(self, name: str) -> str:
        """Render one tool as it appears in ``build_summary``'s body."""
        rec = self._summary_records.get(name)
        if rec is not None:
            from .meta_tools_summary import render_summary_line
            return render_summary_line(rec)
        desc = self._descriptions.get(name, "")
        params = self._param_signatures.get(name, "")
        if params and desc:
            return f"- {name}({params}) — {desc}"
        if params:
            return f"- {name}({params})"
        if desc:
            return f"- {name}() — {desc}"
        return f"- {name}()"


# --------------------------------------------------------------------------- #
# MetaToolHandler — executes meta-tool calls
# --------------------------------------------------------------------------- #


class MetaToolHandler:
    """Handles meta-tool calls for on-demand schema retrieval.

    Mirrors the ContextManager pattern: check is_meta_tool(), then execute().
    """

    def __init__(self, index: ToolIndex):
        self._index = index

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Get meta-tool definitions to include in the tool list."""
        return [t.copy() for t in META_TOOLS]

    def is_meta_tool(self, tool_name: str) -> bool:
        """Check if a tool name is a meta-tool."""
        return tool_name in META_TOOL_NAMES

    def execute(self, tool_name: str, args: Dict[str, Any]) -> str:
        """Execute a meta-tool and return the result string.

        Args:
            tool_name: Name of the meta-tool.
            args: Tool arguments.

        Returns:
            Result string (JSON for structured results, plain text for errors).
        """
        if tool_name == "search_tools":
            return self._search_tools(args)
        elif tool_name == "get_tool_schema":
            return self._get_tool_schema(args)
        elif tool_name == "list_service_tools":
            return self._list_service_tools(args)
        else:
            return json.dumps({"error": f"Unknown meta-tool: {tool_name}"})

    def _search_tools(self, args: Dict[str, Any]) -> str:
        query = args.get("query", "")
        if not query:
            return json.dumps({"error": "query is required"})

        results = self._index.search(query)
        if not results:
            return json.dumps({"results": [], "message": f"No tools matching '{query}'"})
        return json.dumps({"results": results})

    def _get_tool_schema(self, args: Dict[str, Any]) -> str:
        name = args.get("name", "")
        if not name:
            return json.dumps({"error": "name is required"})

        schema = self._index.get_schema(name)
        if not schema:
            # Suggest close matches
            suggestions = self._index.search(name, limit=3)
            return json.dumps({
                "error": f"Tool '{name}' not found",
                "suggestions": [s["name"] for s in suggestions],
            })

        # Return the full schema
        func = schema.get("function", schema)
        return json.dumps(func, indent=2)

    def _list_service_tools(self, args: Dict[str, Any]) -> str:
        service = args.get("service", "")
        if not service:
            # Return list of available services
            services = {
                svc: len(names)
                for svc, names in self._index._services.items()
            }
            return json.dumps({"services": services})

        tools = self._index.list_service(service)
        if not tools:
            return json.dumps({
                "error": f"Service '{service}' not found",
                "available_services": self._index.service_names,
            })
        return json.dumps({"service": service, "tools": tools})
