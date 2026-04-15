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
from typing import Any, Dict, List, Optional


# --------------------------------------------------------------------------- #
# Meta-tool definitions (OpenAI function-calling format)
# --------------------------------------------------------------------------- #

META_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_tools",
            "description": (
                "Search available tools by keyword. Returns matching tool names "
                "and descriptions. Use this to find tools relevant to your task."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Keyword to search for in tool names and descriptions",
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
                "Get the full parameter schema for a specific tool. "
                "Call this before using a tool to see its exact parameters."
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
                "List all tools for a specific service (e.g., 'ramp', 'outlook', 'hubspot'). "
                "Returns tool names and one-line descriptions."
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


def _extract_service_prefix(tool_name: str) -> str:
    """Extract the service prefix from a tool name.

    Convention: service_toolname or service-toolname.
    E.g. 'ramp_execute_query' -> 'ramp', 'hubspot_relay-create-company' -> 'hubspot'.

    For single-service envs where tools have no prefix (e.g. 'addIssueComment',
    'getVaultSpaces'), returns None to signal no meaningful grouping.
    """
    # Try underscore first (most common: ramp_execute_query, outlook_send_email)
    if "_" in tool_name:
        prefix = tool_name.split("_")[0]
        # Reject verb-like prefixes that aren't real service names
        # (e.g., 'list_emails' -> 'list', 'get_post_content' -> 'get')
        verbs = {"get", "set", "list", "create", "update", "delete", "search",
                 "add", "remove", "send", "view", "move", "copy", "find", "check",
                 "cancel", "accept", "decline", "forward", "reply", "archive",
                 "mark", "load", "export", "import", "submit", "close", "open"}
        if prefix.lower() not in verbs:
            return prefix
    return None


def _extract_description(tool: Dict[str, Any]) -> str:
    """Extract a clean one-line description from a tool definition."""
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


class ToolIndex:
    """Indexes tool schemas for on-demand retrieval.

    Stores full schemas in memory (not context), provides fast lookup
    by name and keyword search, and generates compact summaries.
    """

    def __init__(self, tools: List[Dict[str, Any]]):
        """Build index from raw tool definitions.

        Args:
            tools: List of tool dicts in OpenAI function-calling format.
        """
        self._tools_by_name: Dict[str, Dict[str, Any]] = {}
        self._services: Dict[str, List[str]] = defaultdict(list)
        self._descriptions: Dict[str, str] = {}
        self._ungrouped: List[str] = []

        for tool in tools:
            func = tool.get("function", tool)
            name = func.get("name", "")
            if not name:
                continue

            self._tools_by_name[name] = tool
            self._descriptions[name] = _extract_description(tool)

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

    def search(self, query: str, limit: int = 15) -> List[Dict[str, str]]:
        """Search tools by keyword in name and description.

        Matches the full query as a substring, and also matches individual
        words from the query for partial/fuzzy matching.

        Returns list of {name, description} dicts, sorted by relevance.
        """
        query_lower = query.lower()
        # Split query into words for partial matching
        query_words = [w for w in re.split(r"[\s_\-]+", query_lower) if len(w) >= 2]
        results = []

        for name, desc in self._descriptions.items():
            name_lower = name.lower()
            desc_lower = desc.lower()
            score = 0

            # Exact substring match (strongest signal)
            if query_lower in name_lower:
                score += 4
            if query_lower in desc_lower:
                score += 2

            # Per-word matching (weaker but catches partial names)
            if score == 0 and query_words:
                word_hits = sum(
                    1 for w in query_words
                    if w in name_lower or w in desc_lower
                )
                if word_hits > 0:
                    score = word_hits

            if score > 0:
                results.append((score, name, desc))

        results.sort(key=lambda x: (-x[0], x[1]))
        return [{"name": r[1], "description": r[2]} for r in results[:limit]]

    def list_service(self, service: str) -> List[Dict[str, str]]:
        """List all tools for a service prefix.

        Returns list of {name, description} dicts.
        """
        # Try exact match first, then case-insensitive
        tool_names = self._services.get(service)
        if not tool_names:
            service_lower = service.lower()
            for svc, names in self._services.items():
                if svc.lower() == service_lower:
                    tool_names = names
                    break

        if not tool_names:
            return []

        return [
            {"name": n, "description": self._descriptions.get(n, "")}
            for n in sorted(tool_names)
        ]

    def build_summary(self, tools_per_service: int = 5) -> str:
        """Build a compact summary of all tools for the system prompt.

        Groups tools by service, shows top N per service with descriptions,
        and indicates how many more are available.

        Args:
            tools_per_service: Number of tools to show per service in the summary.

        Returns:
            Compact summary string (~500-800 tokens for 192 tools).
        """
        lines = [
            f"{self.tool_count} tools across {len(self._services)} services. "
            f"Use search_tools(query) to find tools or get_tool_schema(name) "
            f"to see full parameters before calling a tool.\n"
        ]

        for service in sorted(
            self._services.keys(),
            key=lambda s: -len(self._services[s]),
        ):
            tool_names = sorted(self._services[service])
            count = len(tool_names)

            # Show the first N tools with descriptions
            shown = tool_names[:tools_per_service]
            lines.append(f"### {service} ({count} tools)")
            for name in shown:
                desc = self._descriptions.get(name, "")
                lines.append(f"  {name} — {desc}")
            remaining = count - len(shown)
            if remaining > 0:
                lines.append(f"  ... {remaining} more (use list_service_tools)")
            lines.append("")

        return "\n".join(lines)


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
