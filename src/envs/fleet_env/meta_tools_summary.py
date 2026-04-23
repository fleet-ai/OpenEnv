"""LLM-generated summaries for meta-tools.

Parser-generated summaries (in ``meta_tools.py``) capture names, param types,
and short descriptions. This module generates richer per-tool summaries via
Opus 4.7 that include ID-prefix conventions, default values, enum semantics,
and return shapes. The :class:`ToolIndex` picks up these records when present
and renders them instead of the parser's single-line form.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You summarize tool schemas for an agent's system prompt. The agent is a 9B language model that calls tools by name using JSON arguments. Your summary is inserted into a compact listing where the agent sees every tool up front without discovery round-trips.

For each tool you receive its JSON schema. Emit a strict JSON record with three fields:

  params_signature : compact parameter signature
  use              : one-sentence purpose, with any non-obvious conventions
  returns          : brief description of the return shape

Rules for params_signature:
  - Format each parameter as  name: type  or  name?: type  for optional.
  - If the parameter has a default value in the schema, append  =<default>.
  - Inline enums as  name: 'a'|'b'|'c'  if the enum has 6 or fewer values
    AND the rendered literal list is 80 characters or fewer.
  - For enums that exceed those limits, fall back to  name: string.
  - Preserve the order from the schema's properties dict.
  - Use lowercase type names: string, integer, number, boolean, object, array.

Rules for use:
  - Maximum 200 characters.
  - One sentence, imperative present tense ("Submit a comment", not "Submits").
  - Must mention any of the following that apply and are not obvious from types:
      * ID-prefix conventions (e.g. 't3_' for posts, 't1_' for comments)
      * Format requirements (markdown, ISO-8601 date, UUID, etc.)
      * Pagination or limit defaults (e.g. "returns up to 100 results")
      * Authentication scope (if the description mentions it)
  - Do not restate the parameter list; that is already in params_signature.
  - Do not speculate beyond the schema and description. If the schema does
    not state a convention, do not invent one.

Rules for returns:
  - Maximum 120 characters.
  - Name the 2-4 most important fields of the response object.
  - If the description gives no hint about the return shape, write "see tool response"
    rather than inventing one.
  - If the tool mutates state and returns only status, write "success/error status"

Output format: valid JSON, no prose before or after, no markdown fences.

  {
    "params_signature": "...",
    "use": "...",
    "returns": "..."
  }"""


DEFAULT_MODEL = "claude-opus-4-7"
DEFAULT_CONCURRENCY = 8
MAX_TOKENS = 512


@dataclass
class SummaryRecord:
    """One tool's LLM-generated summary."""

    name: str
    params_signature: str
    use: str
    returns: str
    schema_sha: str

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dict form."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SummaryRecord":
        """Construct from a dict loaded from JSON."""
        return cls(**d)


def schema_sha(tool_schema: Dict[str, Any]) -> str:
    """Stable hash over the tool's JSON schema, truncated for cache keys."""
    canonical = json.dumps(tool_schema, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _tool_name(tool_schema: Dict[str, Any]) -> str:
    """Extract the tool name from a schema dict (handles nested 'function' wrapper)."""
    return tool_schema.get("function", tool_schema).get("name", "")


def _parse_response(text: str, name: str, sha: str) -> Optional[SummaryRecord]:
    """Parse the model's JSON output into a :class:`SummaryRecord`. Returns None on malformed output."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        rec = json.loads(text)
        return SummaryRecord(
            name=name,
            params_signature=rec["params_signature"],
            use=rec["use"],
            returns=rec["returns"],
            schema_sha=sha,
        )
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        logger.warning("meta_tools_summary: parse failure for %s: %s", name, e)
        return None


async def summarize_tool_async(
    client,
    tool_schema: Dict[str, Any],
    model: str = DEFAULT_MODEL,
) -> Optional[SummaryRecord]:
    """Summarize one tool via Opus. Returns None on parse or API failure."""
    name = _tool_name(tool_schema)
    sha = schema_sha(tool_schema)
    user_msg = (
        "TOOL TO SUMMARIZE:\n```json\n"
        + json.dumps(tool_schema, indent=2)
        + "\n```\n\nReturn the JSON record now."
    )
    try:
        resp = await client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
    except Exception as e:
        logger.warning("meta_tools_summary: API error for %s: %s", name, e)
        return None
    return _parse_response(resp.content[0].text, name, sha)


async def summarize_tools_async(
    tools: List[Dict[str, Any]],
    model: str = DEFAULT_MODEL,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> Dict[str, SummaryRecord]:
    """Batch-summarize tools via :class:`AsyncAnthropic`. Failed tools are omitted."""
    from anthropic import AsyncAnthropic

    client = AsyncAnthropic()
    sem = asyncio.Semaphore(concurrency)

    async def bounded(t: Dict[str, Any]) -> Optional[SummaryRecord]:
        async with sem:
            return await summarize_tool_async(client, t, model=model)

    results = await asyncio.gather(*(bounded(t) for t in tools))
    return {r.name: r for r in results if r is not None}


def summarize_tools(
    tools: List[Dict[str, Any]],
    model: str = DEFAULT_MODEL,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> Dict[str, SummaryRecord]:
    """Sync wrapper around :func:`summarize_tools_async`."""
    return asyncio.run(summarize_tools_async(tools, model=model, concurrency=concurrency))


def render_summary_line(record: SummaryRecord) -> str:
    """Render one tool as the three-line Variant B form used in the system prompt."""
    sig = record.params_signature
    head = f"- {record.name}({sig})" if sig else f"- {record.name}()"
    return f"{head}\n    use: {record.use}\n    out: {record.returns}"
