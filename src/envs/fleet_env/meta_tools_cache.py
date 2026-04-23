"""Filesystem cache for LLM-generated meta-tool summaries.

Cache layout: one JSON file per ``(env_key, env_version)`` under
``meta_tools_summaries/`` next to this module. Each file holds a map of
tool name to :class:`SummaryRecord`.

Entries whose ``schema_sha`` no longer matches the live tool schema are
dropped at load time so the caller can regenerate just the affected tools.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from .meta_tools_summary import SummaryRecord

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent / "meta_tools_summaries"
CACHE_SCHEMA_VERSION = 1


def cache_path(env_key: str, env_version: str) -> Path:
    """Return the on-disk path for an env's summary cache."""
    safe = f"{env_key}.{env_version}".replace("/", "_")
    return CACHE_DIR / f"{safe}.json"


def load_cache(env_key: str, env_version: str) -> Optional[Dict[str, SummaryRecord]]:
    """Load cached records for an env. Returns None if the cache file is absent or malformed."""
    path = cache_path(env_key, env_version)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        logger.warning("meta_tools_cache: invalid JSON at %s: %s", path, e)
        return None
    if data.get("version") != CACHE_SCHEMA_VERSION:
        logger.warning("meta_tools_cache: unsupported version at %s", path)
        return None
    return {name: SummaryRecord.from_dict(rec) for name, rec in data.get("tools", {}).items()}


def save_cache(env_key: str, env_version: str, records: Dict[str, SummaryRecord]) -> Path:
    """Write records to the cache file and return the path written."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = cache_path(env_key, env_version)
    payload: Dict[str, Any] = {
        "version": CACHE_SCHEMA_VERSION,
        "env_key": env_key,
        "env_version": env_version,
        "tools": {name: rec.to_dict() for name, rec in records.items()},
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def filter_stale(
    records: Dict[str, SummaryRecord],
    live_tools_by_name: Dict[str, Dict[str, Any]],
) -> Dict[str, SummaryRecord]:
    """Drop cached records whose ``schema_sha`` no longer matches the live schema."""
    from .meta_tools_summary import schema_sha

    kept: Dict[str, SummaryRecord] = {}
    for name, rec in records.items():
        live = live_tools_by_name.get(name)
        if live is None:
            continue
        if schema_sha(live) != rec.schema_sha:
            logger.info("meta_tools_cache: stale entry for %s (schema changed)", name)
            continue
        kept[name] = rec
    return kept
