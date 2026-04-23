#!/usr/bin/env python3
"""Generate LLM summaries for a Fleet env's tools and cache them.

Two input modes:

* ``--tools-json path.json``: read a list of tool schemas (OpenAI function-calling
  format) from a local JSON file. Useful for smoke tests and offline runs.
* ``--env-key X --env-version Y``: fetch live tools from the Fleet API (requires
  FLEET_API_KEY and the ``fleet-sdk`` dependency).

The output is written to
``OpenEnv/src/envs/fleet_env/meta_tools_summaries/<env_key>.<env_version>.json``
and is loaded at :class:`ToolIndex` init to render the richer three-line
summary form.

Stale entries (``schema_sha`` changed) are regenerated; existing matching
entries are kept unless ``--force`` is passed.

Usage:
    python scripts/generate_meta_tool_summaries.py --tools-json reddit_tools.json --env-key reddit --env-version v0.0.98
    python scripts/generate_meta_tool_summaries.py --env-key reddit --env-version v0.0.98  # live fetch
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Load the fleet_env package WITHOUT triggering the package's __init__ (which
# pulls Fleet SDK and other heavy deps). We create a lightweight namespace
# package entry for ``fleet_env_light`` that exposes just the two leaf modules
# we need.
import importlib.util
import types

_MOD_DIR = Path(__file__).parent.parent / "src" / "envs" / "fleet_env"
_PKG_NAME = "fleet_env_light"


def _install_light_package() -> None:
    """Register a bare package stub and load the two leaf modules into it."""
    pkg = types.ModuleType(_PKG_NAME)
    pkg.__path__ = [str(_MOD_DIR)]
    sys.modules[_PKG_NAME] = pkg

    for mod_file in ("meta_tools_summary", "meta_tools_cache"):
        full_name = f"{_PKG_NAME}.{mod_file}"
        spec = importlib.util.spec_from_file_location(full_name, _MOD_DIR / f"{mod_file}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[full_name] = module
        spec.loader.exec_module(module)
        setattr(pkg, mod_file, module)


_install_light_package()

_summary_mod = sys.modules[f"{_PKG_NAME}.meta_tools_summary"]
_cache_mod = sys.modules[f"{_PKG_NAME}.meta_tools_cache"]

DEFAULT_CONCURRENCY = _summary_mod.DEFAULT_CONCURRENCY
DEFAULT_MODEL = _summary_mod.DEFAULT_MODEL
summarize_tools_async = _summary_mod.summarize_tools_async
load_cache = _cache_mod.load_cache
save_cache = _cache_mod.save_cache
filter_stale = _cache_mod.filter_stale

logger = logging.getLogger(__name__)


def load_tools_from_file(path: Path) -> List[Dict[str, Any]]:
    """Load a list of tool schemas from a JSON file."""
    return json.loads(path.read_text())


def fetch_tools_live(env_key: str, env_version: str) -> List[Dict[str, Any]]:
    """Fetch live tool schemas from Fleet for the given env pair."""
    # fleet-sdk is an optional dep; import inside so --tools-json mode works without it
    try:
        from fleet import FleetEnvClient  # type: ignore[import-not-found]
    except ImportError as e:
        raise RuntimeError(
            "fleet-sdk is required for --env-key/--env-version mode; install it or pass --tools-json"
        ) from e
    client = FleetEnvClient(env_id=env_key, version=env_version)
    return client.list_tools()


async def generate(
    tools: List[Dict[str, Any]],
    env_key: str,
    env_version: str,
    model: str,
    concurrency: int,
    force: bool,
) -> Dict[str, Any]:
    """Run the generation loop with cache reuse. Returns a summary dict."""
    live_by_name = {t.get("function", t).get("name", ""): t for t in tools}

    existing = load_cache(env_key, env_version) if not force else None
    if existing:
        existing = filter_stale(existing, live_by_name)
        missing = [t for t in tools if live_by_name.get(t.get("function", t).get("name", "")) is not None
                   and t.get("function", t).get("name", "") not in existing]
        logger.info(
            "Cache hit for %d tools, generating %d new/stale",
            len(existing), len(missing),
        )
    else:
        missing = list(tools)
        existing = {}
        logger.info("No cache, generating %d tools", len(missing))

    t0 = time.time()
    new_records = await summarize_tools_async(missing, model=model, concurrency=concurrency)
    elapsed = time.time() - t0

    records = {**existing, **new_records}
    path = save_cache(env_key, env_version, records)

    return {
        "cache_path": str(path),
        "total": len(records),
        "generated": len(new_records),
        "reused": len(existing),
        "failed": len(missing) - len(new_records),
        "elapsed_s": round(elapsed, 1),
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-key", required=True)
    parser.add_argument("--env-version", required=True)
    parser.add_argument("--tools-json", type=Path, help="Load tools from a JSON file instead of Fleet")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--force", action="store_true", help="Regenerate every tool, ignoring cache")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.tools_json:
        tools = load_tools_from_file(args.tools_json)
    else:
        tools = fetch_tools_live(args.env_key, args.env_version)

    logger.info("Input tools: %d", len(tools))
    summary = asyncio.run(
        generate(tools, args.env_key, args.env_version, args.model, args.concurrency, args.force)
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
