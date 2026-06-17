"""Browser lease client for Fleet's managed browser service.

Creates isolated browser instances that navigate to Fleet environment web UIs,
enabling VL models to interact via screenshots + click/type instead of API tools.

Uses the central Fleet browser API (api.internal.fleet-platform.fleetai.com)
which auto-resolves to the best available cluster.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

import httpx

from .fleet_mcp_client import FleetMCPClient

logger = logging.getLogger(__name__)

_BROWSER_API_URL = "https://api.internal.fleet-platform.fleetai.com"

# Additional hosts the browser is allowed to reach (tile servers, telemetry, etc.)
_ADDITIONAL_ALLOWED_HOSTS = (
    "*.amazonaws.com",
    "*.basemaps.cartocdn.com",
    "*.tile.openstreetmap.org",
    "api.instance-telemetry.fleet-platform.fleetai.com",
    "tileserver.staging.fleetai.com",
)

# Healthcheck constants (mirrors theseus orchestrator)
_NAVIGATE_SETTLE_SECONDS = 5
_SCREENSHOT_MAX_ATTEMPTS = 3
_SCREENSHOT_RETRY_SECONDS = 3
_SCREENSHOT_MIN_BYTES = 8192


@dataclass
class BrowserLeaseResult:
    lease_id: str
    browser_id: str
    mcp_url: str
    cdp_url: str
    stream_url: Optional[str]
    host_domain: str


def _resolve_api_key() -> str:
    """Resolve browser API key from environment."""
    for var in ("FLEET_TEAM_API_KEY", "BROWSER_API_TOKEN", "FLEET_API_KEY"):
        val = os.getenv(var, "").strip()
        if val:
            return val
    raise RuntimeError(
        "No browser API key found. Set FLEET_TEAM_API_KEY, BROWSER_API_TOKEN, or FLEET_API_KEY."
    )


def _allowed_hosts(instance_host: str) -> list[str]:
    hosts = {instance_host.strip().lower(), *_ADDITIONAL_ALLOWED_HOSTS}
    return sorted(hosts)


async def create_browser_lease(
    instance_url: str,
    ttl_seconds: int,
) -> BrowserLeaseResult:
    """Create a browser lease, navigate to the instance URL, and healthcheck."""
    instance_host = urlparse(instance_url).hostname
    if not instance_host:
        raise RuntimeError(f"No hostname in instance URL: {instance_url}")

    api_url = os.getenv("BROWSER_API_BASE_URL", _BROWSER_API_URL).strip()
    api_key = _resolve_api_key()
    allowed = _allowed_hosts(instance_host)

    logger.info(f"Creating browser lease: instance={instance_host}, ttl={ttl_seconds}s")

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{api_url}/v1/browser",
            json={"ttl_seconds": ttl_seconds, "allowed_hosts": allowed},
            headers={"Authorization": f"Bearer {api_key}"},
        )
        resp.raise_for_status()
        data = resp.json()

    lease = BrowserLeaseResult(
        lease_id=data["lease_id"],
        browser_id=data["browser_id"],
        mcp_url=str(data["mcp_url"]),
        cdp_url=str(data["cdp_url"]),
        stream_url=str(data.get("stream_url", "")),
        host_domain=data["host_domain"],
    )
    logger.info(
        f"Browser lease created: lease_id={lease.lease_id}, "
        f"browser_id={lease.browser_id}, mcp_url={lease.mcp_url}"
    )

    # Navigate browser to instance URL and healthcheck
    try:
        await _navigate_and_healthcheck(
            mcp_url=lease.mcp_url,
            target_url=instance_url,
            api_key=api_key,
        )
    except Exception as e:
        logger.warning(f"Browser healthcheck failed, deleting lease: {e}")
        await delete_browser_lease(lease.lease_id)
        raise

    return lease


async def _navigate_and_healthcheck(
    mcp_url: str, target_url: str, api_key: str
) -> None:
    """Navigate browser to target URL and verify via screenshot."""
    mcp = FleetMCPClient(url=mcp_url, api_key=api_key)

    # Pre-navigation settle
    await asyncio.sleep(_NAVIGATE_SETTLE_SECONDS)

    # Navigate
    result = await mcp.call_tool("computer", {"action": "navigate", "url": target_url})
    if isinstance(result, dict) and "error" in result:
        raise RuntimeError(f"Browser navigate failed: {result['error']}")
    logger.info(f"Browser navigated to {target_url}")

    # Post-navigation settle
    await asyncio.sleep(_NAVIGATE_SETTLE_SECONDS)

    # Screenshot healthcheck with retries
    for attempt in range(1, _SCREENSHOT_MAX_ATTEMPTS + 1):
        try:
            screenshot = await mcp.call_tool(
                "computer", {"action": "screenshot"}
            )
            if _validate_screenshot(screenshot):
                logger.info(f"Browser healthcheck passed (attempt {attempt})")
                return
            logger.warning(
                f"Screenshot validation failed (attempt {attempt}/{_SCREENSHOT_MAX_ATTEMPTS})"
            )
        except Exception as e:
            logger.warning(
                f"Screenshot failed (attempt {attempt}/{_SCREENSHOT_MAX_ATTEMPTS}): {e}"
            )
        if attempt < _SCREENSHOT_MAX_ATTEMPTS:
            await asyncio.sleep(_SCREENSHOT_RETRY_SECONDS)

    raise RuntimeError(
        f"Browser healthcheck failed after {_SCREENSHOT_MAX_ATTEMPTS} attempts"
    )


def _validate_screenshot(result) -> bool:
    """Check screenshot is non-trivial (not blank/error)."""
    if isinstance(result, dict) and "error" in result:
        return False
    if isinstance(result, list):
        for item in result:
            if isinstance(item, dict) and item.get("type") == "image_url":
                data_url = item.get("image_url", {}).get("url", "")
                if ";base64," in data_url:
                    base64_data = data_url.split(";base64,", 1)[1]
                    if len(base64_data) >= _SCREENSHOT_MIN_BYTES:
                        return True
    return False


async def delete_browser_lease(lease_id: str) -> None:
    """Delete a browser lease. Best-effort, swallows errors."""
    try:
        api_url = os.getenv("BROWSER_API_BASE_URL", _BROWSER_API_URL).strip()
        api_key = _resolve_api_key()
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.delete(
                f"{api_url}/v1/browser/{lease_id}",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            logger.info(f"Browser lease deleted: lease_id={lease_id}, status={resp.status_code}")
    except Exception as e:
        logger.warning(f"Failed to delete browser lease {lease_id}: {e}")
