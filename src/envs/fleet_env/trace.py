"""Fleet trace upload utilities for eval rollouts.

Provides functions to create trace jobs and upload conversation traces
to the Fleet API for viewing in the Fleet UI (including screenshots).

The Fleet trace API lives on the internal platform host, NOT the env
orchestrator. The previous implementation pointed at
`orchestrator.fleetai.com` (which is where the env-instance API lives) and
authenticated with `X-API-Key` — both wrong, so every trace upload silently
404'd. Verified by hand against the live OpenAPI spec on 2026-06-16:

  POST https://api.internal.fleet-platform.fleetai.com/v1/traces/jobs
       Authorization: Bearer <api_key>
       body: {"name": "<job_name>"}
       returns: {"job_id": "...", "name": "...", "status": "completed"}

  POST https://api.internal.fleet-platform.fleetai.com/v1/traces/logs
       Authorization: Bearer <api_key>
       body: {
         "history": [
           {"role": "user", "parts": [{"type": "text", "text": "..."}]},
           {"role": "assistant", "parts": [{"type": "text", "text": "..."}]},
           ...
         ],
         "job_id": "<from /v1/traces/jobs>",
         "task_key": "...",
         "model": "...",
         "score": <float>,
         "instance_id": "<optional>",
         "metadata": {...},
       }
       returns: {"success": true, "session_id": "...", ...}

The body shape for /v1/traces/logs uses `history` (not `messages`) with each
message holding a `parts` array of typed blocks (not a `content` string).
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Fleet trace API host (NOT the env orchestrator).
_FLEET_TRACE_HOST = "https://api.internal.fleet-platform.fleetai.com"


def _convert_image_block(block: Dict[str, Any]) -> Dict[str, Any]:
    """Convert an OpenAI image_url block to Fleet ingest image format.

    Fleet ingest API expects: {"type": "image", "mime_type": "image/png", "data": "<base64>"}
    It then uploads base64 to S3 and replaces with URL for the UI to render.
    """
    url = block.get("image_url", {}).get("url", "")
    if url.startswith("data:"):
        # data:image/png;base64,ABC... -> extract mime_type and base64 data
        header, base64_data = url.split(",", 1)
        mime_type = header.split(":")[1].split(";")[0] if ":" in header else "image/png"
        return {"type": "image", "mime_type": mime_type, "data": base64_data}
    else:
        # HTTPS URL - pass as text since ingest API expects base64 for images
        return {"type": "text", "text": url}


def _convert_content(content: Any) -> Any:
    """Convert OpenAI-format content blocks to Anthropic format for Fleet UI."""
    if not isinstance(content, list):
        return content
    converted = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "image_url":
            converted.append(_convert_image_block(block))
        else:
            converted.append(block)
    return converted


async def create_trace_job(api_key: str, name: str) -> str:
    """Create a Fleet trace job for grouping eval traces.

    Hits the trace host directly with Bearer auth. The Fleet SDK's
    `AsyncFleet.trace_job` points at the env orchestrator with `X-API-Key`
    which returns 404; that's why every prior caller was silently failing.

    Args:
        api_key: Fleet API key.
        name: Name for the trace job (e.g. "run_name_step_100").

    Returns:
        The job_id string.

    Raises:
        httpx.HTTPStatusError on non-2xx; callers should wrap in try/except.
    """
    import httpx

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{_FLEET_TRACE_HOST}/v1/traces/jobs",
            json={"name": name},
            headers={"Authorization": f"Bearer {api_key}"},
        )
        resp.raise_for_status()
        return resp.json()["job_id"]


async def upload_trace(
    api_key: str,
    job_id: str,
    task_key: str,
    model: str,
    chat_history: List[Dict[str, Any]],
    reward: float,
    instance_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    verifier_execution_id: Optional[str] = None,
) -> Optional[str]:
    """Upload a conversation trace to the Fleet API.

    Converts chat_history (OpenAI message format) to Fleet SessionIngestMessage
    format and ingests it as a trace session.

    Args:
        api_key: Fleet API key.
        job_id: Trace job ID from create_trace_job().
        task_key: Fleet task key.
        model: Model identifier (e.g. model path or name).
        chat_history: List of messages in OpenAI format (system/user/assistant).
            May contain multimodal content with image_url entries.
        reward: Episode reward (>0 = completed, else failed).
        instance_id: Optional Fleet environment instance ID.
        metadata: Optional additional metadata dict.

    Returns:
        The session_id string, or None if upload failed.
    """
    try:
        import httpx

        # Fleet /v1/traces/logs expects each history entry as
        #   {"role": "...", "parts": [{"type": "text"|"image", ...}, ...]}
        # The legacy `messages`/`content` shape was the wrong endpoint's
        # contract. Image entries are emitted as
        #   {"type": "image", "mime_type": "...", "data": "<base64>"}
        # so the UI can render screenshots inline.
        history = []
        for msg in chat_history:
            converted = _convert_content(msg.get("content"))
            if isinstance(converted, list):
                parts = []
                for block in converted:
                    if isinstance(block, dict):
                        if block.get("type") == "image":
                            parts.append(block)
                        elif block.get("type") == "text":
                            parts.append({"type": "text", "text": block.get("text", "")})
                        else:
                            # Unknown block — flatten to text so the trace still uploads.
                            parts.append({"type": "text", "text": str(block)})
                    else:
                        parts.append({"type": "text", "text": str(block)})
            else:
                parts = [{"type": "text", "text": str(converted) if converted is not None else ""}]
            history.append({"role": msg.get("role", "user"), "parts": parts})

        payload: Dict[str, Any] = {
            "history": history,
            "job_id": job_id,
            "task_key": task_key,
            "model": model,
            "score": reward,
        }
        if instance_id:
            payload["instance_id"] = instance_id
        if metadata:
            payload["metadata"] = metadata
        # Without this, Fleet UI ignores the score field in group aggregations.
        if verifier_execution_id:
            payload["verifier_execution_id"] = verifier_execution_id

        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                f"{_FLEET_TRACE_HOST}/v1/traces/logs",
                json=payload,
                headers={"Authorization": f"Bearer {api_key}"},
            )
            response.raise_for_status()
            return response.json().get("session_id")
    except Exception as e:
        logger.warning(f"Failed to upload trace for {task_key}: {e}")
        return None
