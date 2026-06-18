"""Fleet trace upload utilities for eval rollouts.

Provides functions to create trace jobs and upload conversation traces
to the Fleet API for viewing in the Fleet UI (including screenshots).

The Fleet trace API lives on the internal platform host, NOT the env
orchestrator. /v1/traces/logs binds `TraceLogIngestRequest` and reads each
`parts[*]` as a Gemini `Part`. Only these `Part` fields are inspected:
`text`, `thought`, `inline_data` ({"data", "mime_type"}), `function_call`,
`function_response`. Extra keys (e.g. `type`) are silently ignored — which
is why the previous `{"type":"image","mime_type":...,"data":...}` shape
uploaded fine but the server saw `inline_data=None` and dropped the image
before the S3 sidecar that renders it in the UI ran.

  POST https://api.internal.fleet-platform.fleetai.com/v1/traces/jobs
       Authorization: Bearer <api_key>
       body: {"name": "<job_name>"}
       returns: {"job_id": "...", "name": "...", "status": "completed"}

  POST https://api.internal.fleet-platform.fleetai.com/v1/traces/logs
       Authorization: Bearer <api_key>
       body: {
         "history": [
           {"role": "user", "parts": [{"text": "..."}]},
           {"role": "assistant", "parts": [{"text": "..."}]},
           {"role": "user", "parts": [
              {"inline_data": {"mime_type": "image/png", "data": "<b64>"}},
              {"text": "[Turn 5/64]"},
           ]},
         ],
         "job_id": "<from /v1/traces/jobs>",
         "task_key": "...",
         "model": "...",
         "score": <float>,
         "instance_id": "<optional>",
         "metadata": {...},
       }
       returns: {"success": true, "session_id": "...", ...}

S3 upload of base64 image bytes is handled server-side by `process_content`
(theseus/orchestrator/core/session_content.py) — no client-side presigned-URL
step. Send the base64 in `inline_data.data` directly.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Fleet trace API host (NOT the env orchestrator).
_FLEET_TRACE_HOST = "https://api.internal.fleet-platform.fleetai.com"


def _openai_block_to_gemini_part(block: Dict[str, Any]) -> Dict[str, Any]:
    """Convert one OpenAI content block to a Gemini Part the trace API reads.

    Server reads `part.inline_data.{data,mime_type}` for images and `part.text`
    for text. Anything else is silently ignored.
    """
    btype = block.get("type")
    if btype == "image_url":
        url = block.get("image_url", {}).get("url", "")
        if url.startswith("data:"):
            header, base64_data = url.split(",", 1)
            mime_type = (
                header.split(":")[1].split(";")[0] if ":" in header else "image/png"
            )
            return {"inline_data": {"mime_type": mime_type, "data": base64_data}}
        # Remote URL — server has no fetcher; surface the URL as text so the
        # trace at least carries a clickable reference instead of a black hole.
        return {"text": f"[image url] {url}"}
    if btype == "text":
        return {"text": block.get("text", "")}
    # Unknown block — preserve as text so the trace still uploads.
    return {"text": str(block)}


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

        # Each chat_history entry → one Gemini-shaped history entry. Each
        # OpenAI content block → one Part. Images use `inline_data` (server
        # routes those through process_content for S3 upload + UI rendering);
        # text uses `text`. Server ignores extra keys, so the prior
        # `{"type":"image",...}` shape silently dropped screenshots.
        history = []
        for msg in chat_history:
            content = msg.get("content")
            if isinstance(content, list):
                parts = [
                    _openai_block_to_gemini_part(b) if isinstance(b, dict)
                    else {"text": str(b)}
                    for b in content
                ]
            else:
                parts = [{"text": str(content) if content is not None else ""}]
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
