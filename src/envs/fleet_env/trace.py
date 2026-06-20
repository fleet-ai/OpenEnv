"""Fleet trace upload utilities for eval rollouts.

Provides functions to create trace jobs and upload conversation traces to
the Fleet API for viewing in the Fleet UI (including screenshots).

## Why this file is the way it is

There are two Fleet API endpoints that can accept session data, and only
one of them stores images where the Fleet UI can render them:

  /v1/traces/logs           → uploads to s3://theseus-model-traces (PRIVATE)
                              UI fetch returns HTTP 403. Image renders as broken.
  /v1/sessions/ingest       → does NOT pre-upload images for you. If you send
                              base64 data URLs inline, the server stores the
                              data URL verbatim and the UI tries to <img src=
                              the data URL — which works only for tiny images
                              and bloats the message store.

The path that actually works end-to-end is:

  1. Client uploads each screenshot directly to
     s3://fleet-sessions-images/harness/screenshot/<sha256>.jpeg
     using its own AWS credentials. This bucket is PUBLIC-READ, so the
     Fleet UI can <img src=https://.../...jpeg> with no auth.
  2. Client rewrites each `data:image/...;base64,...` URL in the chat
     history to the resulting HTTPS URL.
  3. Client POSTs the rewritten chat history to
     `https://orchestrator.fleetai.com/v1/sessions/ingest` as
     `SessionIngestMessage` shape (role + content, OpenAI multimodal blocks).

That mirrors what the legacy skyrl harness did and what nova/import_traces.py
does today (see theseus/nova/import_traces.py:upload_screenshot_to_s3).

## Earlier mistake

A prior version of this file targeted /v1/traces/logs and emitted Gemini
`inline_data` parts. The server accepted the upload and uploaded the
base64 to s3://theseus-model-traces — a private bucket. The UI then could
not render any of the screenshots (403 on every <img>). The current code
abandons that path.
"""

import asyncio
import base64
import hashlib
import logging
import os
import uuid
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# Two Fleet hosts are involved. They share one Supabase job pool — a job_id
# minted on one is recognized by the other and renders correctly under the
# dashboard's /jobs/<id> URL either way — but the create-job and
# session-ingest endpoints live on different services:
#
#   _FLEET_TRACE_JOBS_HOST    : platform host. POST /v1/traces/jobs to mint
#                               a trace job_id. Returns {"job_id","name","status"}.
#   _FLEET_SESSION_INGEST_HOST: orchestrator. POST /v1/sessions/ingest for
#                               per-session uploads (OpenAI message shape,
#                               accepts image_url blocks with HTTPS URLs).
#
# Earlier mistake: collapsed both into a single host (orchestrator), which
# 404'd `POST orchestrator/v1/traces/jobs` and silently disabled all session
# uploads. set_trace_config() never ran because create_trace_job had thrown
# inside an outer try/except, episode_done found no trace config, no
# sessions were posted, training proceeded silently with an empty UI.
# Keep these split.
_FLEET_TRACE_JOBS_HOST = "https://api.internal.fleet-platform.fleetai.com"
_FLEET_SESSION_INGEST_HOST = "https://orchestrator.fleetai.com"

# Public-read bucket the Fleet UI fetches screenshots from. The harness IAM
# user (AWS_ACCESS_KEY_ID) needs s3:PutObject on this prefix.
_SCREENSHOT_BUCKET = "fleet-sessions-images"
_SCREENSHOT_PREFIX = "harness/screenshot"
_SCREENSHOT_REGION = "us-east-1"


def _split_data_url(url: str) -> Optional[Tuple[bytes, str]]:
    """Decode a `data:image/...;base64,...` URL into `(bytes, mime_type)`.

    Returns None for URLs that aren't `data:` (already an HTTPS URL — leave
    them alone) or that fail to decode.
    """
    if not url.startswith("data:"):
        return None
    try:
        header, base64_data = url.split(",", 1)
        mime_type = header.split(":")[1].split(";")[0] if ":" in header else "image/png"
        return base64.b64decode(base64_data), mime_type
    except Exception:
        return None


def _s3_key_for(image_bytes: bytes, mime_type: str) -> str:
    """Deduplicating S3 key. Same image bytes -> same key, so re-uploaded
    screenshots (browser cache hits, screenshot repeats) reuse the upload."""
    ext = "jpeg" if "jpeg" in mime_type or "jpg" in mime_type else "png"
    return f"{_SCREENSHOT_PREFIX}/{hashlib.sha256(image_bytes).hexdigest()[:24]}.{ext}"


def _upload_image_to_public_bucket(image_bytes: bytes, mime_type: str) -> Optional[str]:
    """Best-effort S3 put. Returns the HTTPS URL on success, None on failure.

    boto3 head_object is checked first so a second turn with the same screenshot
    skips a put. Failures here downgrade gracefully — the trace upload still
    proceeds, just with the original data URL (which the UI won't render but
    won't crash on).
    """
    try:
        import boto3
    except Exception as e:
        logger.warning(f"boto3 unavailable for screenshot upload: {e}")
        return None

    key = _s3_key_for(image_bytes, mime_type)
    url = f"https://{_SCREENSHOT_BUCKET}.s3.{_SCREENSHOT_REGION}.amazonaws.com/{key}"
    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        region_name=_SCREENSHOT_REGION,
    )
    try:
        try:
            s3.head_object(Bucket=_SCREENSHOT_BUCKET, Key=key)
            return url  # already exists, dedup hit
        except Exception:
            pass
        s3.put_object(
            Bucket=_SCREENSHOT_BUCKET,
            Key=key,
            Body=image_bytes,
            ContentType=mime_type,
        )
        return url
    except Exception as e:
        logger.warning(f"S3 put_object failed for {key}: {e}")
        return None


def _rewrite_block_with_https_image(block: Dict[str, Any]) -> Dict[str, Any]:
    """If block is an OpenAI `image_url` block carrying a base64 data URL,
    upload it and return the same shape with the URL swapped for the public
    HTTPS one. Other blocks pass through unchanged.

    Sync (boto3 is sync). Called from `to_thread` to avoid blocking the
    upload_trace coroutine.
    """
    if not isinstance(block, dict) or block.get("type") != "image_url":
        return block
    iu = block.get("image_url")
    url = iu.get("url", "") if isinstance(iu, dict) else (iu or "")
    decoded = _split_data_url(url)
    if decoded is None:
        return block  # already an HTTPS URL (or malformed) — leave alone
    image_bytes, mime_type = decoded
    https = _upload_image_to_public_bucket(image_bytes, mime_type)
    if https is None:
        return block
    return {"type": "image_url", "image_url": {"url": https}}


async def _rewrite_chat_history(
    chat_history: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Walk every message; replace base64 image_url blocks with HTTPS URLs.

    S3 puts run in a thread pool so we don't block the event loop. Identical
    screenshots within a session dedup at the S3 key level.
    """
    rewritten: List[Dict[str, Any]] = []
    for msg in chat_history:
        c = msg.get("content")
        if isinstance(c, list):
            new_blocks = await asyncio.gather(*[
                asyncio.to_thread(_rewrite_block_with_https_image, b) for b in c
            ])
            new_msg = dict(msg)
            new_msg["content"] = list(new_blocks)
            rewritten.append(new_msg)
        else:
            rewritten.append(msg)
    return rewritten


async def create_trace_job(api_key: str, name: str) -> str:
    """Create a trace job (a grouping container for sessions) on the
    PLATFORM host. The session ingest endpoint (on the orchestrator) accepts
    the returned job_id because both services share a Supabase job pool.

    Endpoint: POST <platform-host>/v1/traces/jobs
    Returns the new job_id (a UUID string).
    """
    import httpx

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{_FLEET_TRACE_JOBS_HOST}/v1/traces/jobs",
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
    """Upload a conversation trace to Fleet via /v1/sessions/ingest.

    Steps:
      1. Walk the chat_history; for every multimodal image_url block carrying
         a base64 data URL, upload to s3://fleet-sessions-images and rewrite
         to the HTTPS URL (deduplicated by content hash).
      2. POST the rewritten chat_history (still OpenAI multimodal shape) as
         SessionIngestMessage records.

    Returns the new session_id on success, or None on failure (logged).
    Failures here never propagate — a missed trace upload should not abort
    the actual training rollout.
    """
    try:
        import httpx

        rewritten = await _rewrite_chat_history(chat_history)

        # Rename `role:'user'` → `role:'tool'` for messages that carry an
        # `image_url` content block. The Fleet UI's `SessionDetailView.tsx`
        # has two transformers for content arrays: `role:'tool'` JSON-
        # stringifies the array (image_url blocks survive to the renderer),
        # but `role:'user'` runs a "text-only" strip that drops image_url
        # blocks before the renderer sees them. The model's chat_history
        # must keep role:'user' for tool observations because Kimi's chat
        # template binds them that way — rename ONLY at upload time, never
        # in the env's chat_history. Cross-checked with an A/B probe on
        # 2026-06-20 (incantations skill: trace-upload-with-images).
        #
        # The renderer also keys off `tool_call_id` to look up the tool
        # name. We carry the most recent assistant tool_calls[0].id forward
        # if it exists, otherwise synthesize a UUID so the UI has something
        # stable to bind against.
        messages: List[Dict[str, Any]] = []
        pending_tool_call_id: Optional[str] = None
        for msg in rewritten:
            role = msg.get("role", "user")
            content = msg.get("content")
            if role == "assistant":
                tcs = msg.get("tool_calls") or []
                if tcs:
                    try:
                        pending_tool_call_id = tcs[0].get("id") or f"call_{uuid.uuid4().hex[:8]}"
                    except Exception:
                        pending_tool_call_id = f"call_{uuid.uuid4().hex[:8]}"
            carries_image = isinstance(content, list) and any(
                isinstance(b, dict) and b.get("type") == "image_url" for b in content
            )
            if role == "user" and carries_image:
                entry: Dict[str, Any] = {
                    "role": "tool",
                    "content": content,
                    "tool_call_id": pending_tool_call_id or f"call_{uuid.uuid4().hex[:8]}",
                }
                messages.append(entry)
                pending_tool_call_id = None  # consumed by this tool result
            else:
                messages.append({"role": role, "content": content})

        payload: Dict[str, Any] = {
            "messages": messages,
            "job_id": job_id,
            "task_key": task_key,
            "model": model,
            # Fleet UI surfaces session pass/fail in groupings; the field is
            # called "score" on the legacy traces endpoint but the session
            # ingest endpoint reads it from metadata.
            "metadata": dict(metadata or {}, score=reward),
        }
        if instance_id:
            payload["instance_id"] = instance_id
        if verifier_execution_id:
            payload["verifier_execution_id"] = verifier_execution_id

        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                f"{_FLEET_SESSION_INGEST_HOST}/v1/sessions/ingest",
                json=payload,
                headers={"Authorization": f"Bearer {api_key}"},
            )
            response.raise_for_status()
            return response.json().get("session_id")
    except Exception as e:
        logger.warning(f"Failed to upload trace for {task_key}: {e}")
        return None
