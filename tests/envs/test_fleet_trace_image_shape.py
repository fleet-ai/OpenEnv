"""Tests for the Fleet trace upload pipeline.

History — read before changing this file:

  A prior version of `envs.fleet_env.trace` hit `/v1/traces/logs` and emitted
  Gemini-shaped parts (`inline_data: {data, mime_type}`). The server accepted
  those uploads but stored screenshots in the PRIVATE bucket
  `s3://theseus-model-traces`, so the Fleet UI rendered <img src=...> tags
  returning HTTP 403 for every screenshot.

The current pipeline uploads each base64 screenshot DIRECTLY to the
public-read bucket `s3://fleet-sessions-images` via boto3, rewrites the
chat history to use the HTTPS URL the bucket exposes, then POSTs the
rewritten history to `/v1/sessions/ingest`. These tests pin that behavior:

  - data URLs get rewritten to HTTPS URLs from the public bucket,
  - text blocks pass through untouched,
  - already-HTTPS image_url blocks pass through (no re-upload, no rewrite),
  - non-multimodal string content passes through,
  - S3 upload failures degrade gracefully (block left alone, no crash),
  - the session ingest payload uses OpenAI multimodal shape (`role`/`content`),
    NOT the old Gemini `history`/`parts` shape.
"""

from __future__ import annotations

import asyncio
import base64
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

from envs.fleet_env.trace import (
    _rewrite_chat_history,
    _s3_key_for,
    _split_data_url,
    upload_trace,
)


# --------------------------------------------------------------------------- #
# data URL parsing
# --------------------------------------------------------------------------- #

class TestSplitDataUrl:
    def test_png_data_url(self):
        # `iVBORw0KGgo` is the literal "PNG header" magic when base64-decoded.
        result = _split_data_url("data:image/png;base64,iVBORw0KGgo=")
        assert result is not None
        image_bytes, mime = result
        assert mime == "image/png"
        assert image_bytes == base64.b64decode("iVBORw0KGgo=")

    def test_jpeg_data_url(self):
        result = _split_data_url("data:image/jpeg;base64,/9j/4AAQ")
        assert result is not None
        _, mime = result
        assert mime == "image/jpeg"

    def test_https_url_returns_none(self):
        # Already-uploaded screenshots must not be re-decoded.
        assert _split_data_url("https://x.s3.amazonaws.com/y.png") is None

    def test_malformed_returns_none(self):
        assert _split_data_url("data:image/png;base64,!!!not-base64!!!") is None


# --------------------------------------------------------------------------- #
# S3 key dedup
# --------------------------------------------------------------------------- #

class TestS3Key:
    def test_identical_bytes_produce_identical_key(self):
        """A second turn that re-screenshots the exact same pixels must
        produce the same S3 key — that is the dedup contract."""
        b = b"identical screenshot bytes"
        assert _s3_key_for(b, "image/png") == _s3_key_for(b, "image/png")

    def test_jpeg_extension(self):
        assert _s3_key_for(b"x", "image/jpeg").endswith(".jpeg")

    def test_png_extension(self):
        assert _s3_key_for(b"x", "image/png").endswith(".png")

    def test_prefix_is_harness_screenshot(self):
        """Keep the prefix stable so ops can grep the bucket by prefix and
        find harness uploads vs. UI-side uploads."""
        assert _s3_key_for(b"x", "image/png").startswith("harness/screenshot/")


# --------------------------------------------------------------------------- #
# Chat history rewrite
# --------------------------------------------------------------------------- #

@pytest.fixture
def fake_upload(monkeypatch):
    """Replace the real boto3 put_object with an in-memory shim that records
    every uploaded payload and returns a deterministic HTTPS URL. This is the
    only realistic way to assert behavior without an S3 round-trip."""
    uploaded: List[Dict[str, Any]] = []

    def _fake(image_bytes: bytes, mime_type: str):
        uploaded.append({"bytes": image_bytes, "mime": mime_type})
        ext = "jpeg" if "jpeg" in mime_type else "png"
        return f"https://fleet-sessions-images.s3.us-east-1.amazonaws.com/fake/{len(uploaded)}.{ext}"

    monkeypatch.setattr(
        "envs.fleet_env.trace._upload_image_to_public_bucket", _fake
    )
    return uploaded


def test_data_url_block_rewritten_to_https(fake_upload):
    history = [
        {
            "role": "tool",
            "content": [
                {"type": "text", "text": "screenshot taken"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQ"},
                },
            ],
        }
    ]
    out = asyncio.run(_rewrite_chat_history(history))
    blocks = out[0]["content"]
    assert blocks[0] == {"type": "text", "text": "screenshot taken"}
    assert blocks[1]["type"] == "image_url"
    assert blocks[1]["image_url"]["url"].startswith(
        "https://fleet-sessions-images.s3.us-east-1.amazonaws.com/"
    )
    # The fake recorded exactly one upload — text blocks must not trigger S3.
    assert len(fake_upload) == 1
    assert fake_upload[0]["mime"] == "image/jpeg"


def test_already_https_url_not_re_uploaded(fake_upload):
    """If a previous turn already swapped to HTTPS (because the same
    screenshot was uploaded earlier in the session), don't double-upload."""
    existing = "https://fleet-sessions-images.s3.us-east-1.amazonaws.com/already/here.png"
    history = [{
        "role": "tool",
        "content": [{"type": "image_url", "image_url": {"url": existing}}],
    }]
    out = asyncio.run(_rewrite_chat_history(history))
    assert out[0]["content"][0]["image_url"]["url"] == existing
    assert fake_upload == []


def test_string_content_passes_through(fake_upload):
    """system/user/assistant messages with plain string content must not be
    walked block-by-block — they have no blocks."""
    history = [
        {"role": "system", "content": "you are a helpful agent"},
        {"role": "assistant", "content": "I'll click the button."},
    ]
    out = asyncio.run(_rewrite_chat_history(history))
    assert out == history
    assert fake_upload == []


def test_s3_upload_failure_leaves_block_alone(monkeypatch):
    """If S3 put fails, log + leave the block alone. The trace upload still
    proceeds; we'd rather have a broken screenshot than crash the rollout."""
    monkeypatch.setattr(
        "envs.fleet_env.trace._upload_image_to_public_bucket",
        lambda b, m: None,
    )
    block = {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="},
    }
    history = [{"role": "tool", "content": [block]}]
    out = asyncio.run(_rewrite_chat_history(history))
    # Block left exactly as it came in — no rewrite, no exception.
    assert out[0]["content"][0] == block


# --------------------------------------------------------------------------- #
# upload_trace integration
# --------------------------------------------------------------------------- #

class _FakeHTTPResponse:
    def __init__(self, payload: Dict[str, Any]):
        self._payload = payload
        self.status_code = 200

    def raise_for_status(self): pass
    def json(self): return self._payload


class _FakeHTTPClient:
    """Captures the POST so tests can assert payload shape."""

    def __init__(self, captured: List[Dict[str, Any]]):
        self._captured = captured

    async def __aenter__(self): return self
    async def __aexit__(self, *args): return False

    async def post(self, url, json=None, headers=None):
        self._captured.append({"url": url, "json": json, "headers": headers})
        return _FakeHTTPResponse({"session_id": "fake-session-id"})


def test_upload_trace_posts_openai_shape_to_sessions_ingest(monkeypatch):
    """The session ingest endpoint reads `messages: [{role, content}]` —
    OpenAI multimodal shape. The earlier `history: [{role, parts}]` Gemini
    shape was for /v1/traces/logs and is incompatible."""
    captured: List[Dict[str, Any]] = []

    monkeypatch.setattr(
        "envs.fleet_env.trace._upload_image_to_public_bucket",
        lambda b, m: "https://fleet-sessions-images.s3.us-east-1.amazonaws.com/x.png",
    )
    monkeypatch.setattr(
        "httpx.AsyncClient", lambda timeout=None: _FakeHTTPClient(captured)
    )

    history = [
        {"role": "system", "content": "you are a helpful agent"},
        {
            "role": "tool",
            "content": [
                {"type": "text", "text": "screenshot"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="},
                },
            ],
        },
    ]

    session_id = asyncio.run(upload_trace(
        api_key="k",
        job_id="job-1",
        task_key="task-1",
        model="kimi-k2.6",
        chat_history=history,
        reward=1.0,
    ))
    assert session_id == "fake-session-id"
    assert len(captured) == 1
    body = captured[0]["json"]

    # Endpoint and auth shape. Session ingest MUST go to the orchestrator
    # host; the platform host does not expose /v1/sessions/ingest.
    assert captured[0]["url"] == (
        "https://orchestrator.fleetai.com/v1/sessions/ingest"
    )
    assert captured[0]["headers"]["Authorization"] == "Bearer k"

    # Payload is OpenAI shape — `messages`, not `history`.
    assert "messages" in body
    assert "history" not in body
    assert "parts" not in str(body)

    # Image block was rewritten to HTTPS before the POST.
    img = body["messages"][1]["content"][1]
    assert img["type"] == "image_url"
    assert img["image_url"]["url"].startswith("https://fleet-sessions-images")
    assert "data:" not in img["image_url"]["url"]

    # Reward survives on metadata.score (the UI groupings expect that key).
    assert body["metadata"]["score"] == 1.0


# --------------------------------------------------------------------------- #
# create_trace_job — host MUST be platform-internal, not orchestrator.
# Regression: an earlier rewrite collapsed both calls to the orchestrator
# host, which 404'd /v1/traces/jobs and silently disabled all session
# uploads (set_trace_config never ran). Pin the right host explicitly.
# --------------------------------------------------------------------------- #

def test_create_trace_job_posts_to_platform_host(monkeypatch):
    """create_trace_job MUST post to the platform host's /v1/traces/jobs,
    NOT to the orchestrator. Both endpoints exist (the orchestrator's
    /v1/jobs is for eval jobs, not trace grouping), but only the platform
    host returns a trace job_id the session ingest endpoint accepts."""
    from envs.fleet_env.trace import create_trace_job

    captured: List[Dict[str, Any]] = []
    monkeypatch.setattr(
        "httpx.AsyncClient", lambda timeout=None: _FakeHTTPClient(captured)
    )
    # _FakeHTTPClient returns {"session_id": ...} by default; we need
    # {"job_id": ...} for this endpoint. Patch the response payload.
    original = _FakeHTTPClient.post

    async def _post_returning_job_id(self, url, json=None, headers=None):
        self._captured.append({"url": url, "json": json, "headers": headers})
        return _FakeHTTPResponse({"job_id": "trace-job-abc"})

    monkeypatch.setattr(_FakeHTTPClient, "post", _post_returning_job_id)
    try:
        job_id = asyncio.run(create_trace_job(api_key="k", name="my-trace-job"))
    finally:
        monkeypatch.setattr(_FakeHTTPClient, "post", original)

    assert job_id == "trace-job-abc"
    assert len(captured) == 1

    # Host MUST be the platform-internal host. Critical regression pin.
    assert captured[0]["url"] == (
        "https://api.internal.fleet-platform.fleetai.com/v1/traces/jobs"
    )
    assert captured[0]["json"] == {"name": "my-trace-job"}
    assert captured[0]["headers"]["Authorization"] == "Bearer k"


def test_create_trace_job_uses_different_host_than_upload(monkeypatch):
    """Explicit pin: create_trace_job and upload_trace target DIFFERENT
    hosts (platform vs orchestrator). The previous bug collapsed them."""
    from envs.fleet_env.trace import (
        _FLEET_SESSION_INGEST_HOST,
        _FLEET_TRACE_JOBS_HOST,
    )
    assert _FLEET_TRACE_JOBS_HOST != _FLEET_SESSION_INGEST_HOST
    assert _FLEET_TRACE_JOBS_HOST == (
        "https://api.internal.fleet-platform.fleetai.com"
    )
    assert _FLEET_SESSION_INGEST_HOST == "https://orchestrator.fleetai.com"


def test_upload_trace_returns_none_on_exception(monkeypatch):
    """Trace upload failures must not propagate — the actual rollout's
    success cannot depend on the trace endpoint being reachable."""

    def _boom(*args, **kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr("httpx.AsyncClient", _boom)
    result = asyncio.run(upload_trace(
        api_key="k", job_id="j", task_key="t",
        model="m", chat_history=[], reward=0.0,
    ))
    assert result is None
