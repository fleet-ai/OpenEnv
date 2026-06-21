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


def test_turn_footer_block_stripped_from_multimodal_upload(fake_upload):
    """The env appends `{"type":"text","text":"[Turn 3/64]"}` to every
    multimodal observation so the model knows its trajectory position.
    The model SHOULD see it; the Fleet UI should NOT — when the content
    array carries [image_url, footer-text], the dashboard renders both as
    siblings of a JSON tree, nesting the screenshot. Stripping the footer
    at upload time lets the UI render the screenshot as primary content."""
    history = [{
        "role": "tool",
        "content": [
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQ"}},
            {"type": "text", "text": "[Turn 3/64]"},
        ],
    }]
    out = asyncio.run(_rewrite_chat_history(history))
    blocks = out[0]["content"]
    # The footer text block must be gone.
    assert len(blocks) == 1, f"expected only the image_url block, got {blocks}"
    assert blocks[0]["type"] == "image_url"


def test_turn_footer_pattern_matches_variations(fake_upload):
    """Match `[Turn N/M]` with surrounding whitespace and across plausible
    spacing variants. The env strips its own leading newline before adding
    the block, but be robust against future env tweaks."""
    history = [{
        "role": "tool",
        "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw"}},
            {"type": "text", "text": "  [Turn 12/64]  "},   # padded
            {"type": "text", "text": "[Turn 64 / 64]"},      # extra spaces inside
            {"type": "text", "text": "[Turn 1/64]\n"},        # trailing newline
        ],
    }]
    out = asyncio.run(_rewrite_chat_history(history))
    blocks = out[0]["content"]
    assert len(blocks) == 1
    assert blocks[0]["type"] == "image_url"


def test_non_footer_text_block_preserved(fake_upload):
    """Text blocks that AREN'T the turn footer must survive the upload.
    Example: tool result with text + screenshot."""
    history = [{
        "role": "tool",
        "content": [
            {"type": "text", "text": "search returned 12 results"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/4"}},
            {"type": "text", "text": "[Turn 5/64]"},
        ],
    }]
    out = asyncio.run(_rewrite_chat_history(history))
    blocks = out[0]["content"]
    # Footer stripped, search-result text kept.
    assert len(blocks) == 2
    text_blocks = [b for b in blocks if b.get("type") == "text"]
    assert len(text_blocks) == 1
    assert "search returned" in text_blocks[0]["text"]


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


# --------------------------------------------------------------------------- #
# Role mapping: chat_history role:'user' carrying image_url MUST be uploaded
# as role:'tool' with a tool_call_id, because Fleet UI's SessionDetailView
# strips image_url blocks from role:'user' content arrays but preserves them
# for role:'tool'. Verified live with A/B probe on 2026-06-20.
# (incantations skill: trace-upload-with-images)
# --------------------------------------------------------------------------- #

def test_role_mapping_user_with_image_becomes_tool(monkeypatch):
    """Tool observations in chat_history are role:'user' (Kimi chat template
    requirement) but must be uploaded as role:'tool' for the UI renderer."""
    captured: List[Dict[str, Any]] = []
    monkeypatch.setattr(
        "envs.fleet_env.trace._upload_image_to_public_bucket",
        lambda b, m: "https://fleet-sessions-images.s3.us-east-1.amazonaws.com/x.jpeg",
    )
    monkeypatch.setattr(
        "httpx.AsyncClient", lambda timeout=None: _FakeHTTPClient(captured)
    )

    history = [
        {"role": "system", "content": "you are a helpful agent"},
        {"role": "user", "content": "open the page"},
        {"role": "assistant", "content": "I'll click."},
        # Tool observation in chat_history — role:'user' with image_url block
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "[Turn 1/64]"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQ"}},
            ],
        },
    ]

    asyncio.run(upload_trace(
        api_key="k", job_id="j", task_key="t", model="m",
        chat_history=history, reward=1.0,
    ))
    assert len(captured) == 1
    msgs = captured[0]["json"]["messages"]
    # System and user prompts pass through unchanged.
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert msgs[2]["role"] == "assistant"
    # Tool observation: role rewritten to 'tool' + tool_call_id present.
    obs = msgs[3]
    assert obs["role"] == "tool", (
        "image-carrying user message must be uploaded as role:'tool' "
        "so SessionDetailView preserves the image_url content array. "
        "Otherwise the UI strips images and renders text only."
    )
    assert isinstance(obs.get("tool_call_id"), str) and obs["tool_call_id"]
    # Content array preserved end-to-end.
    assert isinstance(obs["content"], list)
    assert any(b.get("type") == "image_url" for b in obs["content"])


def test_role_mapping_carries_assistant_tool_call_id(monkeypatch):
    """When the preceding assistant message has a tool_calls[0].id, the
    rewritten tool message MUST reuse that ID so the renderer can resolve
    the tool name from the assistant turn."""
    captured: List[Dict[str, Any]] = []
    monkeypatch.setattr(
        "envs.fleet_env.trace._upload_image_to_public_bucket",
        lambda b, m: "https://fleet-sessions-images.s3.us-east-1.amazonaws.com/x.jpeg",
    )
    monkeypatch.setattr(
        "httpx.AsyncClient", lambda timeout=None: _FakeHTTPClient(captured)
    )

    history = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": "calling tool",
            "tool_calls": [
                {"id": "call_abc123", "type": "function",
                 "function": {"name": "computer", "arguments": "{}"}}
            ],
        },
        {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw"}}],
        },
    ]
    asyncio.run(upload_trace(
        api_key="k", job_id="j", task_key="t", model="m",
        chat_history=history, reward=0.0,
    ))
    msgs = captured[0]["json"]["messages"]
    tool_msg = next(m for m in msgs if m.get("role") == "tool")
    assert tool_msg["tool_call_id"] == "call_abc123", (
        "tool_call_id must propagate from the preceding assistant.tool_calls[0].id "
        "so the UI can match the tool result back to its call."
    )


def test_role_mapping_does_not_touch_user_text_messages(monkeypatch):
    """Plain text user messages (the initial task prompt) MUST stay
    role:'user' — only image-carrying ones get rewritten."""
    captured: List[Dict[str, Any]] = []
    monkeypatch.setattr(
        "envs.fleet_env.trace._upload_image_to_public_bucket",
        lambda b, m: "https://x/y.png",
    )
    monkeypatch.setattr(
        "httpx.AsyncClient", lambda timeout=None: _FakeHTTPClient(captured)
    )

    history = [
        {"role": "user", "content": "the task prompt"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "follow-up text only, no image"},
    ]
    asyncio.run(upload_trace(
        api_key="k", job_id="j", task_key="t", model="m",
        chat_history=history, reward=0.0,
    ))
    msgs = captured[0]["json"]["messages"]
    # No tool roles — all user/assistant.
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]


def test_role_mapping_preserves_tool_role_when_already_set(monkeypatch):
    """If the env ever DOES write role:'tool' directly, leave it alone
    — don't double-rewrite."""
    captured: List[Dict[str, Any]] = []
    monkeypatch.setattr(
        "envs.fleet_env.trace._upload_image_to_public_bucket",
        lambda b, m: "https://x/y.png",
    )
    monkeypatch.setattr(
        "httpx.AsyncClient", lambda timeout=None: _FakeHTTPClient(captured)
    )
    history = [
        {
            "role": "tool",
            "tool_call_id": "call_existing",
            "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw"}}],
        },
    ]
    asyncio.run(upload_trace(
        api_key="k", job_id="j", task_key="t", model="m",
        chat_history=history, reward=0.0,
    ))
    msgs = captured[0]["json"]["messages"]
    assert msgs[0]["role"] == "tool"
    # NO tool_call_id duplicated/overwritten since the caller already set one.
    # (Implementation passes through; if the caller didn't set one we don't
    # synthesize a new one for already-tool messages.)


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


# --------------------------------------------------------------------------- #
# tool_calls forwarding — Fleet trace viewer linkage
# --------------------------------------------------------------------------- #

def test_upload_trace_forwards_assistant_tool_calls(monkeypatch):
    """The Fleet trace viewer keys off `assistant.tool_calls[0].id`
    matching the next tool message's `tool_call_id` to link a screenshot
    back to the call that produced it. _assemble_messages must preserve
    `tool_calls` on the assistant when forwarding to /v1/sessions/ingest;
    dropping it (which the pre-fix version did) made the viewer show
    Tool Calls=0 and render screenshots only inside the JSON tree.

    A/B verified 2026-06-21 via dummy-job probe: identical session
    content with tool_calls set renders the screenshot top-level; the
    same content without it doesn't."""
    captured: List[Dict[str, Any]] = []
    monkeypatch.setattr(
        "envs.fleet_env.trace._upload_image_to_public_bucket",
        lambda b, m: "https://fleet-sessions-images.s3.us-east-1.amazonaws.com/x.png",
    )
    monkeypatch.setattr(
        "httpx.AsyncClient", lambda timeout=None: _FakeHTTPClient(captured)
    )

    history = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "do a thing"},
        {
            "role": "assistant",
            "content": "calling computer screenshot",
            # The skyrl-fleet env.py:step_async sets this after a
            # successful parse_tool_call. Must round-trip through upload.
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "computer",
                    "arguments": '{"action": "screenshot"}',
                },
            }],
        },
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}},
            ],
        },
    ]

    asyncio.run(upload_trace(
        api_key="k", job_id="job-1", task_key="task-1", model="kimi-k2.6",
        chat_history=history, reward=0.0,
    ))
    msgs = captured[0]["json"]["messages"]

    # Find the assistant message — tool_calls must round-trip verbatim.
    asst = next(m for m in msgs if m.get("role") == "assistant")
    assert "tool_calls" in asst, (
        "assistant tool_calls dropped on upload — viewer will show Tool Calls=0 "
        "and screenshots won't render top-of-session"
    )
    assert asst["tool_calls"][0]["id"] == "call_1"
    assert asst["tool_calls"][0]["function"]["name"] == "computer"

    # The next tool message must bind to the assistant's call_id (this is
    # the linkage that the viewer renders top-level on).
    tool_msg = next(m for m in msgs if m.get("role") == "tool")
    assert tool_msg.get("tool_call_id") == "call_1"


def test_upload_trace_no_tool_calls_field_on_non_assistant(monkeypatch):
    """tool_calls is an assistant-only field. Don't accidentally promote
    it onto system/user/tool messages even if the source dict carries it."""
    captured: List[Dict[str, Any]] = []
    monkeypatch.setattr(
        "envs.fleet_env.trace._upload_image_to_public_bucket",
        lambda b, m: "https://fleet-sessions-images.s3.us-east-1.amazonaws.com/x.png",
    )
    monkeypatch.setattr(
        "httpx.AsyncClient", lambda timeout=None: _FakeHTTPClient(captured)
    )

    history = [
        # Defensive: pretend something set tool_calls on a system message.
        # Must NOT be forwarded — assistant-only contract.
        {"role": "system", "content": "sys", "tool_calls": [{"id": "x"}]},
        {"role": "user", "content": "task"},
    ]
    asyncio.run(upload_trace(
        api_key="k", job_id="j", task_key="t", model="m",
        chat_history=history, reward=0.0,
    ))
    msgs = captured[0]["json"]["messages"]
    sys = next(m for m in msgs if m.get("role") == "system")
    assert "tool_calls" not in sys


def test_upload_trace_no_tool_calls_field_when_assistant_has_none(monkeypatch):
    """Forwarder must NOT add an empty tool_calls list. Empty list could
    still confuse the viewer's linkage; absence is the right signal for
    'this turn had no parseable call'."""
    captured: List[Dict[str, Any]] = []
    monkeypatch.setattr(
        "envs.fleet_env.trace._upload_image_to_public_bucket",
        lambda b, m: "https://fleet-sessions-images.s3.us-east-1.amazonaws.com/x.png",
    )
    monkeypatch.setattr(
        "httpx.AsyncClient", lambda timeout=None: _FakeHTTPClient(captured)
    )

    history = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        # No tool_calls on assistant — parser failed for this turn
        {"role": "assistant", "content": "plain text, no parseable call"},
    ]
    asyncio.run(upload_trace(
        api_key="k", job_id="j", task_key="t", model="m",
        chat_history=history, reward=0.0,
    ))
    asst = next(m for m in captured[0]["json"]["messages"] if m.get("role") == "assistant")
    assert "tool_calls" not in asst
